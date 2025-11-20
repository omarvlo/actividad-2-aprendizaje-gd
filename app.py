import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import pickle
from google.cloud import storage
from river import linear_model, preprocessing, metrics

# =========================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea desde GCS", page_icon="")
st.title("Aprendizaje en línea con River (Cloud Storage + Cloud Run)")

st.markdown("""
Esta aplicación lee archivos **CSV desde un bucket de Google Cloud Storage**,  
uno por uno, para entrenar un modelo de **aprendizaje incremental** con River.
""")

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def _parse_time_fields(row):
    for c in ("pickup_datetime", "tpep_pickup_datetime", "lpep_pickup_datetime"):
        if c in row and pd.notna(row[c]):
            dt = pd.to_datetime(row[c], errors="coerce", utc=False)
            if pd.notna(dt):
                return dt, dt.hour
    return None, 0

def _extract_x(row):
    dist = float(pd.to_numeric(row.get("trip_distance", 0), errors="coerce") or 0)
    psg = float(pd.to_numeric(row.get("passenger_count", 0), errors="coerce") or 0)

    dt, hour = _parse_time_fields(row)
    dow = int(dt.weekday()) if isinstance(dt, pd.Timestamp) else 0
    is_weekend = 1 if dow >= 5 else 0

    return {
        "dist": dist,
        "log_dist": float(np.log1p(max(dist, 0))),
        "pass": psg,
        "hour": float(hour),
        "dow": float(dow),
        "is_weekend": float(is_weekend),
    }

def _valid_target(v):
    y = pd.to_numeric(v, errors="coerce")
    if pd.isna(y) or not np.isfinite(y):
        return None
    return float(y)

# =========================================================
# MANEJO DEL MODELO EN GCS
# =========================================================
def save_model_to_gcs(model, bucket_name, blob_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado en gs://{bucket_name}/{blob_name}")
    except Exception as e:
        st.error(f"No se pudo guardar el modelo: {e}")

def load_model_from_gcs(bucket_name, blob_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if blob.exists():
            data = blob.download_as_bytes()
            st.info("Modelo cargado desde GCS.")
            return pickle.loads(data)
        return None
    except Exception as e:
        st.warning(f"No se pudo cargar modelo previo: {e}")
        return None

# =========================================================
# PARÁMETROS DE LA INTERFAZ
# =========================================================
bucket_name = st.text_input("Bucket de GCS:", "bucket_131025")
prefix = st.text_input("Prefijo/carpeta dentro del bucket:", "tlc_yellow_trips_2022/")
limite = st.number_input("Máximo de filas por archivo:", 1000, 100000, step=500)

MODEL_PATH = "models/model_incremental.pkl"

# =========================================================
# INICIALIZACIÓN DEL MODELO
# =========================================================
if "model" not in st.session_state:
    model = load_model_from_gcs(bucket_name, MODEL_PATH)
    if model is None:
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()

    st.session_state.model = model
    st.session_state.metric = metrics.R2()
    st.session_state.history = []
    st.session_state.file_index = 0   # <-- Aquí controlamos archivo por archivo

model = st.session_state.model
r2 = st.session_state.metric

# =========================================================
# LECTURA DE ARCHIVOS DESDE EL BUCKET (UNO POR CLIC)
# =========================================================
def get_blobs(bucket_name, prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return sorted(list(bucket.list_blobs(prefix=prefix)), key=lambda b: b.name)

def process_single_blob(blob, limite=1000, chunksize=500):
    st.write(f"Procesando archivo: `{blob.name}`")

    content = blob.download_as_bytes()
    buffer = io.BytesIO(content)

    count = 0
    for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):

        # Validación mínima
        if not {"trip_distance", "passenger_count", "fare_amount"}.issubset(chunk.columns):
            continue

        for col in ["trip_distance", "passenger_count", "fare_amount"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        chunk = chunk[
            chunk["fare_amount"].between(2, 200)
            & chunk["trip_distance"].between(0.1, 50)
            & chunk["passenger_count"].between(1, 6)
        ]

        if chunk.empty:
            continue

        for _, row in chunk.iterrows():
            if count >= limite:
                break

            y = _valid_target(row["fare_amount"])
            if y is None:
                continue

            x = _extract_x(row)

            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            r2.update(y, y_pred)

            count += 1

        if count >= limite:
            break

    return r2.get()

# =========================================================
# BOTÓN: PROCESAR SIGUIENTE ARCHIVO
# =========================================================
if st.button("Procesar siguiente archivo del bucket"):
    blobs = get_blobs(bucket_name, prefix)

    if st.session_state.file_index >= len(blobs):
        st.warning("Ya no hay más archivos por procesar.")
    else:
        blob = blobs[st.session_state.file_index]
        score = process_single_blob(blob, limite)

        st.session_state.history.append(score)
        st.session_state.file_index += 1

        st.success(f"R² acumulado tras `{blob.name}`: **{score:.3f}**")

        save_model_to_gcs(model, bucket_name, MODEL_PATH)

# =========================================================
# ESTADO DEL MODELO
# =========================================================
st.markdown("---")
st.subheader("Estado del modelo")
st.write(f"R² actual: **{r2.get():.3f}**")

if st.session_state.history:
    st.line_chart(st.session_state.history, height=200)

st.caption("Cloud Run + River + GCS — Aprendizaje incremental real")


st.caption("Cloud Run + River • Dataset público de taxis NYC (2022)")

