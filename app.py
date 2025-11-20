import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from google.cloud import storage
from river import linear_model, preprocessing, metrics

# =========================================================
# CONFIG STREAMLIT
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea con River", page_icon="")
st.title("Aprendizaje en línea con River (Cloud Storage + Cloud Run)")

st.markdown("""
Esta aplicación lee archivos CSV desde un bucket de Google Cloud Storage,
uno por uno, para entrenar un modelo de **aprendizaje incremental** con River.
""")

# =========================================================
# GUARDAR / CARGAR MODELO DESDE GCS
# =========================================================
def save_model_to_gcs(model, bucket_name, destination_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado en gs://{bucket_name}/{destination_blob}")
    except Exception as e:
        st.warning(f"No se pudo guardar el modelo: {e}")

def load_model_from_gcs(bucket_name, source_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        if blob.exists():
            data = blob.download_as_bytes()
            st.info("Modelo cargado desde GCS.")
            return pickle.loads(data)
        else:
            st.info("No se encontró modelo previo. Se creará uno nuevo.")
            return None
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo: {e}")
        return None

# =========================================================
# PANEL DE PARÁMETROS
# =========================================================
bucket_name = st.text_input("Bucket GCS:", "bucket_131025")
prefix = st.text_input("Prefijo / carpeta:", "tlc_yellow_trips_2022/")
limite = st.number_input("Filas por archivo:", value=1000, step=100)
mostrar_grafico = st.checkbox("Mostrar gráfica del R²", value=True)

MODEL_PATH = "models/model_incremental.pkl"

# =========================================================
# INICIALIZACIÓN DEL MODELO EN SESIÓN
# =========================================================
if "model" not in st.session_state:
    model = load_model_from_gcs(bucket_name, MODEL_PATH)
    if model is None:
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()

    st.session_state.model = model
    st.session_state.metric = metrics.R2()
    st.session_state.history = []

model = st.session_state.model
metric = st.session_state.metric

# =========================================================
# PARSER ROBUSTO DE TIEMPO
# =========================================================
def _parse_time_fields(row):
    TS_COLS = [
        "pickup_datetime",
        "tpep_pickup_datetime",
        "lpep_pickup_datetime",
        "pickup_hour"
    ]

    for col in TS_COLS:
        val = row.get(col)
        if val is None:
            continue

        if col == "pickup_hour":
            hour = pd.to_numeric(val, errors="coerce")
            if pd.notna(hour) and 0 <= hour <= 23:
                return None, int(hour), 0, 0.0

        dt = pd.to_datetime(val, errors="coerce", utc=False)
        if pd.notna(dt):
            hour = int(dt.hour)
            dow = int(dt.weekday())
            is_weekend = 1.0 if dow >= 5 else 0.0
            return dt, hour, dow, is_weekend

    return None, 0, 0, 0.0

# =========================================================
# EXTRACTOR ROBUSTO DE FEATURES
# =========================================================
def _extract_x(row):
    # distancia
    dist = pd.to_numeric(row.get("trip_distance"), errors="coerce")
    if pd.isna(dist) or dist <= 0 or dist > 200:
        return None
    dist = float(dist)

    # pasajeros
    psg = pd.to_numeric(row.get("passenger_count"), errors="coerce")
    if pd.isna(psg) or psg < 1 or psg > 6:
        return None
    psg = float(psg)

    # log-dist
    log_dist = float(np.log1p(dist))

    # tiempo
    dt, hour, dow, is_weekend = _parse_time_fields(row)

    return {
        "dist": dist,
        "log_dist": log_dist,
        "pass": psg,
        "hour": float(hour),
        "dow": float(dow),
        "is_weekend": is_weekend,
    }

# =========================================================
# VALIDACIÓN DEL TARGET
# =========================================================
def _valid_target(v):
    y = pd.to_numeric(v, errors="coerce")
    if pd.isna(y):
        return None
    y = float(y)
    if y <= 0 or y > 200 or not np.isfinite(y):
        return None
    return y

# =========================================================
# LECTURA INCREMENTAL DESDE GCS
# =========================================================
def stream_from_bucket(bucket_name, prefix, limite=1000, chunksize=500):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    st.info(f"Archivos encontrados: {len(blobs)}")

    for idx, blob in enumerate(blobs, start=1):
        st.write(f"Procesando archivo {idx}/{len(blobs)}: {blob.name.split('/')[-1]}")

        try:
            content = blob.download_as_bytes()
            buffer = io.BytesIO(content)
            count = 0

            for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):
                if not {"trip_distance", "passenger_count", "fare_amount"}.issubset(chunk.columns):
                    continue

                # Limpieza mínima
                for col in ["trip_distance", "passenger_count", "fare_amount"]:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

                chunk = chunk.dropna(subset=["trip_distance", "passenger_count", "fare_amount"])

                for _, row in chunk.iterrows():
                    if count >= limite:
                        break

                    y = _valid_target(row.get("fare_amount"))
                    if y is None:
                        continue

                    x = _extract_x(row)
                    if x is None:
                        continue

                    y_pred = model.predict_one(x)
                    model.learn_one(x, y)
                    metric.update(y, y_pred)
                    count += 1

                if count >= limite:
                    break

        except Exception as e:
            st.warning(f"Error en {blob.name}: {e}")
            continue

        yield blob.name, metric.get()

# =========================================================
# BOTÓN PRINCIPAL
# =========================================================
if st.button("Actualizar modelo con datos del bucket"):
    st.info("Procesando archivos...")

    blobs = list(storage.Client().bucket(bucket_name).list_blobs(prefix=prefix))
    total = len(blobs) if blobs else 1

    progreso = st.progress(0)
    valores = []

    for i, (fname, score) in enumerate(stream_from_bucket(bucket_name, prefix, limite)):
        st.session_state.history.append(score)
        valores.append(score)

        progreso.progress(min((i + 1) / total, 1.0))
        st.write(f"{fname} — R² acumulado: **{score:.3f}**")

    progreso.empty()
    save_model_to_gcs(model, bucket_name, MODEL_PATH)
    st.success("Entrenamiento finalizado.")

    if mostrar_grafico and valores:
        st.line_chart(valores)

# =========================================================
# ESTADO FINAL
# =========================================================
st.markdown("---")
st.subheader("Estado actual del modelo")
st.write(f"R² actual: **{metric.get():.3f}**")

if st.session_state.history:
    st.line_chart(st.session_state.history)

st.caption("Cloud Run + River • Dataset público de taxis NYC (2022)")


