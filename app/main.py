from fastapi import FastAPI, UploadFile, File, Request
import tensorflow as tf
import numpy as np
from io import BytesIO
import time
import logging

from src.preprocess import load_and_resize, batch_stack

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------
# Metrics counters
# -----------------------
REQUEST_COUNT = 0
TOTAL_LATENCY = 0.0

# -----------------------
# App + model load
# -----------------------
app = FastAPI()
model = tf.keras.models.load_model("model.h5")


# -----------------------
# Middleware for logging
# -----------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):

    global REQUEST_COUNT, TOTAL_LATENCY

    start_time = time.time()
    REQUEST_COUNT += 1

    logger.info(f"Incoming request: {request.method} {request.url.path}")

    response = await call_next(request)

    latency = time.time() - start_time
    TOTAL_LATENCY += latency

    logger.info(f"Completed {request.url.path} "
                f"status={response.status_code} "
                f"latency={latency:.3f}s")

    return response


# -----------------------
# Health endpoint
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        img = load_and_resize(BytesIO(contents))
        batch = batch_stack([img])

        prob = float(model.predict(batch)[0][0])
        label = "dog" if prob > 0.5 else "cat"

        logger.info(f"Prediction generated: label={label}, prob={prob:.3f}")

        return {
            "probability": prob,
            "label": label
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": "prediction failed"}


# -----------------------
# Metrics endpoint
# -----------------------
@app.get("/metrics")
def metrics():

    avg_latency = (
        TOTAL_LATENCY / REQUEST_COUNT if REQUEST_COUNT > 0 else 0
    )

    return {
        "request_count": REQUEST_COUNT,
        "average_latency_seconds": round(avg_latency, 4)
    }