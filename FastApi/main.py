"""
main.py
────────
Flood Prediction API — serves the champion model + preprocessing pipeline
from the MLflow Model Registry (via DagsHub tracking).

Changes from the original version:
  - Model/pipeline loading moved into a `lifespan` handler with a
    stage -> latest-version fallback chain, instead of hardcoded version
    numbers ("/2", "/1") at import time. Promoting a new model no longer
    requires editing this file.
  - /predict now takes a JSON body (Pydantic model) instead of 17 query
    params on a POST — standard REST shape, automatic validation, and it
    matches what Swagger/clients expect from a POST endpoint.
  - Added /health and /model/reload for operability.
  - Deduplicated imports; allow_origins is now a proper list.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional

import mlflow
import mlflow.spark
import sentry_sdk
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession

# ── Environment setup (must happen before Spark/MLflow are touched) ────────
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["MLFLOW_TMP_DIR"] = os.path.join(tempfile.gettempdir(), "mlflow")

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    send_default_pii=True,
)

MODEL_NAME = "Champion"
PIPELINE_NAME = "Flood Prediction Preprocessing"
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

FEATURE_COLUMNS = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
    "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "WetlandLoss",
]

# ── Shared Spark session ────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("FastApi-Spark")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.warehouse.dir", "file:///C:/temp/spark-warehouse")
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    .config("spark.hadoop.fs.AbstractFileSystem.file.impl", "org.apache.hadoop.fs.local.RawLocalFs")
    .getOrCreate()
)

# ── Global model state ──────────────────────────────────────────────────
model = None
pipeline_model = None
model_source: Optional[str] = None
pipeline_source: Optional[str] = None


def _resolve_registered_uri(name: str, stage: str) -> List[str]:
    """Stage -> latest-version fallback chain for a registered model name."""
    candidates = [f"models:/{name}/{stage}"]
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{name}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            candidates.append(f"models:/{name}/{latest.version}")
    except Exception:
        pass
    return candidates


def load_champion_and_pipeline():
    """Loads both artifacts with a stage -> latest fallback. Never raises —
    logs and leaves globals as None so the API can start in a degraded state
    instead of crashing outright if the registry is briefly unreachable."""
    global model, pipeline_model, model_source, pipeline_source

    for uri in _resolve_registered_uri(MODEL_NAME, MODEL_STAGE):
        try:
            model = mlflow.spark.load_model(model_uri=uri)
            model_source = uri
            break
        except Exception as exc:  # noqa: BLE001
            sentry_sdk.capture_message(f"Failed loading model from {uri}: {exc}")

    for uri in _resolve_registered_uri(PIPELINE_NAME, MODEL_STAGE):
        try:
            pipeline_model = mlflow.spark.load_model(model_uri=uri)
            pipeline_source = uri
            break
        except Exception as exc:  # noqa: BLE001
            sentry_sdk.capture_message(f"Failed loading pipeline from {uri}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_champion_and_pipeline()
    yield
    spark.stop()


app = FastAPI(
    title="Welcome to Flood Prediction API",
    version="2.0.0",
    description="Flood Prediction using Spark MLlib + MLflow",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────
class FloodFeatures(BaseModel):
    MonsoonIntensity: float = Field(..., example=1)
    TopographyDrainage: float = Field(..., example=3)
    RiverManagement: float = Field(..., example=4)
    Deforestation: float = Field(..., example=5)
    Urbanization: float = Field(..., example=3)
    ClimateChange: float = Field(..., example=2)
    DamsQuality: float = Field(..., example=8)
    Siltation: float = Field(..., example=9)
    AgriculturalPractices: float = Field(..., example=3)
    Encroachments: float = Field(..., example=2)
    IneffectiveDisasterPreparedness: float = Field(..., example=9)
    DrainageSystems: float = Field(..., example=8)
    CoastalVulnerability: float = Field(..., example=6)
    Landslides: float = Field(..., example=2)
    Watersheds: float = Field(..., example=1)
    DeterioratingInfrastructure: float = Field(..., example=1)
    WetlandLoss: float = Field(..., example=4)


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Flood Prediction API Running"}


@app.get("/health")
def health():
    return {
        "status": "ok" if (model and pipeline_model) else "degraded",
        "model_loaded": model is not None,
        "model_source": model_source,
        "pipeline_loaded": pipeline_model is not None,
        "pipeline_source": pipeline_source,
    }


@app.post("/model/reload")
def reload_model():
    load_champion_and_pipeline()
    if not (model and pipeline_model):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Reload attempted but model and/or pipeline still not loaded — check logs/Sentry.",
        )
    return {"status": "ok", "model_source": model_source, "pipeline_source": pipeline_source}


@app.get("/tester")
def tester():
    return {"message": "tester endpoint is working fine"}


@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0


@app.post("/predict")
def predict(payload: FloodFeatures):
    if model is None or pipeline_model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Model or preprocessing pipeline not loaded. Try POST /model/reload.",
        )

    try:
        row = [tuple(getattr(payload, col) for col in FEATURE_COLUMNS)]
        input_df = spark.createDataFrame(row, FEATURE_COLUMNS)

        transformed_df = pipeline_model.transform(input_df)
        prediction_df = model.transform(transformed_df)
        prediction = prediction_df.select("prediction").collect()[0][0]

        return {
            "flood_probability": round(float(prediction), 4),
            "model_name": MODEL_NAME,
            "model_source": model_source,
            "status": "success",
        }

    except Exception as e:
        sentry_sdk.capture_exception(e)
        return {"status": "failed", "error": str(e)}