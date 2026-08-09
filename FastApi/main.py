# =============================================================================
# app.py  —  Flood Prediction FastAPI Service
# =============================================================================

import dagshub
from dotenv import load_dotenv
load_dotenv()
import math
import sys
import os
import json
import os
import mlflow
import numpy as np


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass



# Set MLflow tracking URI directly from env variable
mlflow.set_tracking_uri(os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/Muhammad-Musharraf/Mlops-Flood-Prediction.mlflow"
))

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ── Standard Library ──────────────────────────────────────────────────────────
import logging
from contextlib import asynccontextmanager

# ── Third Party ───────────────────────────────────────────────────────────────
import yaml

# ── Redis ─────────────────────────────────────────────────────────────────────
import redis

# ── Rate Limiting ─────────────────────────────────────────────────────────────
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── PySpark ───────────────────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level    = logging.INFO,
    format   = "[ %(asctime)s ] %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    encoding = "utf-8",
)
logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

# FastApi/app.py  →  two dirname() calls  →  project root
PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_latest")
PARAMS_PATH       = os.path.join(PROJECT_ROOT, "params.yaml")

# =============================================================================
# READ params.yaml — single source of truth for features
# =============================================================================

with open(PARAMS_PATH, "r") as f:
    _params = yaml.safe_load(f)

FEATURE_COLS  = _params["data"]["feature_columns"]   # 17 features
TARGET_COL    = _params["data"]["target_column"]      # FloodProbability
SAMPLE_INPUT  = _params["sample_input"]               # from params.yaml

logger.info("Loaded %d features from params.yaml", len(FEATURE_COLS))
logger.info("Features : %s", FEATURE_COLS)




# =============================================================================
# GLOBAL SINGLETONS
# =============================================================================

spark_session     = None
flood_model       = None
feature_assembler = None
redis_client      = None
native_model      = None   # ← extracted model weights for fast inference

limiter = Limiter(
    key_func     = get_remote_address,
    storage_uri  = os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
)

# =============================================================================
# SPARK SESSION
# =============================================================================

def get_spark() -> SparkSession:
    global spark_session
    if spark_session is None:
        logger.info("Creating Spark session...")
        spark_session = (
            SparkSession.builder
            .appName("FloodPrediction_API")
            .config("spark.driver.host",            "127.0.0.1")
            .config("spark.driver.bindAddress",     "127.0.0.1")
            .config("spark.python.worker.reuse",    "false")
            .config("spark.driver.memory",          "2g")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        spark_session.sparkContext.setLogLevel("WARN")
        logger.info("[OK] Spark session ready.")
    return spark_session

# =============================================================================
# REDIS CLIENT
# =============================================================================

def get_redis() -> redis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(
            host    = os.environ.get("REDIS_HOST", "localhost"),
            port    = int(os.environ.get("REDIS_PORT", "6379")),
            db      = int(os.environ.get("REDIS_DB", "0")),
            decode_responses         = True,
            socket_connect_timeout   = 2,
        )
        try:
            redis_client.ping()
            logger.info("[OK] Redis connection ready.")
        except Exception as e:
            logger.warning("Redis unavailable — caching disabled: %s", e)
            redis_client = None
    return redis_client

# =============================================================================
# MODEL LOADER
# =============================================================================

def load_model():
    global flood_model
    if flood_model is not None:
        return flood_model

    if not os.path.exists(LATEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {LATEST_MODEL_PATH}\n"
            "→ Run best_model.py first."
        )

    if not os.path.exists(os.path.join(LATEST_MODEL_PATH, "metadata")) or \
       not os.path.exists(os.path.join(LATEST_MODEL_PATH, "stages")):
        raise FileNotFoundError(
            f"Model folder incomplete (missing metadata/ or stages/).\n"
            "→ Re-run best_model.py."
        )

    logger.info("Loading model from: %s", LATEST_MODEL_PATH)
    get_spark()
    flood_model = PipelineModel.load(LATEST_MODEL_PATH)
    logger.info("[OK] Model loaded: %s", type(flood_model).__name__)
    return flood_model

# =============================================================================
# FEATURE ASSEMBLER  —  built once at startup, reused per request
# =============================================================================

def init_assembler():
    global feature_assembler
    if feature_assembler is None:
        logger.info("Building VectorAssembler...")
        feature_assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
        logger.info("[OK] VectorAssembler ready for %d features.", len(FEATURE_COLS))
    return feature_assembler

# =============================================================================
# NATIVE MODEL EXTRACTOR
# Pulls weights/stage out of the PipelineModel once at startup so we can
# run inference in pure Python/NumPy — no Spark job per request.
#
# Supported:
#   LinearRegression  →  numpy dot-product   (~10 ms)
#   GBT / RF          →  single-row Spark     (~800 ms, still faster than
#                        full pipeline path)
# =============================================================================

def init_native_model():
    """Extract the predictive stage from the pipeline for fast inference."""
    global native_model
    pipeline = load_model()

    for stage in pipeline.stages:
        # ── Linear Regression ───────────────────────────────────────────────
        if hasattr(stage, "coefficients") and hasattr(stage, "intercept"):
            coeffs    = stage.coefficients.toArray()   # numpy array
            intercept = float(stage.intercept)
            native_model = ("linear", coeffs, intercept)
            logger.info(
                "[OK] Native model: LinearRegression  |  coefficients: %d", len(coeffs)
            )
            return

        # ── Tree Ensemble (GBT / RandomForest) ──────────────────────────────
        if hasattr(stage, "featureImportances"):
            native_model = ("tree", stage)
            logger.info(
                "[OK] Native model: %s  |  will use single-row Spark path",
                type(stage).__name__
            )
            return

    raise ValueError(
        "No supported model stage found in pipeline "
        "(expected LinearRegression, GBTRegressor, or RandomForestRegressor)."
    )

# =============================================================================
# FAST PREDICT HELPER
# Called by the /predict route — no Spark overhead for linear models.
# =============================================================================

def fast_predict(row_values: list[float]) -> float:
    """Run inference using the pre-extracted native model."""
    kind = native_model[0]

    if kind == "linear":
        _, coeffs, intercept = native_model
        x     = np.array(row_values, dtype=np.float64)
        logit = float(np.dot(coeffs, x) + intercept)
        return 1.0 / (1.0 + math.exp(-logit))          # sigmoid

    elif kind == "tree":
        # Single-row Spark path — still avoids the full pipeline overhead
        _, stage = native_model
        spark    = get_spark()
        row_dict = dict(zip(FEATURE_COLS, row_values))
        df       = spark.createDataFrame([row_dict])
        df       = feature_assembler.transform(df)
        result   = stage.transform(df).collect()[0]["prediction"]
        return 1.0 / (1.0 + math.exp(-result))

    raise RuntimeError(f"Unknown native model kind: {kind!r}")

# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI starting up...")
    try:
        load_model()
        init_assembler()
        init_native_model()   # ← extract weights once; fast_predict uses them
        get_redis()
        logger.info("[OK] Ready — %d features, model loaded.", len(FEATURE_COLS))
    except Exception as e:
        logger.error("Startup failed: %s", e)
        raise
    yield
    logger.info("Shutting down Spark...")
    try:
        s = SparkSession.getActiveSession()
        if s:
            s.stop()
    except Exception as e:
        logger.warning("Spark stop error: %s", e)

# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title       = "Flood Prediction API",
    description = "PySpark · MLflow · DagsHub  |  17-feature flood probability predictor",
    version     = "1.0.0",
    lifespan    = lifespan,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code = 429,
        content     = {"detail": "Rate limit exceeded: 10 requests per minute. Try again later."},
    )

# =============================================================================
# SCHEMAS  —  17 exact features from params.yaml
# =============================================================================

class FloodInput(BaseModel):
    # 3 columns were dropped during preprocessing (params.yaml → drop_columns)
    # PoliticalFactors, InadequatePlanning, PopulationScore  ← do NOT send these
    MonsoonIntensity:                float = Field(..., example=SAMPLE_INPUT["MonsoonIntensity"])
    TopographyDrainage:              float = Field(..., example=SAMPLE_INPUT["TopographyDrainage"])
    RiverManagement:                 float = Field(..., example=SAMPLE_INPUT["RiverManagement"])
    Deforestation:                   float = Field(..., example=SAMPLE_INPUT["Deforestation"])
    Urbanization:                    float = Field(..., example=SAMPLE_INPUT["Urbanization"])
    ClimateChange:                   float = Field(..., example=SAMPLE_INPUT["ClimateChange"])
    DamsQuality:                     float = Field(..., example=SAMPLE_INPUT["DamsQuality"])
    Siltation:                       float = Field(..., example=SAMPLE_INPUT["Siltation"])
    AgriculturalPractices:           float = Field(..., example=SAMPLE_INPUT["AgriculturalPractices"])
    Encroachments:                   float = Field(..., example=SAMPLE_INPUT["Encroachments"])
    IneffectiveDisasterPreparedness: float = Field(..., example=SAMPLE_INPUT["IneffectiveDisasterPreparedness"])
    DrainageSystems:                 float = Field(..., example=SAMPLE_INPUT["DrainageSystems"])
    CoastalVulnerability:            float = Field(..., example=SAMPLE_INPUT["CoastalVulnerability"])
    Landslides:                      float = Field(..., example=SAMPLE_INPUT["Landslides"])
    Watersheds:                      float = Field(..., example=SAMPLE_INPUT["Watersheds"])
    DeterioratingInfrastructure:     float = Field(..., example=SAMPLE_INPUT["DeterioratingInfrastructure"])
    WetlandLoss:                     float = Field(..., example=SAMPLE_INPUT["WetlandLoss"])

    model_config = {
        "json_schema_extra": {
            "example": SAMPLE_INPUT   # pre-fills Swagger UI with sample values
        }
    }


class FloodOutput(BaseModel):
    prediction    : float
    features_used : int
    status        : str = "success"

# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", tags=["Health"])
def root():
    return {"status": "running", "service": "Flood Prediction API"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status"        : "healthy" if flood_model is not None else "model not loaded",
        "model_loaded"  : flood_model is not None,
        "model_type"    : native_model[0] if native_model else "unknown",
        "model_path"    : LATEST_MODEL_PATH,
        "features_count": len(FEATURE_COLS),
    }


@app.get("/features", tags=["Info"])
def get_features():
    """Exact feature names and order the model expects."""
    return {
        "feature_count" : len(FEATURE_COLS),
        "feature_cols"  : FEATURE_COLS,
        "target_col"    : TARGET_COL,
        "dropped_cols"  : _params["data"]["drop_columns"],
        "sample_input"  : SAMPLE_INPUT,
    }


@app.post("/predict", response_model=FloodOutput, tags=["Prediction"])
@limiter.limit("10/minute")
def predict(request: Request, data: FloodInput):
    try:
        dcache = get_redis()

        # Feature values in the exact order FEATURE_COLS expects
        row_values = [float(getattr(data, col)) for col in FEATURE_COLS]

        # Cache key: ordered feature vector (fixed order, no sorting bug)
        cache_key = "predict:" + json.dumps(row_values)

        if dcache is not None:
            cached = dcache.get(cache_key)
            if cached:
                logger.info("Cache hit — returning instantly.")
                return FloodOutput(prediction=float(cached), features_used=len(FEATURE_COLS))

        # ── Fast inference — no Spark job for linear models ──────────────────
        result = fast_predict(row_values)

        if dcache is not None:
            dcache.set(cache_key, str(result), ex=60)

        logger.info(
            "Prediction: %.6f  |  model: %s  |  features: %d",
            result, native_model[0], len(FEATURE_COLS)
        )
        return FloodOutput(prediction=float(result), features_used=len(FEATURE_COLS))

    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# RUN:  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# =============================================================================