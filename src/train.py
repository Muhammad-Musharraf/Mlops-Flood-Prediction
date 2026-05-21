import os
import yaml
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from logger import logging

import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

# Set credentials as env vars — MLflow picks them up automatically
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Your_Experiment_Name")


# ── Load params ───────────────────────────────────────────────────────────────
try:
    params = yaml.safe_load(open("params.yaml"))
    logging.info("params.yaml loaded successfully.")
except Exception as e:
    logging.critical(f"Failed to load params.yaml: {e}")
    raise

TARGET_COL   = params["data"]["target_column"]
FEATURE_COLS = params["data"]["feature_columns"]
TEST_SIZE    = params["split"]["test_size"]
RANDOM_STATE = params["split"]["random_state"]
TRAIN_PATH   = params["output"]["train_path"]
TEST_PATH    = params["output"]["test_path"]
MODEL_PARAMS = params["models"]

EXPERIMENT_NAME  = "Flood_Prediction_Regression_Experiment"
REGISTRY_PREFIX  = "FloodPrediction"   # registered model names: FloodPrediction_<ModelName>
BEST_REGISTRY_NAME = "FloodPrediction_BestModel"


# ── MLflow tracking URI ───────────────────────────────────────────────────────
try:
    tracking_uri = os.getenv("Mlfow_Tracking_Uri")
    if not tracking_uri:
        raise ValueError("Environment variable 'Mlflow_Tracking_Uri' is not set.")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logging.info(f"MLflow tracking URI  : {tracking_uri}")
    logging.info(f"MLflow experiment    : {EXPERIMENT_NAME}")
except Exception as e:
    logging.critical(f"Failed to configure MLflow: {e}")
    raise


# ── PySpark-compatible model registry ─────────────────────────────────────────
def build_models(model_params: dict) -> dict:
    registry = {}

    for name, cfg in model_params.items():
        if not cfg.get("enabled", True):
            logging.info(f"Skipping disabled model: {name}")
            continue

        p = cfg.get("params", {})

        if name == "LinearRegression":
            registry[name] = {
                "model": LinearRegression(featuresCol="features", labelCol=TARGET_COL),
                "params": {
                    "regParam":        [0.01, 0.1],
                    "elasticNetParam": [0.0, 0.5, 1.0],
                },
            }

        elif name == "Ridge":
            alpha = p.get("alpha", 1.0)
            registry[name] = {
                "model": LinearRegression(featuresCol="features", labelCol=TARGET_COL, elasticNetParam=0.0),
                "params": {"regParam": [alpha / 2, alpha, alpha * 2]},
            }

        elif name == "Lasso":
            alpha = p.get("alpha", 1.0)
            registry[name] = {
                "model": LinearRegression(featuresCol="features", labelCol=TARGET_COL, elasticNetParam=1.0),
                "params": {"regParam": [alpha / 2, alpha, alpha * 2]},
            }

        elif name == "DecisionTreeRegressor":
            max_depth         = p.get("max_depth") or 5
            min_samples_split = p.get("min_samples_split", 2)
            registry[name] = {
                "model": DecisionTreeRegressor(featuresCol="features", labelCol=TARGET_COL),
                "params": {
                    "maxDepth":            [3, max_depth, max_depth + 2],
                    "minInstancesPerNode": [1, min_samples_split],
                },
            }

        elif name == "RandomForestRegressor":
            n_estimators      = p.get("n_estimators", 100)
            max_depth         = p.get("max_depth") or 5
            min_samples_split = p.get("min_samples_split", 2)
            registry[name] = {
                "model": RandomForestRegressor(featuresCol="features", labelCol=TARGET_COL),
                "params": {
                    "numTrees":            [50, n_estimators],
                    "maxDepth":            [max_depth, max_depth + 5],
                    "minInstancesPerNode": [1, min_samples_split],
                },
            }

        elif name == "GradientBoostingRegressor":
            max_depth    = p.get("max_depth", 3)
            n_estimators = p.get("n_estimators", 100)
            registry[name] = {
                "model": GBTRegressor(featuresCol="features", labelCol=TARGET_COL),
                "params": {
                    "maxDepth": [max_depth, max_depth + 2],
                    "maxIter":  [20, n_estimators],
                },
            }

        elif name == "XGBRegressor":
            max_depth    = p.get("max_depth", 5)
            n_estimators = p.get("n_estimators", 500)
            registry[name] = {
                "model": GBTRegressor(featuresCol="features", labelCol=TARGET_COL),
                "params": {
                    "maxDepth": [max_depth, max_depth + 2],
                    "maxIter":  [50, min(n_estimators, 100)],
                },
            }

        else:
            logging.warning(f"Model '{name}' has no PySpark equivalent. Skipping.")

    return registry


# ── Helper: transition a registered model version to a stage ──────────────────
def transition_model_stage(client: MlflowClient, model_name: str, version: str, stage: str):
    """Transition a registered model version to Staging / Production / Archived."""
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,   # archive any previous Production version
        )
        logging.info(f"Model '{model_name}' v{version} transitioned to '{stage}'.")
    except Exception as e:
        logging.error(f"Failed to transition '{model_name}' v{version} to '{stage}': {e}")
        raise


# ── Main training function ─────────────────────────────────────────────────────
def train():
    logging.info("Training pipeline started.")

    # ── 1. Spark Session ──────────────────────────────────────────────────────
    try:
        spark = (
            SparkSession.builder
            .appName("FloodPrediction_Training")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        logging.info("SparkSession initialised.")
    except Exception as e:
        logging.critical(f"Failed to initialise SparkSession: {e}")
        raise

    try:
        # ── 2. Load Parquet data ──────────────────────────────────────────────
        try:
            train_df = spark.read.parquet(TRAIN_PATH)
            test_df  = spark.read.parquet(TEST_PATH)
            logging.info(f"Train rows : {train_df.count()} | Test rows : {test_df.count()}")
        except Exception as e:
            logging.error(f"Failed to load Parquet data: {e}")
            raise

        # ── 3. Assemble feature vector ────────────────────────────────────────
        try:
            assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
            train_df  = assembler.transform(train_df).cache()
            test_df   = assembler.transform(test_df).cache()
            logging.info("Feature vector assembled and datasets cached.")
        except Exception as e:
            logging.error(f"Failed to assemble feature vector: {e}")
            raise

        # ── 4. Evaluators ─────────────────────────────────────────────────────
        try:
            evaluator     = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="r2")
            mse_evaluator = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="mse")
            mae_evaluator = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="mae")
            logging.info("Regression evaluators initialised (r2, mse, mae).")
        except Exception as e:
            logging.error(f"Failed to initialise evaluators: {e}")
            raise

        # ── 5. Build model registry ───────────────────────────────────────────
        try:
            models = build_models(MODEL_PARAMS)
            logging.info(f"Models to train: {list(models.keys())}")
        except Exception as e:
            logging.error(f"Failed to build model registry: {e}")
            raise

        # ── 6. MLflow client (for registry operations) ────────────────────────
        client = MlflowClient()

        best_model_name    = None
        best_r2            = float("-inf")
        best_model         = None
        best_run_id        = None
        best_artifact_path = None

        # ── 7. Training loop ──────────────────────────────────────────────────
        for name, config in models.items():
            logging.info(f"Training model: {name}")
            registered_name = f"{REGISTRY_PREFIX}_{name}"

            try:
                with mlflow.start_run(run_name=name) as run:
                    run_id = run.info.run_id

                    # Build param grid
                    try:
                        model      = config["model"]
                        param_grid = ParamGridBuilder()
                        for k, v in config["params"].items():
                            param_grid = param_grid.addGrid(getattr(model, k), v)
                        param_grid = param_grid.build()
                        logging.info(f"[{name}] Param grid: {len(param_grid)} combinations.")
                    except Exception as e:
                        logging.error(f"[{name}] Param grid build failed: {e}")
                        raise

                    # Cross-validation
                    try:
                        cv = CrossValidator(
                            estimator=model,
                            estimatorParamMaps=param_grid,
                            evaluator=evaluator,
                            numFolds=3,
                        )
                        cv_model    = cv.fit(train_df)
                        predictions = cv_model.transform(test_df)
                        logging.info(f"[{name}] Cross-validation complete.")
                    except Exception as e:
                        logging.error(f"[{name}] Cross-validation failed: {e}")
                        raise

                    # Evaluate
                    try:
                        r2  = evaluator.evaluate(predictions)
                        mse = mse_evaluator.evaluate(predictions)
                        mae = mae_evaluator.evaluate(predictions)
                        logging.info(f"[{name}] R2={r2:.4f} | MSE={mse:.4f} | MAE={mae:.4f}")
                    except Exception as e:
                        logging.error(f"[{name}] Metric evaluation failed: {e}")
                        raise

                    # Log params + metrics + model → MLflow Registry
                    try:
                        mlflow.log_params({
                            "model":        name,
                            "test_size":    TEST_SIZE,
                            "random_state": RANDOM_STATE,
                            "num_features": len(FEATURE_COLS),
                            "target_col":   TARGET_COL,
                            "cv_folds":     3,
                        })
                        mlflow.log_metrics({"r2": r2, "mse": mse, "mae": mae})

                        # ✅ log_model with registered_model_name saves to the Registry
                        model_info = mlflow.spark.log_model(
                            cv_model.bestModel,
                            artifact_path="model",
                            registered_model_name=registered_name,   # ← Registry entry
                        )
                        logging.info(f"[{name}] Registered in MLflow Registry as '{registered_name}'.")

                        # Transition this version to Staging
                        latest_version = client.get_latest_versions(registered_name, stages=["None"])[0].version
                        transition_model_stage(client, registered_name, latest_version, "Staging")

                    except Exception as e:
                        logging.error(f"[{name}] MLflow logging/registration failed: {e}")
                        raise

                    # Track best
                    if r2 > best_r2:
                        best_r2            = r2
                        best_model         = cv_model.bestModel
                        best_model_name    = name
                        best_run_id        = run_id
                        best_artifact_path = model_info.artifact_path
                        logging.info(f"[{name}] New best model — R2: {best_r2:.4f}")

            except Exception as e:
                logging.error(f"[{name}] Run failed, skipping: {e}")
                continue

        # ── 8. Final summary ──────────────────────────────────────────────────
        if best_model is None:
            logging.critical("No model trained successfully.")
            raise RuntimeError("All model training runs failed.")

        logging.info("===========================================")
        logging.info(f"BEST MODEL : {best_model_name}")
        logging.info(f"BEST R2    : {best_r2:.4f}")
        logging.info("===========================================")

        # ── 9. Register best model separately + promote to Production ─────────
        try:
            with mlflow.start_run(run_name="best_model") as best_run:
                mlflow.log_params({
                    "best_model_name": best_model_name,
                    "best_r2":         best_r2,
                })

                # ✅ Register the best model under its own dedicated registry name
                best_model_info = mlflow.spark.log_model(
                    best_model,
                    artifact_path="best_model",
                    registered_model_name=BEST_REGISTRY_NAME,   # ← Registry entry
                )
                logging.info(f"Best model registered as '{BEST_REGISTRY_NAME}'.")

                # Promote best model to Production
                best_version = client.get_latest_versions(BEST_REGISTRY_NAME, stages=["None"])[0].version
                transition_model_stage(client, BEST_REGISTRY_NAME, best_version, "Production")

        except Exception as e:
            logging.error(f"Failed to register/promote best model: {e}")
            raise

        logging.info("Training pipeline completed successfully.")
        logging.info(f"View registry : {tracking_uri}/#/models")

    except Exception as e:
        logging.critical(f"Training pipeline failed: {e}")
        raise
    finally:
        spark.stop()
        logging.info("SparkSession stopped.")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logging.critical(f"Unhandled exception in __main__: {e}")
        raise