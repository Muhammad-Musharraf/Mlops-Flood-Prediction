<<<<<<< HEAD
import os
import yaml
import mlflow.spark
from mlflow.tracking import MlflowClient
=======
from dotenv import load_dotenv
load_dotenv()

import os
import json
import yaml
from datetime import datetime
>>>>>>> 3fa2cca (Add all files)
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
<<<<<<< HEAD
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
=======
import mlflow
print(mlflow.__version__)



def train():
    try:
        print("Starting model training process...")

        # ── Load params ───────────────────────────────────────────────────────
        params = yaml.safe_load(open("params.yaml"))

        TARGET_COL   = params["data"]["target_column"]
        FEATURE_COLS = params["data"]["feature_columns"]
        TEST_SIZE    = params["split"]["test_size"]
        RANDOM_STATE = params["split"]["random_state"]
        TRAIN_CSV    = params["output"]["train_path"]       # data/processed/train
        TEST_CSV     = params["output"]["test_path"]        # data/processed/test
        CV_FOLDS     = params["model"]["cv"]
        SCORES_PATH  = params["artifacts"]["scores_file"]  # artifacts/scores.json

        EXPERIMENT_NAME = "FloodPrediction_Training"

        # ── 1. Spark Session ──────────────────────────────────────────────────
        spark = (
            SparkSession.builder
            .appName("FloodPrediction_Training")
            .config(
                "spark.hadoop.fs.mlflow-artifacts.impl",
                "org.mlflow.tracking.creds.MlflowContextFileSystem",
            )
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        # ── 2. Load preprocessed CSVs ─────────────────────────────────────────
        # preprocessing.py saves with coalesce(1) so Spark wrote a folder;
        # reading the folder directly picks up the single part file inside.
        train_df = spark.read.csv(TRAIN_CSV, header=True, inferSchema=True)
        test_df  = spark.read.csv(TEST_CSV,  header=True, inferSchema=True)

        print("===================================")
        print("Loaded Preprocessed Dataset")
        print(f"  Train rows : {train_df.count()}")
        print(f"  Test  rows : {test_df.count()}")
        print(f"  Features   : {FEATURE_COLS}")
        print(f"  Target     : {TARGET_COL}")
        print("===================================")

        # Assemble feature vector (features already scaled by preprocessing.py)
        assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
        train_df  = assembler.transform(train_df).cache()
        test_df   = assembler.transform(test_df).cache()

        # ── 3. MLflow setup ───────────────────────────────────────────────────
        REMOTE_URI = os.getenv("MLFLOW_TRACKING_URI")
        if REMOTE_URI:
            mlflow.set_tracking_uri(REMOTE_URI)
            print(f"MLflow tracking → Remote : {REMOTE_URI}")
        else:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            print("MLflow tracking → Local  : sqlite:///mlflow.db")

        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"MLflow experiment set    : {EXPERIMENT_NAME}")

        # ── 4. Evaluators ─────────────────────────────────────────────────────
        def make_evaluator(metric):
            return RegressionEvaluator(
                labelCol=TARGET_COL,
                predictionCol="prediction",
                metricName=metric,
            )

        evaluator_r2   = make_evaluator("r2")
        evaluator_rmse = make_evaluator("rmse")
        evaluator_mse  = make_evaluator("mse")
        evaluator_mae  = make_evaluator("mae")

        # ── 5. Models ─────────────────────────────────────────────────────────
        models = {
            "LinearRegression": {
>>>>>>> 3fa2cca (Add all files)
                "model": LinearRegression(featuresCol="features", labelCol=TARGET_COL),
                "params": {
                    "regParam":        [0.01, 0.1],
                    "elasticNetParam": [0.0, 0.5, 1.0],
                },
<<<<<<< HEAD
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
=======
            },
            "DecisionTreeRegressor": {
                "model": DecisionTreeRegressor(featuresCol="features", labelCol=TARGET_COL),
                "params": {
                    "maxDepth":            [3, 5, 7],
                    "minInstancesPerNode": [1, 5, 10],
                },
            },
            "RandomForestRegressor": {
                "model": RandomForestRegressor(featuresCol="features", labelCol=TARGET_COL),
                "params": {
                    "numTrees":            [50, 100],
                    "maxDepth":            [5, 10],
                    "minInstancesPerNode": [1, 5],
                },
            },
            "GradientBoostingRegressor": {
                "model": GBTRegressor(featuresCol="features", labelCol=TARGET_COL),
                "params": {
                    "maxDepth": [3, 5],
                    "maxIter":  [20, 50],
                },
            },
        }

        # ── 6. Best-model tracker ─────────────────────────────────────────────
        best_model_name = None
        best_rmse       = float("inf")   # lower RMSE = better model
        best_model      = None
        results         = {}             # all scores → scores.json

        # ── 7. Training loop ──────────────────────────────────────────────────
        for name, config in models.items():

            print("===================================")
            print(f"Training : {name}")
            print("===================================")

            with mlflow.start_run(run_name=name):

                estimator = config["model"]

                # Build param grid
                param_grid = ParamGridBuilder()
                for param_name, values in config["params"].items():
                    param_grid = param_grid.addGrid(
                        getattr(estimator, param_name), values
                    )
                param_grid = param_grid.build()
                print(f"  ParamGrid size : {len(param_grid)} combinations")

                # Cross-validation (optimise on RMSE)
                cv = CrossValidator(
                    estimator=estimator,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator_rmse,
                    numFolds=CV_FOLDS,
                    seed=RANDOM_STATE,
                )

                cv_model = cv.fit(train_df)
                best     = cv_model.bestModel

                # Predictions on both splits
                test_preds  = cv_model.transform(test_df)
                train_preds = cv_model.transform(train_df)

                # ── All metrics ───────────────────────────────────────────────
                test_r2    = evaluator_r2.evaluate(test_preds)
                train_r2   = evaluator_r2.evaluate(train_preds)
                test_rmse  = evaluator_rmse.evaluate(test_preds)
                train_rmse = evaluator_rmse.evaluate(train_preds)
                test_mse   = evaluator_mse.evaluate(test_preds)
                train_mse  = evaluator_mse.evaluate(train_preds)
                test_mae   = evaluator_mae.evaluate(test_preds)
                train_mae  = evaluator_mae.evaluate(train_preds)

                print(f"  Train R2   : {train_r2:.4f}  |  Test R2   : {test_r2:.4f}")
                print(f"  Train RMSE : {train_rmse:.4f}  |  Test RMSE : {test_rmse:.4f}")
                print(f"  Train MAE  : {train_mae:.4f}  |  Test MAE  : {test_mae:.4f}")

                # ── Collect per-model results for scores.json ─────────────────
                results[name] = {
                    "Train_R2"   : round(train_r2,   4),
                    "Test_R2"    : round(test_r2,    4),
                    "Train_RMSE" : round(train_rmse, 4),
                    "Test_RMSE"  : round(test_rmse,  4),
                    "Train_MSE"  : round(train_mse,  4),
                    "Test_MSE"   : round(test_mse,   4),
                    "Train_MAE"  : round(train_mae,  4),
                    "Test_MAE"   : round(test_mae,   4),
                }

                # ── Log params ────────────────────────────────────────────────
                mlflow.log_param("model_name",   name)
                mlflow.log_param("test_size",    TEST_SIZE)
                mlflow.log_param("random_state", RANDOM_STATE)
                mlflow.log_param("num_features", len(FEATURE_COLS))
                mlflow.log_param("target_col",   TARGET_COL)
                mlflow.log_param("cv_folds",     CV_FOLDS)

                # Best hyperparams chosen by CV
                best_idx    = cv_model.avgMetrics.index(min(cv_model.avgMetrics))
                best_params = cv_model.getEstimatorParamMaps()[best_idx]
                for param, value in best_params.items():
                    mlflow.log_param(param.name, value)

                # ── Log all metrics (columns visible in MLflow UI) ────────────
                mlflow.log_metric("Train_R2",   train_r2)
                mlflow.log_metric("Test_R2",    test_r2)
                mlflow.log_metric("Train_RMSE", train_rmse)
                mlflow.log_metric("Test_RMSE",  test_rmse)
                mlflow.log_metric("Train_MSE",  train_mse)
                mlflow.log_metric("Test_MSE",   test_mse)
                mlflow.log_metric("Train_MAE",  train_mae)
                mlflow.log_metric("Test_MAE",   test_mae)

                # ── Overfit / underfit detection ──────────────────────────────
                diff = train_r2 - test_r2
                if diff > 0.1:
                    fit_status = "overfit"
                    mlflow.log_metric("overfit_gap", diff)
                elif test_r2 < 0.7:
                    fit_status = "underfit"
                else:
                    fit_status = "good"

                mlflow.set_tag("fit_status", fit_status)
                mlflow.set_tag("model_type", name)
                mlflow.set_tag("framework",  "pyspark")
                print(f"  Fit Status : {fit_status.upper()}"
                      + (f"  (gap={diff:.4f})" if fit_status == "overfit" else ""))

                # ── Log model artifact ────────────────────────────────────────
                mlflow.spark.log_model(best, artifact_path="model")

                # ── Track best model (lowest test RMSE) ───────────────────────
                if test_rmse < best_rmse:
                    best_rmse       = test_rmse
                    best_model      = best
                    best_model_name = name
                    print(f"  *** New best model: {name}  (RMSE={test_rmse:.4f}) ***")

        # ── 8. Summary ────────────────────────────────────────────────────────
        print("===================================")
        print(f"  BEST MODEL : {best_model_name}")
        print(f"  BEST RMSE  : {best_rmse:.4f}")
        print("===================================")

        # ── 9. Champion run ───────────────────────────────────────────────────
        with mlflow.start_run(run_name="Best_Model_" + best_model_name):
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_metric("best_RMSE",      best_rmse)
            mlflow.set_tag("stage",      "champion")
            mlflow.set_tag("model_type", best_model_name)
            mlflow.set_tag("framework",  "pyspark")
            mlflow.spark.log_model(best_model, artifact_path="best_model")

        print("Best model logged to MLflow under 'best_model'.")

        # ── 10. scores.json — append history of every run ─────────────────────
        run_record = {
            "run_timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_model"    : best_model_name,
            "best_rmse"     : round(best_rmse, 4),
            "models"        : results,
        }

        os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)

        if os.path.exists(SCORES_PATH):
            with open(SCORES_PATH, "r") as f:
                all_scores = json.load(f)
            print("Existing scores.json found — appending new run.")
        else:
            all_scores = []
            print("No scores.json found — creating new file.")

        all_scores.append(run_record)

        with open(SCORES_PATH, "w") as f:
            json.dump(all_scores, f, indent=4)

        print("===================================")
        print(f"Scores saved to    : {os.path.abspath(SCORES_PATH)}")
        print(f"Total runs in file : {len(all_scores)}")
        print("===================================")
        print("Training Completed Successfully")
        print("Run 'mlflow ui' → http://127.0.0.1:5000")
        print("===================================")

    except Exception as e:
        print(f"Error during model training: {e}")
        raise
>>>>>>> 3fa2cca (Add all files)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
<<<<<<< HEAD
        logging.critical(f"Unhandled exception in __main__: {e}")
        raise
=======
        print(f"Fatal error: {e}")
>>>>>>> 3fa2cca (Add all files)
