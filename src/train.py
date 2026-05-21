from dotenv import load_dotenv
load_dotenv()

import os
import json
import yaml
from datetime import datetime
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
        train_df = spark.read.format("parquet").load(TRAIN_CSV)
        test_df = spark.read.format("parquet").load(TEST_CSV)




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
                "model": LinearRegression(featuresCol="features", labelCol=TARGET_COL),
                "params": {
                    "regParam":        [0.01, 0.1],
                    "elasticNetParam": [0.0, 0.5, 1.0],
                },
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


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Fatal error: {e}")
