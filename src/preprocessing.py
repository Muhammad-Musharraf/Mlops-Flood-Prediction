import os
import yaml
import mlflow
import mlflow.spark
import dagshub
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from logger import logging


# ── Load params ───────────────────────────────────────────────────────────────
try:
    params = yaml.safe_load(open("params.yaml"))
    logging.info("params.yaml loaded successfully.")
except Exception as e:
    logging.critical(f"Failed to load params.yaml: {e}")
    raise

DATA_PATH    = params["data"]["path"]
TARGET_COL   = params["data"]["target_column"]
DROP_COLS    = params["data"]["drop_columns"]
FEATURE_COLS = params["data"]["feature_columns"]
TEST_SIZE    = params["split"]["test_size"]
RANDOM_STATE = params["split"]["random_state"]
TRAIN_OUT    = params["output"]["train_path"]    # e.g. "data/transform/train"
TEST_OUT     = params["output"]["test_path"]     # e.g. "data/transform/test"
PIPELINE_OUT = params["output"]["pipeline_path"] # e.g. "models/pipeline"

EXPERIMENT_NAME = "Flood Prediction Preprocessing"

# ── DagsHub / MLflow tracking setup ─────────────────────────────────────────
try:
    dagshub.init(repo_owner='Muhammad-Musharraf', repo_name='Mlops-Flood-Prediction', mlflow=True)
    logging.info("DagsHub initialized successfully.")
except Exception as e:
    logging.warning(f"DagsHub init failed, continuing without remote tracking: {e}")

REMOTE_URI = os.getenv("MLFLOW_TRACKING_URI")
if REMOTE_URI:
    mlflow.set_tracking_uri(REMOTE_URI)
    logging.info(f"MLflow tracking -> Remote : {REMOTE_URI}")
else:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    logging.info("MLflow tracking -> Local  : sqlite:///mlflow.db")

mlflow.set_experiment(EXPERIMENT_NAME)
logging.info(f"MLflow experiment set : {EXPERIMENT_NAME}")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocessing():
    logging.info("Preprocessing pipeline started.")

    # ── Initialise Spark ──────────────────────────────────────────────────────
    try:
        spark = (
            SparkSession.builder
            .appName("FloodPrediction_Preprocessing")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        logging.info("SparkSession initialised successfully.")
    except Exception as e:
        logging.critical(f"Failed to initialise SparkSession: {e}")
        raise

    with mlflow.start_run(run_name="preprocessing"):

        # ── Log static params up front ──────────────────────────────────────
        mlflow.log_param("data_path", DATA_PATH)
        mlflow.log_param("target_col", TARGET_COL)
        mlflow.log_param("drop_cols", DROP_COLS)
        mlflow.log_param("num_features", len(FEATURE_COLS))
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("scaler", "StandardScaler(withMean=True, withStd=True)")
        mlflow.set_tag("stage", "preprocessing")
        mlflow.set_tag("framework", "pyspark")

        try:
            # ── 1. Load raw data ──────────────────────────────────────────────────
            try:
                df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
                raw_rows = df.count()
                raw_cols = len(df.columns)
                logging.info(f"Data loaded — Rows: {raw_rows} | Cols: {raw_cols}")
                mlflow.log_metric("raw_rows", raw_rows)
                mlflow.log_metric("raw_cols", raw_cols)
            except Exception as e:
                logging.error(f"Failed to read CSV from '{DATA_PATH}': {e}")
                raise

            # ── 2. Drop unwanted columns ──────────────────────────────────────────
            try:
                df = df.drop(*DROP_COLS)
                logging.info(f"Columns after drop: {df.columns}")
            except Exception as e:
                logging.error(f"Failed to drop columns {DROP_COLS}: {e}")
                raise

            # ── 3. Keep only needed columns (features + target) ───────────────────
            try:
                keep_cols = FEATURE_COLS + [TARGET_COL]
                df = df.select([col(c) for c in keep_cols])
                logging.info(f"Selected {len(keep_cols)} columns: {keep_cols}")
            except Exception as e:
                logging.error(f"Failed to select required columns: {e}")
                raise

            # ── 4. Train / test split ─────────────────────────────────────────────
            try:
                train_ratio = 1.0 - TEST_SIZE
                train_df, test_df = df.randomSplit([train_ratio, TEST_SIZE], seed=RANDOM_STATE)
                train_rows = train_df.count()
                test_rows = test_df.count()
                logging.info(f"Train rows: {train_rows} | Test rows: {test_rows}")
                mlflow.log_metric("train_rows", train_rows)
                mlflow.log_metric("test_rows", test_rows)
            except Exception as e:
                logging.error(f"Failed to split data into train/test: {e}")
                raise

            # ── 5. Build PySpark ML Pipeline ──────────────────────────────────────
            try:
                assembler = VectorAssembler(
                    inputCols=FEATURE_COLS,
                    outputCol="features_raw"
                )
                scaler = StandardScaler(
                    inputCol="features_raw",
                    outputCol="features_scaled",
                    withMean=True,
                    withStd=True
                )
                pipeline = Pipeline(stages=[assembler, scaler])
                pipeline_model = pipeline.fit(train_df)
                logging.info("ML pipeline fitted on training data.")
            except Exception as e:
                logging.error(f"Failed to build or fit ML pipeline: {e}")
                raise

            # ── 6. Transform both splits ──────────────────────────────────────────
            try:
                train_transformed = pipeline_model.transform(train_df)
                test_transformed  = pipeline_model.transform(test_df)
                logging.info("Train and test datasets transformed successfully.")
            except Exception as e:
                logging.error(f"Failed to transform datasets: {e}")
                raise

            # ── 7. Expand scaled vector back into named columns ───────────────────
            try:
                def expand_features(transformed_df):
                    arr_df = transformed_df.withColumn(
                        "features_arr", vector_to_array("features_scaled")
                    )
                    feature_exprs = [
                        col("features_arr")[i].alias(FEATURE_COLS[i])
                        for i in range(len(FEATURE_COLS))
                    ]
                    return arr_df.select(*feature_exprs, TARGET_COL)

                train_out_df = expand_features(train_transformed)
                test_out_df  = expand_features(test_transformed)
                logging.info("Scaled features expanded into individual named columns.")
            except Exception as e:
                logging.error(f"Failed to expand feature vector columns: {e}")
                raise

            # ── 8. Save as Parquet ────────────────────────────────────────────────
            try:
                os.makedirs("data/transform", exist_ok=True)
                train_out_df.write.mode("overwrite").parquet(TRAIN_OUT)
                logging.info(f"Train Parquet saved -> {TRAIN_OUT}")

                test_out_df.write.mode("overwrite").parquet(TEST_OUT)
                logging.info(f"Test Parquet saved  -> {TEST_OUT}")
            except Exception as e:
                logging.error(f"Failed to save Parquet files: {e}")
                raise

            # ── 9. Save fitted pipeline (local disk) ───────────────────────────────
            try:
                os.makedirs(PIPELINE_OUT, exist_ok=True)
                pipeline_model.write().overwrite().save(PIPELINE_OUT)
                logging.info(f"Pipeline model saved -> {PIPELINE_OUT}")
            except Exception as e:
                logging.error(f"Failed to save pipeline model: {e}")
                raise

            # ── 10. Log the fitted pipeline to MLflow as an artifact ───────────────
            try:
                mlflow.spark.log_model(pipeline_model, artifact_path="preprocessing_pipeline")
                logging.info("Preprocessing pipeline logged to MLflow under 'preprocessing_pipeline'.")
            except Exception as e:
                # Non-fatal: the pipeline is already safe on local disk (step 9).
                logging.warning(f"Failed to log pipeline model to MLflow (non-fatal): {e}")

            mlflow.set_tag("status", "success")

        except Exception as e:
            mlflow.set_tag("status", "failed")
            logging.critical(f"Preprocessing pipeline failed: {e}")
            raise
        finally:
            spark.stop()
            logging.info("SparkSession stopped.")

    logging.info("Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    try:
        preprocessing()
    except Exception as e:
        logging.critical(f"Unhandled exception in __main__: {e}")
        raise