import dagshub
from dotenv import load_dotenv
load_dotenv()

import sys
import os
import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

dagshub.init(
    repo_owner="Muhammad-Musharraf",
    repo_name="Mlops-Flood-Prediction",
    mlflow=True
)

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from logger import logging
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession


def best_model():
    try:
        logging.info("Starting best model selection process...")

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        run_id_path  = "logs/best_run_id.txt"

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # ── Step 1: best_run_id file se lo ───────────────────────────────────
        if not os.path.exists(run_id_path):
            raise Exception("logs/best_run_id.txt nahi mili — pehle train.py chalao")

        with open(run_id_path, "r") as f:
            best_run_id = f.read().strip()

        if not best_run_id:
            raise Exception("logs/best_run_id.txt khali hai — train.py phir se chalao")

        logging.info("===================================")
        logging.info("BEST RUN ID : %s", best_run_id)
        logging.info("===================================")

        # ── Step 2: Champion run details ──────────────────────────────────────
        champion_run    = client.get_run(best_run_id)
        best_model_name = champion_run.data.params.get("best_model_name", "UnknownModel")
        best_rmse       = champion_run.data.metrics.get("best_RMSE", 0.0)

        logging.info("===================================")
        logging.info("Best Model Name : %s", best_model_name)
        logging.info("Best RMSE       : %.6f", best_rmse)
        logging.info("===================================")

        # ── Step 3: Spark session ─────────────────────────────────────────────
        spark = (
            SparkSession.builder
            .appName("FloodPrediction_BestModel")
            .config("spark.driver.host",        "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.python.worker.reuse","false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        logging.info("Spark session started.")

        # ── Step 4: Champion model MLflow se load karo ────────────────────────
        logging.info("Loading champion model from MLflow...")
        champion_model_uri = f"runs:/{best_run_id}/best_model"
        champion_model     = mlflow.spark.load_model(champion_model_uri)
        logging.info("[OK] Champion model loaded: %s", type(champion_model).__name__)

        # ── Step 5: Disk pe versioned + latest save karo ──────────────────────
        os.makedirs("models", exist_ok=True)

        existing       = [d for d in os.listdir("models") if d.startswith("best_model_v")]
        version        = len(existing) + 1
        versioned_path = f"models/best_model_v{version}"
        latest_path    = "models/best_model_latest"

        champion_model.write().overwrite().save(versioned_path)
        logging.info("[OK] Saved versioned : %s", versioned_path)

        champion_model.write().overwrite().save(latest_path)
        logging.info("[OK] Saved latest    : %s", latest_path)

        logging.info("===================================")
        logging.info("Best model selection completed!")
        logging.info("===================================")

        # ── Step 6: Log file save karo ────────────────────────────────────────
        os.makedirs("logs", exist_ok=True)
        with open("logs/best_model.log", "w") as f:
            f.write(f"Best model selection completed at {datetime.datetime.now()}\n")
            f.write(f"Model      : Flood_best_model\n")
            f.write(f"Algorithm  : {best_model_name}\n")
            f.write(f"Run ID     : {best_run_id}\n")
            f.write(f"Best RMSE  : {best_rmse:.6f}\n")
            f.write(f"Version    : v{version}\n")
            f.write(f"Versioned  : {versioned_path}\n")
            f.write(f"Latest     : {latest_path}\n")
        logging.info("Log file saved: logs/best_model.log")

    except Exception as e:
        logging.error("Error: %s", str(e))
        raise


if __name__ == "__main__":
    try:
        best_model()
    except Exception as e:
        logging.error("Error: %s", str(e))