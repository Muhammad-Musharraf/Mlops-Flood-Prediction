import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import logging
import os

import dagshub
dagshub.init(repo_owner='Muhammad-Musharraf', repo_name='Mlops-Flood-Prediction', mlflow=True)

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(TRACKING_URI)


def register_best_model():
    try:
        # ── Step 1: Fetch Best Run from DagsHub ──────────────────────────────
        logger.info("Connecting to MLflow tracking server...")
        client = MlflowClient(tracking_uri=TRACKING_URI)

        experiment = client.get_experiment_by_name("Flood Prediction Training")
        if experiment is None:
            raise ValueError("Experiment 'Flood Prediction Training' not found.")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.Test_R2 DESC"]
        )

        if not runs:
            raise ValueError("No runs found in the experiment.")

        best_run        = runs[0]
        best_run_id     = best_run.info.run_id
        best_model_name = best_run.data.params.get("model_name", "Unknown")
        best_r2         = best_run.data.metrics.get("Test_R2", 0.0)

        logger.info(f"Best Model  : {best_model_name}")
        logger.info(f"Best Test R2: {best_r2:.4f}")
        logger.info(f"Best Run ID : {best_run_id}")

        # ── Step 2: Register Model ────────────────────────────────────────────
        logger.info("Registering best model as 'best_model'...")

        client.create_registered_model("best_model")
        client.create_model_version(
            name="best_model",
            source=f"runs:/{best_run_id}/model",
            run_id=best_run_id
        )

        logger.info("✅ Best Model Registered Successfully as 'best_model'!")

    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")

    except mlflow.exceptions.RestException as re:
        logger.error(f"MLflow REST Error (DagsHub may not support Model Registry): {re}")

    except mlflow.exceptions.MlflowException as me:
        logger.error(f"MLflow Error: {me}")

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")


if __name__ == "__main__":
    register_best_model()