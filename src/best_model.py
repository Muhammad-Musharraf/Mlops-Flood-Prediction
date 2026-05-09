import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os

load_dotenv()  

# Set MLflow Tracking URI from environment variable (username, uri, password) of daghub for remote storage
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))


# MLflow Tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db") 

def register_best_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Flood_Prediction_Models")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.Test_R2 DESC"]
    )

    best_run        = runs[0]
    best_run_id     = best_run.info.run_id
    best_model_name = best_run.data.params["model_name"]
    best_r2         = best_run.data.metrics["Test_R2"]

    print(f"Best Model  : {best_model_name}")
    print(f"Best Test R2: {best_r2:.4f}")
    print(f"Best Run ID : {best_run_id}")

    mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name="Flood_Prediction_Model"
    )

    print("Best Model Registered!")


if __name__ == "__main__":
    register_best_model()