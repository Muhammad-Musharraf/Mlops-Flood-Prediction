import numpy as np
import pandas as pd
import scipy.sparse as sp
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import warnings
import yaml
from dotenv import load_dotenv
import os

load_dotenv()  

# Set MLflow Tracking URI from environment variable (username, uri, password) of daghub for remote storage
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))


# MLflow Tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db") 

# Params load 
params = yaml.safe_load(open("params.yaml")) 


def train():
    def _load_npz_flexible(path):
        try:
            return sp.load_npz(path)
        except Exception:
            # Fall back to numpy loader for non-sparse .npz
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.lib.npyio.NpzFile):
                keys = list(arr.keys())
                if len(keys) == 1:
                    return arr[keys[0]]
                # prefer common names if multiple arrays saved
                for k in ("arr_0", "data", "x", "X"):
                    if k in arr:
                        return arr[k]
                return arr[keys[0]]
            else:
                return arr

    x_train = _load_npz_flexible("src/data/processed/x_train.npz")
    x_test = _load_npz_flexible("src/data/processed/x_test.npz")

    # data squeeze (2D to 1D)
    y_train = pd.read_csv("src/data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("src/data/processed/y_test.csv").squeeze()
    print(f"Train: {x_train.shape} | Test: {x_test.shape}")

    mlflow.set_experiment("Flood_Prediction_Models")

    # Define models and (optional) parameter grids for GridSearchCV
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {}
        },
        "KNeighborsRegressor": {
            "model": KNeighborsRegressor(),
            "params": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
        },
        "DecisionTree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {"max_depth": [3, 5, 7], "min_samples_split": [10, 20, 30], "min_samples_leaf": [5, 10, 15]}
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {"n_estimators": [50, 100], "max_depth": [3, 5, 7], "min_samples_split": [10, 20], "min_samples_leaf": [5, 10]}
        },
        "SVR": {
            "model": SVR(),
            "params": {"kernel": ["rbf", "linear"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
        },
        "Ridge": {
            "model": Ridge(),
            "params": {"alpha": [0.1, 1.0, 10.0]}
        },
        "Lasso": {
            "model": Lasso(max_iter=10000),
            "params": {"alpha": [0.001, 0.01, 0.1, 1.0]}
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [2, 3], "subsample": [0.8, 1.0], "min_samples_leaf": [5, 10]}
        }
    }

    warnings.filterwarnings("ignore")
    for model_name, config in models.items():
        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name=model_name):

            best_params = {}
            best_model = config["model"]

            if config.get("params"):
                try:
                    grid_search = GridSearchCV(
                        config["model"],
                        config["params"],
                        cv=params['model']['cv'],
                        scoring=params['model']['scoring'],
                        n_jobs=-1
                    )
                    grid_search.fit(x_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                except Exception as e:
                    print(f"GridSearch failed for {model_name}: {e}")
                    print("Falling back to default estimator fit.")
                    best_model = config["model"]
                    best_model.fit(x_train, y_train)
            else:
                best_model.fit(x_train, y_train)

            # Predictions and evaluation
            y_pred = best_model.predict(x_test)
            y_train_pred = best_model.predict(x_train)

            test_r2 = r2_score(y_test, y_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # Log to MLflow
            mlflow.log_param("model_name", model_name)
            if best_params:
                mlflow.log_params(best_params)
            mlflow.log_metric("Train_R2", train_r2)
            mlflow.log_metric("Test_R2", test_r2)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            # Print summary
            print(f"Train R2  : {train_r2:.4f}")
            print(f"Test R2   : {test_r2:.4f}")
            print(f"RMSE      : {rmse:.4f}")
            print(f"MAE       : {mae:.4f}")
            print(f"Best Params: {best_params}")

            diff = train_r2 - test_r2
            if diff > 0.1:
                print(f"OVERFIT — Difference: {diff:.4f}")
            elif test_r2 < 0.7:
                print(f"UNDERFIT — Test R2: {test_r2:.4f}")
            else:
                print("GOOD FIT!")

    print("\nAll models trained!")


if __name__ == "__main__":
    train()