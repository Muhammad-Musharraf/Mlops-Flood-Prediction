from fastapi import FastAPI
import pickle
import mlflow.sklearn
import pandas as pd
import mlflow
from dotenv import load_dotenv
import os

load_dotenv()  

# Set MLflow Tracking URI from environment variable (username, uri, password) of daghub for remote storage
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

app = FastAPI()

# MLflow Tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(
    model_uri="models:/Flood_Prediction_Model/1"
)

with open("src/models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)


@app.get("/")
def home():
    return {"message": "Flood Prediction API"}


@app.post("/predict")
def predict(
    ship_type: str,
    month: str,
    distance: float,
    fuel_type: str,
    weather_conditions: str,
    engine_efficiency: float
):
    input_data = pd.DataFrame([{
        "ship_type": ship_type,
        "month": month,
        "distance": distance,
        "fuel_type": fuel_type,
        "weather_conditions": weather_conditions,
        "engine_efficiency": engine_efficiency
    }])

    input_transformed = preprocessor.transform(input_data)
    prediction = model.predict(input_transformed)

    return {
        "predicted_fuel_consumption": round(float(prediction[0]), 2),
        "unit": "Litres"
    }