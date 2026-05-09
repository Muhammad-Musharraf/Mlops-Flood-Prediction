import pandas as pd
import pickle
import mlflow.sklearn


def predict():

   
    # 1. Model Load 
    model = mlflow.sklearn.load_model(
        model_uri="models:/Flood_Prediction_Model/1"
    )
    print("Model Loaded!")

   
    # 2. Preprocessor Load 
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    print("Preprocessor Loaded!")

    
    # 3. New Unseen Data
    new_data = pd.DataFrame([
        {
            "ship_type": "Cargo",
            "month": "January",
            "distance": 500,
            "fuel_type": "Diesel",
            "weather_conditions": "Clear",
            "engine_efficiency": 0.85
        },
        {
            "ship_type": "Tanker",
            "month": "June",
            "distance": 1200,
            "fuel_type": "HFO",
            "weather_conditions": "Stormy",
            "engine_efficiency": 0.72
        }
    ])

    print(f"\nNew Data:\n{new_data}")


    # 4. Preprocess data
    new_data_transformed = preprocessor.transform(new_data)

    # 5. Prediction 
    predictions = model.predict(new_data_transformed)

    print("\n--- Predictions ---")
    for i, pred in enumerate(predictions):
        print(f"Record {i+1} — Predicted Fuel Consumption: {pred:.2f} Litres")


if __name__ == "__main__":
    predict()