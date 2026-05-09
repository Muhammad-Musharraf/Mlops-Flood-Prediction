import pandas as pd
import scipy.sparse as sp
import numpy as np
import pickle
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import yaml

# Params load 
params = yaml.safe_load(open("params.yaml")) 

# Preprocessing: Load raw data, transform features, and save processed data for training
def preprocessing():
    if "processed" in os.listdir("src/data"):
        print("Processed data already exists. Skipping preprocessing.")
        return

    df = pd.read_csv("src/data/raw/flood.csv")
    print(f"Data Loaded — Shape: {df.shape}")

    df.drop(columns=["PoliticalFactors","InadequatePlanning","PopulationScore"], axis=1, inplace=True)

    print(f"Columns after drop: {list(df.columns)}")

    x = df.drop(columns=["FloodProbability"]) 
    y = df['FloodProbability']

    print(f"Features: {x.shape} | Target: {y.shape}")

    numeric_cols = x.columns.tolist()  # All are numeric  

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols)
    ])

    x = preprocessor.fit_transform(x)
    print(f"After preprocessing — Shape: {x.shape}")


    # Train-test split using params from params.yaml
    x_train, x_test, y_train, y_test = train_test_split(
        #x,y, test_size=0.2, random_state=42
        x, y, test_size=params["data"]["test_size"], random_state=params["data"]["random_state"]
    )
    print(f"Train: {x_train.shape} | Test: {x_test.shape}")

    # Processed data save 
    os.makedirs("src/data/processed", exist_ok=True)

    # Save processed arrays in compressed .npz format and target variable in .csv
    np.savez_compressed("src/data/processed/x_train.npz", x_train=x_train)
    np.savez_compressed("src/data/processed/x_test.npz", x_test=x_test)
    y_train.to_csv("src/data/processed/y_train.csv", index=False)
    y_test.to_csv("src/data/processed/y_test.csv", index=False)

    # Preprocessor save to pkl 
    os.makedirs("src/models", exist_ok=True)
    with open("src/models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("Processed data saved to src/data/processed/")
    print("Preprocessor saved to src/models/preprocessor.pkl")


if __name__ == "__main__":
    preprocessing()