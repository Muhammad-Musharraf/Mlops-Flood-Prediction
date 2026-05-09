import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os

# Data Ingestion: Load raw data from Kaggle and save locally for preprocessing
def data_ingestion():
    if "data" in os.listdir("src") and "raw" in os.listdir("src/data") and "flood.csv" in os.listdir("src/data/raw"):
        print("Raw data already exists. Skipping ingestion.")
        return
    
    print("Data ingestion start...")

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "naiyakhalid/flood-prediction-dataset",
        "flood.csv",
    )

    print(f"Data Loaded from Kaggle — Shape: {df.shape}")
    print(f"First 5 Records:\n{df.head()}")
    print(f"Null Values:\n{df.isnull().sum()}")
    print(f"Duplicated Rows: {df.duplicated().sum()}")

    os.makedirs("src/data/raw", exist_ok=True)
    df.to_csv("src/data/raw/flood.csv", index=False)
    print("Raw data saved to src/data/raw/flood.csv")


if __name__ == "__main__":
    data_ingestion()