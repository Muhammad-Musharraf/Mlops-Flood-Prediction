import kagglehub
from kagglehub import KaggleDatasetAdapter
from logger import logging
import pandas as pd
import os


# Data Ingestion: Load raw data from Kaggle and save locally for preprocessing
def data_ingestion():
    logging.info("Data ingestion started.")

    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "naiyakhalid/flood-prediction-dataset",
            "flood.csv",
        )
        logging.info(f"Data loaded from Kaggle — Shape: {df.shape}")

    except Exception as e:
        logging.error(f"Failed to load dataset from Kaggle: {e}")
        raise

    try:
        logging.info(f"First 5 Records:\n{df.head()}")
        logging.info(f"Null Values:\n{df.isnull().sum()}")
        logging.info(f"Duplicated Rows: {df.duplicated().sum()}")

    except Exception as e:
        logging.warning(f"Error during data profiling: {e}")

    try:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/flood_raw.csv", index=False)
        logging.info("Raw data saved to data/raw/flood_raw.csv")

    except OSError as e:
        logging.error(f"Failed to save raw data to disk: {e}")
        raise


if __name__ == "__main__":
    try:
        data_ingestion()
    except Exception as e:
        logging.critical(f"Data ingestion pipeline failed: {e}")
        raise
    