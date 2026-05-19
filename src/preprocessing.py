<<<<<<< HEAD
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
=======
import os
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline


# ── Load params ───────────────────────────────────────────────────────────────
params = yaml.safe_load(open("params.yaml"))

DATA_PATH    = params["data"]["path"]
TARGET_COL   = params["data"]["target_column"]
DROP_COLS    = params["data"]["drop_columns"]
FEATURE_COLS = params["data"]["feature_columns"]
TEST_SIZE    = params["split"]["test_size"]
RANDOM_STATE = params["split"]["random_state"]
TRAIN_OUT    = params["output"]["train_path"]   # e.g. "data/processed/train"
TEST_OUT     = params["output"]["test_path"]    # e.g. "data/processed/test"
PIPELINE_OUT = params["output"]["pipeline_path"] # e.g. "models/pipeline"


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocessing():
    # Initialise Spark
    spark = (
        SparkSession.builder
        .appName("FloodPrediction_Preprocessing")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
    print(f"Data Loaded — Rows: {df.count()} | Cols: {len(df.columns)}")

    # ── 2. Drop unwanted columns ──────────────────────────────────────────────
    df = df.drop(*DROP_COLS)
    print(f"Columns after drop: {df.columns}")

    # ── 3. Keep only needed columns (features + target) ───────────────────────
    keep_cols = FEATURE_COLS + [TARGET_COL]
    df = df.select([col(c) for c in keep_cols])

    # ── 4. Train / test split ─────────────────────────────────────────────────
    train_ratio = 1.0 - TEST_SIZE
    train_df, test_df = df.randomSplit(
        [train_ratio, TEST_SIZE], seed=RANDOM_STATE
    )
    print(f"Train rows: {train_df.count()} | Test rows: {test_df.count()}")

    # ── 5. Build PySpark ML Pipeline (VectorAssembler -> StandardScaler) ──────
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

    # Fit ONLY on training data
    pipeline_model = pipeline.fit(train_df)
    print("Pipeline fitted on training data.")

    # Transform both splits
    train_transformed = pipeline_model.transform(train_df)
    test_transformed  = pipeline_model.transform(test_df)

    # ── 6. Expand scaled vector back into individual named columns ────────────
    # PySpark stores the scaled output as a single DenseVector column.
    # We unpack it back into one column per feature so the CSV is readable.
    from pyspark.ml.functions import vector_to_array

    def expand_features(transformed_df):
        arr_df = transformed_df.withColumn("features_arr", vector_to_array("features_scaled"))
        feature_exprs = [
            col("features_arr")[i].alias(FEATURE_COLS[i])
            for i in range(len(FEATURE_COLS))
        ]
        return arr_df.select(*feature_exprs, TARGET_COL)

    train_out_df = expand_features(train_transformed)
    test_out_df  = expand_features(test_transformed)

    # ── 7. Save as CSV into data/processed  (NO pipeline saved here) ──────────
    os.makedirs("data/processed", exist_ok=True)

    # coalesce(1) writes a single CSV file instead of partitioned chunks
    train_out_df.coalesce(1).write.mode("overwrite").option("header", True).csv(TRAIN_OUT)
    test_out_df.coalesce(1).write.mode("overwrite").option("header", True).csv(TEST_OUT)

    print(f"Train CSV saved  -> {TRAIN_OUT}")
    print(f"Test  CSV saved  -> {TEST_OUT}")

    # ── 8. Save fitted pipeline separately (outside data/processed) ───────────
    os.makedirs(PIPELINE_OUT, exist_ok=True)
    pipeline_model.write().overwrite().save(PIPELINE_OUT)
    print(f"Pipeline model saved -> {PIPELINE_OUT}")

    spark.stop()
    print("Done.")
>>>>>>> 2b1d0b6 (add ETL file)


if __name__ == "__main__":
    preprocessing()