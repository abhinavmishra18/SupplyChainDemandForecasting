# test_predict.py
"""
Quick artifact smoke-test:
- loads model (Keras), scaler (joblib), and feature_columns.json
- creates a small sample row from data/supply_chain_data.csv
- runs prepare -> scale -> model.predict and prints result
Run with: python test_predict.py
"""

import json
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL = os.path.join(ROOT, "model", "demand_forecasting_model.keras")
SCALER = os.path.join(ROOT, "model", "scaler.save")
FEATURES = os.path.join(ROOT, "feature_columns.json")
DATA = os.path.join(ROOT, "data", "supply_chain_data.csv")

def prepare_rows_for_prediction(df_rows, feature_columns, scaler):
    df_enc = pd.get_dummies(df_rows)
    df_enc = df_enc.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_enc.values)
    return X_scaled, df_enc

def main():
    print("Working directory:", ROOT)

    # existence checks
    for p in [MODEL, SCALER, FEATURES, DATA]:
        print("Check:", os.path.relpath(p, ROOT), "->", "OK" if os.path.exists(p) else "MISSING")
        if not os.path.exists(p):
            print("ERROR: Missing:", p)
            sys.exit(2)

    print("Loading model...")
    model = keras.models.load_model(MODEL)
    print("Loaded model from:", MODEL)

    print("Loading scaler...")
    scaler = joblib.load(SCALER)
    print("Loaded scaler type:", type(scaler))

    print("Loading feature list...")
    with open(FEATURES, "r") as f:
        feature_columns = json.load(f)
    print("Number of features:", len(feature_columns))

    print("Loading sample data row...")
    df = pd.read_csv(DATA)
    sample = df.head(1)
    print("Sample columns:", list(sample.columns))
    print(sample.to_dict(orient="records")[0])

    print("Prepare -> scale -> predict")
    X_scaled, df_enc = prepare_rows_for_prediction(sample, feature_columns, scaler)
    pred = model.predict(X_scaled).flatten()
    print("Prediction result (array):", pred)
    print("Sample prediction (first):", float(pred[0]))

if __name__ == "__main__":
    main()
