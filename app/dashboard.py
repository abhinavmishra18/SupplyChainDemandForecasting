# app/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow import keras
import logging
import os
import time

# reduce streamlit internal logging noise (optional)
logging.getLogger('streamlit').setLevel(logging.ERROR)

st.set_page_config(page_title="Supply Chain Demand Forecast", layout="wide")

@st.cache_resource
def load_artifacts():
    model_path = "model/demand_forecasting_model.keras"
    scaler_path = "model/scaler.save"
    features_path = "feature_columns.json"
    data_path = "data/supply_chain_data.csv"

    # load model
    model = keras.models.load_model(model_path)

    # load scaler
    scaler = joblib.load(scaler_path)

    # load feature list
    with open(features_path, "r") as f:
        feature_columns = json.load(f)

    # load dataset preview
    df_sample = pd.read_csv(data_path)

    return model, scaler, feature_columns, df_sample

try:
    model, scaler, feature_columns, df_sample = load_artifacts()
except Exception as e:
    st.error("Error loading model/scaler/feature list or sample data: " + str(e))
    st.stop()

st.title("Supply Chain — Demand Forecasting")
st.write("Predict `Number of products sold` for a product row using the trained model.")

# Determine required raw columns from the sample dataset (used for validation)
REQUIRED_RAW_COLUMNS = list(df_sample.columns)

# --- Robust logs directory helper ---
def get_logs_dir(preferred="app/logs"):
    """
    Ensure a usable logs directory exists. If preferred exists and is a directory -> return it.
    If preferred exists but is a file, create a timestamped alternative directory and return that.
    If preferred does not exist, attempt to create it and return it.
    On failure, return None.
    """
    try:
        if os.path.isdir(preferred):
            return preferred
        if os.path.exists(preferred):
            # exists but not a directory (likely a file) -> create alternative directory
            alt = f"{preferred}_dir_{time.strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(alt, exist_ok=True)
            st.warning(f"'{preferred}' exists and is not a directory. Using new logs dir: `{alt}`.")
            return alt
        # preferred does not exist -> create it
        os.makedirs(preferred, exist_ok=True)
        return preferred
    except Exception as e:
        st.error(f"Could not create logs directory (tried '{preferred}'): {e}")
        return None

LOGS_DIR = get_logs_dir("app/logs")  # resolve once at startup

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    show_head = st.checkbox("Show dataset head", value=True)
    show_eda = st.checkbox("Show simple EDA", value=True)
    st.markdown("---")
    st.caption("Upload raw CSV rows (not one-hot encoded). The app will encode and align with training features.")

# EDA & preview
if show_head:
    st.subheader("Dataset preview")
    st.dataframe(df_sample.head())

if show_eda:
    st.subheader("Basic EDA")
    col1, col2 = st.columns(2)
    with col1:
        if 'Revenue generated' in df_sample.columns:
            st.write("Revenue (top values)")
            st.bar_chart(df_sample['Revenue generated'].value_counts().head(20))
    with col2:
        if 'Production volumes' in df_sample.columns:
            st.write("Production volumes (sample)")
            st.line_chart(df_sample['Production volumes'].head(50))

st.markdown("---")

# helper: encode, reindex to training features, scale
def prepare_rows_for_prediction(df_rows, feature_columns, scaler):
    """
    Input: raw dataframe rows (same raw columns as original dataset)
    Output: scaled numpy array (aligned with training features), and encoded dataframe
    """
    df_enc = pd.get_dummies(df_rows)
    # Reindex to match training feature columns. Missing -> 0
    df_enc = df_enc.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_enc.values)
    return X_scaled, df_enc

# Section: CSV uploader for batch predictions (with validation + logging)
st.subheader("Upload CSV (batch prediction)")
uploaded = st.file_uploader("Upload CSV (raw columns, not encoded)", type=["csv"])
if uploaded is not None:
    try:
        df_uploaded = pd.read_csv(uploaded)
        st.write(f"Uploaded rows: {df_uploaded.shape[0]}")
        st.dataframe(df_uploaded.head())

        # ---- VALIDATION: check required raw columns ----
        missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df_uploaded.columns]
        if missing_cols:
            st.error(
                "Uploaded CSV is missing required columns. "
                "Please include these columns (case-sensitive):\n\n" + ", ".join(missing_cols)
            )
            st.info("Tip: Use `data/sample_input.csv` as a template for column names and order.")
        else:
            # spinner while preparing & predicting
            with st.spinner("Running predictions..."):
                X_scaled, df_enc = prepare_rows_for_prediction(df_uploaded, feature_columns, scaler)
                preds = model.predict(X_scaled).flatten()

            df_out = df_uploaded.copy()
            df_out["Predicted_Number_of_products_sold"] = preds

            # --- Logging: save predictions with timestamp ---
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_fname = None
            if LOGS_DIR:
                log_fname = os.path.join(LOGS_DIR, f"preds_batch_{timestamp}.csv")
                try:
                    df_out.to_csv(log_fname, index=False)
                except Exception as e:
                    st.warning(f"Warning: could not save predictions log to `{log_fname}`: {e}")
                    log_fname = None
            else:
                st.warning("No writable logs directory is available; skipping saving prediction logs.")

            st.success("Predictions complete")
            if log_fname:
                st.write(f"Saved predictions to `{log_fname}`")
            st.dataframe(df_out.head(20))
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error("Prediction failed: " + str(e))

st.markdown("---")

# Interactive single-row prediction form
st.subheader("Interactive single-row prediction")

with st.form("single_row_form"):
    # create inputs for a small set of raw fields present in your dataset
    # expand this section to include any raw columns you want to allow users to set
    c1, c2, c3 = st.columns(3)
    product_type = c1.selectbox("Product type", options=sorted(df_sample['Product type'].unique()))
    price = c2.number_input("Price", min_value=0.0, value=float(df_sample['Price'].median()))
    availability = c3.number_input("Availability", min_value=0, value=int(df_sample['Availability'].median()))
    customer_demographics = st.selectbox("Customer demographics", options=sorted(df_sample['Customer demographics'].unique()))
    location = st.selectbox("Location", options=sorted(df_sample['Location'].unique()))
    submitted = st.form_submit_button("Predict")

if submitted:
    # create a single-row raw dataframe with the selected fields
    df_row = pd.DataFrame([{
        "Product type": product_type,
        "Price": price,
        "Availability": availability,
        "Customer demographics": customer_demographics,
        "Location": location
    }])
    st.write("Input row:")
    st.write(df_row)

    # Validate that the row has required raw columns (it should, but double-check)
    missing_cols_row = [c for c in REQUIRED_RAW_COLUMNS if c not in df_row.columns]
    if missing_cols_row:
        st.error("Interactive input is missing required columns: " + ", ".join(missing_cols_row))
        st.info("This is unexpected. Ensure form fields map to dataset column names.")
    else:
        try:
            # spinner while preparing & predicting single row
            with st.spinner("Predicting..."):
                X_scaled, df_enc = prepare_rows_for_prediction(df_row, feature_columns, scaler)
                pred = model.predict(X_scaled).flatten()[0]

            # --- Logging single-row prediction ---
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_fname = None
            if LOGS_DIR:
                df_row_out = df_row.copy()
                df_row_out["Predicted_Number_of_products_sold"] = pred
                log_fname = os.path.join(LOGS_DIR, f"preds_single_{timestamp}.csv")
                try:
                    df_row_out.to_csv(log_fname, index=False)
                except Exception as e:
                    st.warning(f"Warning: could not save single-row prediction log to `{log_fname}`: {e}")
                    log_fname = None
            else:
                st.warning("No writable logs directory is available; skipping saving prediction logs.")

            st.success(f"Predicted Number of products sold: {pred:.2f}")
            if log_fname:
                st.write(f"Saved prediction to `{log_fname}`")
        except Exception as e:
            st.error("Prediction error: " + str(e))

st.markdown("---")
st.caption("Note: For production, ensure uploaded CSVs have the same raw column names used during training. "
           "This demo attempts to align one-hot columns with the training features (missing ones are filled with 0).")


