# Supply Chain Demand Forecasting — Project

## Summary
This project builds a demand forecasting model using a supplied supply-chain dataset, creates an ML pipeline (preprocessing, model, scaler), and exposes a Streamlit dashboard for predictions and EDA.

## Files / Structure

├─ app/
│ └─ dashboard.py
├─ data/
│ └─ supply_chain_data.csv
│ └─ sample_input.csv
├─ model/
│ ├─ demand_forecasting_model.keras
│ ├─ demand_forecasting_model.h5
│ └─ scaler.save
├─ notebooks/
│ └─ data_preprocessing.ipynb
├─ reports/
│ ├─ loss_curve.png
│ ├─ actual_vs_pred.png
│ └─ final_report.pdf
├─ feature_columns.json
├─ requirements.txt
└─ README.md