# Supply Chain Demand Forecasting — Final Report (1 Page)

## 1. Project Objective  
The goal of this project is to build a predictive model that forecasts the **Number of Products Sold** for supply chain items, helping organizations optimize inventory, manage procurement, and improve supply-chain decision-making.  
A complete ML pipeline was implemented including preprocessing, model training, evaluation, and deployment through an interactive Streamlit dashboard.

---

## 2. Dataset Summary  
- Total rows: ~100  
- Features include Pricing, Availability, Customer Demographics, Location, Lead Times, Production Volumes, Manufacturing Costs, Shipping Info, and Supplier Info.  
- Dataset does *not* contain temporal features (Date), so forecasting is treated as a regression task predicting `Number of products sold`.  
- Categorical columns were one-hot encoded; numerical columns were scaled using StandardScaler.

---

## 3. Preprocessing Performed  
- One-hot encoding for all categorical features  
- StandardScaler applied to numerical features  
- Train–Test Split: **80% / 20%**  
- Saved artifacts:  
  - `demand_forecasting_model.keras`  
  - `scaler.save`  
  - `feature_columns.json`

---

## 4. Models Trained  

### **Neural Network (Primary Model)**  
- Architecture: Dense → (128 → 64 → 32 → 1)  
- Activation: ReLU  
- Optimizer: Adam  
- Loss: MSE  
- Includes validation split & early stopping  

**NN Performance:**  
- **MSE:** 135621.72  
- **MAE:** 326.04  
- **R²:** -0.42114  

---

### **RandomForest (Baseline Model)**  
- 300 trees, RandomState=42  
- Provides a classical ML benchmark  

**RF Performance:**  
- **MSE:** 160405.03  
- **MAE:** 359.63  
- **R²:** -0.60084  

---

## 5. Evaluation Visuals  
- **Neural Network Loss Curve:** `reports/nn_loss_curve.png`  
- **NN Actual vs Predicted:** `reports/nn_actual_vs_pred.png`  
- **RF Actual vs Predicted:** `reports/rf_actual_vs_pred.png`  
- **RF Feature Importances:** `reports/rf_feature_importances.png`  

These plots visualize prediction quality, learning behavior, and the impact of key features.

---

## 6. Deployment  
A fully interactive dashboard was built using **Streamlit** (`app/dashboard.py`), enabling:  
- Dataset preview & basic EDA  
- Batch predictions via CSV upload  
- Interactive single-row predictions  
- Automatic preprocessing (encoding + scaling)  
- Prediction downloads  

Run locally:

```bash
conda activate supplychain
cd E:\SupplyChainProject
streamlit run app/dashboard.py
