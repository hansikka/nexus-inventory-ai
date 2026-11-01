# Inventory Demand Forecasting System

An end-to-end, production-grade Machine Learning solution for **Inventory Demand Forecasting** built with Python, scikit-learn, XGBoost, LightGBM, ExtraTrees, Stacking Ensembles, Hybrid Residual Models, and SHAP explainability.

---

## 📁 Project Structure

```text
inventory ml project/
│
├── data/
│   └── train.csv                 # Primary weekly store-SKU inventory dataset
│
├── src/
│   └── inventory_forecasting.py   # Main pipeline script & forecast_demand() inference function
│
├── models/
│   ├── best_model.pkl            # Serialized best model (Hybrid Ridge + XGBoost)
│   └── preprocessor.pkl          # Metadata and column encodings
│
├── outputs/
│   ├── model_comparison.csv      # Detailed metric comparison across all models
│   ├── predictions.csv           # Test set actuals vs predictions from all models
│   ├── research_paper_graphs/    # 🏆 High-resolution 300 DPI research paper graphs suite
│   │   ├── 01_model_comparison_rmse.png
│   │   ├── 02_model_comparison_mae.png
│   │   ├── 03_model_comparison_r2.png
│   │   ├── 04_model_comparison_mape.png
│   │   ├── 05_actual_vs_predicted.png
│   │   ├── 06_actual_vs_predicted_over_time.png
│   │   ├── 07_demand_distribution.png
│   │   ├── 08_average_weekly_demand.png
│   │   ├── 09_feature_importance.png
│   │   ├── 10_shap_summary.png
│   │   ├── 11_shap_feature_importance.png
│   │   ├── graph_generation.py   # Standalone graph generation script
│   │   ├── graph_data_summary.txt # Summary of data & methods for paper writing
│   │   ├── graph_manifest.csv    # Manifest detailing data sources & graph metadata
│   │   └── README_graphs.txt     # Instructions and formal figure captions
│   ├── target_distribution.png   # Target distribution & log-transformation plot
│   ├── actual_vs_predicted.png   # Scatter plot with y=x line for best model
│   ├── model_comparison.png      # Bar chart comparing RMSE and MAE across models
│   ├── demand_trend.png          # Time-series weekly demand forecasting trend
│   ├── shap_summary.png          # SHAP summary plot showing feature directionality & impact
│   └── shap_importance.png       # SHAP global feature importance ranking
│
└── README.md                     # Comprehensive project documentation
```

---

## 📊 Model Benchmark Results

All models were trained on historical data up to March 2013, validated on an 8-week validation window, and evaluated on an untouched **10-week test window** (May 2013 - July 2013).

| Model | RMSE (Units) | MAE (Units) | $R^2$ Score | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid (Ridge + XGBoost)** 🏆 | **28.84** | **13.59** | **0.7571** | **35.33%** |
| **Stacking (Ensemble)** | 30.34 | 13.49 | 0.7312 | 34.11% |
| **ExtraTrees** | 31.32 | 13.76 | 0.7135 | 35.84% |
| **XGBoost** | 31.53 | 13.87 | 0.7097 | 35.99% |
| **Random Forest** | 32.44 | 14.08 | 0.6926 | 36.18% |
| **LightGBM** | 34.12 | 14.54 | 0.6599 | 36.71% |
| **Baseline (Naive Lag-1)** | 43.96 | 20.41 | 0.4355 | 53.77% |

---

## 🚀 Key Innovations & Methodology

1. **Strict Time-Series Anti-Leakage Design**:
   - Time-series lag features (`lag_1`, `lag_2`, `lag_3`, `lag_4`, `lag_8`) and rolling statistics (`rolling_mean_4`, `rolling_mean_8`, `rolling_std_4`, `rolling_std_8`) are computed strictly using past observations (`.shift(1)`).
2. **Chronological Splitting**:
   - No random `train_test_split`. Data is split chronologically into Train (104 weeks), Validation (8 weeks), and Test (10 weeks).
3. **Outlier Safety**:
   - Trimming top 1% outliers is applied exclusively to the training set; test data is left raw for realistic evaluation.
4. **Log Transformation**:
   - Right-skewed `units_sold` target is trained using $\log(1 + y)$ and predictions are inverted via $\max(0, e^{\hat{y}} - 1)$.
5. **Hybrid Architecture**:
   - A Ridge Linear Model captures linear price/discount elasticity trends, while XGBoost predicts non-linear store/SKU residual demand. This achieves the lowest RMSE (28.84) and highest $R^2$ (0.7571).

---

## 🛠️ Usage Instructions

### Running the Full Pipeline

```bash
python src/inventory_forecasting.py
```

### Making Predictions on New Data

```python
import pandas as pd
from src.inventory_forecasting import forecast_demand

new_data = pd.read_csv("data/train.csv").head(10)
predictions = forecast_demand(new_data)
### Regenerating Research Paper Graphs

To regenerate the publication-quality 300 DPI graphs:

```bash
python outputs/research_paper_graphs/graph_generation.py
```

