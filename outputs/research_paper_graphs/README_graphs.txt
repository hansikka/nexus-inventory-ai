README: GENERATING RESEARCH PAPER GRAPHS
========================================
Project: Inventory Demand Forecasting using Machine Learning
Output Directory: outputs/research_paper_graphs/

OVERVIEW
--------
This folder contains publication-ready, research-paper-quality graphs generated from the actual dataset, test predictions, model comparison results, and trained model artifacts of the Inventory Demand Forecasting project.

CONTENTS OF THIS DIRECTORY
--------------------------
1. Image Files (300 DPI PNG):
   - 01_model_comparison_rmse.png
   - 02_model_comparison_mae.png
   - 03_model_comparison_r2.png
   - 04_model_comparison_mape.png
   - 05_actual_vs_predicted.png
   - 06_actual_vs_predicted_over_time.png
   - 07_demand_distribution.png
   - 08_average_weekly_demand.png
   - 09_feature_importance.png
   - 10_shap_summary.png
   - 11_shap_feature_importance.png

2. Documentation & Manifest:
   - graph_manifest.csv (Detailed manifest listing data sources, methods, and types)
   - graph_data_summary.txt (Comprehensive data and analytical summary for research paper writing)
   - README_graphs.txt (This instructions file)

3. Source Code:
   - graph_generation.py (Standalone Python script used to build and save all graphs)

HOW TO REGENERATE THE GRAPHS
----------------------------
To regenerate all graphs from scratch:

1. Ensure required Python packages are installed:
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm shap catboost

2. Open terminal in project root folder:
   cd "path/to/inventory ml project"

3. Execute the standalone graph generation script:
   python outputs/research_paper_graphs/graph_generation.py

FIGURE CAPTIONS FOR RESEARCH PAPER
----------------------------------
Figure 1. Comparison of forecasting models based on Root Mean Squared Error (RMSE).
Figure 2. Comparison of forecasting models based on Mean Absolute Error (MAE).
Figure 3. Comparison of forecasting models based on Coefficient of Determination (R² Score).
Figure 4. Comparison of forecasting models based on Mean Absolute Percentage Error (MAPE %).
Figure 5. Hexbin scatter plot of actual vs. predicted demand for the top-performing Hybrid (Ridge+XGB) model on test set observations (N=11,550).
Figure 6. Actual and predicted demand trends over the 10-week test period showing (a) macro average weekly demand across all store-SKU pairs, and (b) micro demand trajectory for a high-volume SKU.
Figure 7. Demand target distribution analysis showing (a) raw right-skewed demand (units_sold), and (b) log-transformed normal distribution log1p(units_sold).
Figure 8. Historical weekly average demand trend from January 2011 to July 2013 annotated with chronological train (104 wks), validation (8 wks), and test (10 wks) splits.
Figure 9. Top 20 relative feature importances derived from the XGBoost Regressor model.
Figure 10. SHAP summary beeswarm plot illustrating feature impact direction and magnitude on log demand predictions.
Figure 11. Global feature importance ranked by mean absolute SHAP value (|SHAP value|).
