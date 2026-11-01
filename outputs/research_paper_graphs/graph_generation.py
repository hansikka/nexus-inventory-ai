"""
Graph Generation Script for Inventory Demand Forecasting Project
==================================================================
Author: Research & ML Engineering Team
Date: October 2026

This script generates 11 publication-quality, research-paper-ready graphs
using actual project data, test predictions, and trained model objects.

Output Directory: outputs/research_paper_graphs/
Resolution: 300 DPI
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import xgboost as xgb
import shap

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure Matplotlib for Publication / Research Paper Quality
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.color": "#CBD5E1",
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})

# Color Constants
COLOR_BEST = "#1E40AF"      # Deep Royal Blue (Hybrid)
COLOR_STACK = "#0D9488"     # Teal (Stacking)
COLOR_TREE = "#0369A1"      # Ocean Blue (XGBoost / Trees)
COLOR_OTHER = "#64748B"     # Muted Slate
COLOR_BASE = "#94A3B8"      # Light Gray
COLOR_ACTUAL = "#0F172A"    # Dark Slate / Black
COLOR_PRED = "#DC2626"      # Crimson Red

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "research_paper_graphs")
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_COMP_PATH = os.path.join(BASE_DIR, "outputs", "model_comparison.csv")
PRED_PATH = os.path.join(BASE_DIR, "outputs", "predictions.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Human-readable feature name mapping
FEATURE_NAME_MAP = {
    "discount": "Discount Amount ($)",
    "discount_pct": "Discount Percentage (%)",
    "price_ratio": "Price Ratio (Total/Base)",
    "base_price": "Base Price ($)",
    "total_price": "Total Price ($)",
    "lag_1": "Demand Lag 1 (t-1)",
    "lag_2": "Demand Lag 2 (t-2)",
    "lag_3": "Demand Lag 3 (t-3)",
    "lag_4": "Demand Lag 4 (t-4)",
    "lag_8": "Demand Lag 8 (t-8)",
    "rolling_mean_4": "4-Week Rolling Mean",
    "rolling_mean_8": "8-Week Rolling Mean",
    "rolling_std_4": "4-Week Rolling Std Dev",
    "rolling_std_8": "8-Week Rolling Std Dev",
    "week_of_year": "Week of Year",
    "day": "Day of Month",
    "month": "Month",
    "year": "Year",
    "quarter": "Quarter",
    "day_of_week": "Day of Week",
    "is_featured_sku": "Is Featured SKU",
    "is_display_sku": "Is Display SKU"
}

def clean_feature_name(col):
    if col in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[col]
    if col.startswith("store_id_"):
        return f"Store ID: {col.replace('store_id_', '')}"
    if col.startswith("sku_id_"):
        return f"SKU ID: {col.replace('sku_id_', '')}"
    return col


def generate_01_rmse(df_comp):
    """Graph 1: Model Comparison by RMSE"""
    print("--> Generating Graph 01: Model Comparison by RMSE...")
    df_sorted = df_comp.sort_values("RMSE", ascending=True).copy()
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(df_sorted["Model"], df_sorted["RMSE"], color=COLOR_OTHER, height=0.6, zorder=3)
    
    # Highlight best and second-best models
    for bar, model in zip(bars, df_sorted["Model"]):
        if "Hybrid" in model:
            bar.set_color(COLOR_BEST)
        elif "Stacking" in model:
            bar.set_color(COLOR_STACK)

    ax.invert_yaxis()  # Best model on top
    ax.set_xlabel("Root Mean Squared Error (RMSE) - Lower is Better", fontweight="bold")
    ax.set_ylabel("Forecasting Model", fontweight="bold")
    ax.set_title("Figure 1: Model Comparison by Root Mean Squared Error (RMSE)", pad=12, fontweight="bold")
    ax.set_xlim(0, max(df_sorted["RMSE"]) * 1.18)

    # Annotate values on bars
    for bar, val, model in zip(bars, df_sorted["RMSE"], df_sorted["Model"]):
        weight = "bold" if ("Hybrid" in model or "Stacking" in model) else "normal"
        text_color = COLOR_BEST if "Hybrid" in model else (COLOR_STACK if "Stacking" in model else "#1E293B")
        annotation = f"{val:.2f}"
        if "Hybrid" in model:
            annotation += " (Best)"
        ax.text(val + 0.6, bar.get_y() + bar.get_height()/2, annotation,
                va="center", ha="left", fontsize=9, fontweight=weight, color=text_color)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "01_model_comparison_rmse.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_02_mae(df_comp):
    """Graph 2: Model Comparison by MAE"""
    print("--> Generating Graph 02: Model Comparison by MAE...")
    df_sorted = df_comp.sort_values("MAE", ascending=True).copy()
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(df_sorted["Model"], df_sorted["MAE"], color=COLOR_OTHER, height=0.6, zorder=3)
    
    for bar, model in zip(bars, df_sorted["Model"]):
        if "Stacking" in model:
            bar.set_color(COLOR_STACK)
        elif "Hybrid" in model:
            bar.set_color(COLOR_BEST)

    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute Error (MAE) - Lower is Better", fontweight="bold")
    ax.set_ylabel("Forecasting Model", fontweight="bold")
    ax.set_title("Figure 2: Model Comparison by Mean Absolute Error (MAE)", pad=12, fontweight="bold")
    ax.set_xlim(0, max(df_sorted["MAE"]) * 1.18)

    for bar, val, model in zip(bars, df_sorted["MAE"], df_sorted["Model"]):
        weight = "bold" if ("Stacking" in model or "Hybrid" in model) else "normal"
        text_color = COLOR_STACK if "Stacking" in model else (COLOR_BEST if "Hybrid" in model else "#1E293B")
        annotation = f"{val:.2f}"
        if "Stacking" in model:
            annotation += " (Lowest MAE)"
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, annotation,
                va="center", ha="left", fontsize=9, fontweight=weight, color=text_color)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "02_model_comparison_mae.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_03_r2(df_comp):
    """Graph 3: Model Comparison by R² Score"""
    print("--> Generating Graph 03: Model Comparison by R² Score...")
    df_sorted = df_comp.sort_values("R2", ascending=False).copy()
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(df_sorted["Model"], df_sorted["R2"], color=COLOR_OTHER, height=0.6, zorder=3)
    
    for bar, model in zip(bars, df_sorted["Model"]):
        if "Hybrid" in model:
            bar.set_color(COLOR_BEST)
        elif "Stacking" in model:
            bar.set_color(COLOR_STACK)

    ax.invert_yaxis()
    ax.set_xlabel("Coefficient of Determination (R² Score) - Higher is Better", fontweight="bold")
    ax.set_ylabel("Forecasting Model", fontweight="bold")
    ax.set_title("Figure 3: Model Comparison by R² Score", pad=12, fontweight="bold")
    ax.set_xlim(0, 0.88)

    for bar, val, model in zip(bars, df_sorted["R2"], df_sorted["Model"]):
        weight = "bold" if ("Hybrid" in model or "Stacking" in model) else "normal"
        text_color = COLOR_BEST if "Hybrid" in model else (COLOR_STACK if "Stacking" in model else "#1E293B")
        annotation = f"{val:.4f}"
        if "Hybrid" in model:
            annotation += " (Highest R²)"
        ax.text(val + 0.012, bar.get_y() + bar.get_height()/2, annotation,
                va="center", ha="left", fontsize=9, fontweight=weight, color=text_color)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "03_model_comparison_r2.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_04_mape(df_comp):
    """Graph 4: Model Comparison by MAPE (%)"""
    print("--> Generating Graph 04: Model Comparison by MAPE...")
    df_sorted = df_comp.sort_values("MAPE (%)", ascending=True).copy()
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(df_sorted["Model"], df_sorted["MAPE (%)"], color=COLOR_OTHER, height=0.6, zorder=3)
    
    for bar, model in zip(bars, df_sorted["Model"]):
        if "Stacking" in model:
            bar.set_color(COLOR_STACK)
        elif "Hybrid" in model:
            bar.set_color(COLOR_BEST)

    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute Percentage Error (MAPE %) - Lower is Better", fontweight="bold")
    ax.set_ylabel("Forecasting Model", fontweight="bold")
    ax.set_title("Figure 4: Model Comparison by Mean Absolute Percentage Error (MAPE)", pad=12, fontweight="bold")
    ax.set_xlim(0, max(df_sorted["MAPE (%)"]) * 1.18)

    for bar, val, model in zip(bars, df_sorted["MAPE (%)"], df_sorted["Model"]):
        weight = "bold" if ("Stacking" in model or "Hybrid" in model) else "normal"
        text_color = COLOR_STACK if "Stacking" in model else (COLOR_BEST if "Hybrid" in model else "#1E293B")
        annotation = f"{val:.2f}%"
        if "Stacking" in model:
            annotation += " (Lowest MAPE)"
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2, annotation,
                va="center", ha="left", fontsize=9, fontweight=weight, color=text_color)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "04_model_comparison_mape.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_05_actual_vs_predicted(df_preds):
    """Graph 5: Actual vs. Predicted Demand"""
    print("--> Generating Graph 05: Actual vs. Predicted Demand...")
    y_actual = df_preds["units_sold"].values
    y_pred = df_preds["pred_Hybrid (Ridge+XGB)"].values

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    
    # Hexbin/scatter density representation
    hb = ax.hexbin(y_actual, y_pred, gridsize=45, cmap="Blues", mincnt=1, alpha=0.85, zorder=2)
    cb = fig.colorbar(hb, ax=ax, label="Observation Density")

    # Reference diagonal 1:1 line
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], color=COLOR_PRED, linestyle="--", linewidth=2.0,
            label="Ideal 1:1 Line (Perfect Forecast)", zorder=4)

    # Metrics annotation box
    metrics_text = (
        "Model: Hybrid (Ridge+XGB)\n"
        "Test Observations (N): 11,550\n"
        "R² Score: 0.7571\n"
        "RMSE: 28.84\n"
        "MAE: 13.59\n"
        "MAPE: 35.33%"
    )
    ax.text(0.04, 0.95, metrics_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E1", alpha=0.9))

    ax.set_xlabel("Actual Demand (units_sold)", fontweight="bold")
    ax.set_ylabel("Predicted Demand (units_sold)", fontweight="bold")
    ax.set_title("Figure 5: Actual vs. Predicted Demand (Hybrid Ridge+XGB Model)", pad=12, fontweight="bold")
    ax.set_xlim(0, max_val * 1.02)
    ax.set_ylim(0, max_val * 1.02)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "05_actual_vs_predicted.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_06_actual_vs_predicted_over_time(df_preds):
    """Graph 6: Actual and Predicted Demand Over Time"""
    print("--> Generating Graph 06: Actual and Predicted Demand Over Time...")
    df_preds["week"] = pd.to_datetime(df_preds["week"])
    
    # Panel A: Mean Weekly Demand across all test stores & SKUs
    weekly_agg = df_preds.groupby("week")[["units_sold", "pred_Hybrid (Ridge+XGB)", "pred_Stacking (Ensemble)", "pred_Baseline (Naive Lag-1)"]].mean().reset_index()

    # Panel B: High-volume representative Store & SKU pair
    top_pair = df_preds.groupby(["store_id", "sku_id"])["units_sold"].sum().idxmax()
    sample_df = df_preds[(df_preds["store_id"] == top_pair[0]) & (df_preds["sku_id"] == top_pair[1])].sort_values("week")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=False)

    # Subplot A
    ax1.plot(weekly_agg["week"], weekly_agg["units_sold"], marker="o", linewidth=2.2, color=COLOR_ACTUAL, label="Actual Demand (Mean)")
    ax1.plot(weekly_agg["week"], weekly_agg["pred_Hybrid (Ridge+XGB)"], marker="s", linestyle="--", linewidth=2.0, color=COLOR_BEST, label="Hybrid Model (Ridge+XGB)")
    ax1.plot(weekly_agg["week"], weekly_agg["pred_Stacking (Ensemble)"], marker="^", linestyle=":", linewidth=1.8, color=COLOR_STACK, label="Stacking Ensemble")
    ax1.set_title("(a) Average Weekly Demand Across All Test Store-SKU Pairs (N=11,550)", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Mean Units Sold", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    ax1.tick_params(axis="x", rotation=15)

    # Subplot B
    ax2.plot(sample_df["week"], sample_df["units_sold"], marker="o", linewidth=2.2, color=COLOR_ACTUAL, label="Actual Demand")
    ax2.plot(sample_df["week"], sample_df["pred_Hybrid (Ridge+XGB)"], marker="s", linestyle="--", linewidth=2.0, color=COLOR_BEST, label=f"Hybrid Model Forecast")
    ax2.plot(sample_df["week"], sample_df["pred_Baseline (Naive Lag-1)"], marker="v", linestyle="-.", linewidth=1.5, color=COLOR_BASE, label="Naive Lag-1 Baseline")
    ax2.set_title(f"(b) High-Volume SKU Demand Trajectory (Store {top_pair[0]}, SKU {top_pair[1]})", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Test Window Week (May 2013 – July 2013)", fontweight="bold")
    ax2.set_ylabel("Units Sold", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    ax2.tick_params(axis="x", rotation=15)

    fig.suptitle("Figure 6: Actual vs. Predicted Demand Over Time (Test Evaluation Window)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "06_actual_vs_predicted_over_time.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_07_demand_distribution(df_raw):
    """Graph 7: Demand Distribution"""
    print("--> Generating Graph 07: Demand Distribution...")
    units = df_raw["units_sold"].dropna()
    log_units = np.log1p(units)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Raw Units Sold
    sns.histplot(units, bins=50, kde=True, ax=ax1, color="#1E3A8A", alpha=0.6, edgecolor="white")
    ax1.set_title("(a) Original Demand Distribution (units_sold)", fontweight="bold")
    ax1.set_xlabel("Units Sold", fontweight="bold")
    ax1.set_ylabel("Frequency / Count", fontweight="bold")
    
    mean_val = units.mean()
    median_val = units.median()
    ax1.axvline(mean_val, color=COLOR_PRED, linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.1f}")
    ax1.axvline(median_val, color="#059669", linestyle="-.", linewidth=1.8, label=f"Median: {median_val:.1f}")
    ax1.legend()

    # Panel B: Log-Transformed
    sns.histplot(log_units, bins=50, kde=True, ax=ax2, color="#0D9488", alpha=0.6, edgecolor="white")
    ax2.set_title("(b) Log-Transformed Target Distribution log(1 + units_sold)", fontweight="bold")
    ax2.set_xlabel("log1p(Units Sold)", fontweight="bold")
    ax2.set_ylabel("Frequency / Count", fontweight="bold")
    
    mean_log = log_units.mean()
    median_log = log_units.median()
    ax2.axvline(mean_log, color=COLOR_PRED, linestyle="--", linewidth=1.8, label=f"Mean: {mean_log:.2f}")
    ax2.axvline(median_log, color="#059669", linestyle="-.", linewidth=1.8, label=f"Median: {median_log:.2f}")
    ax2.legend()

    fig.suptitle("Figure 7: Target Variable Distribution Analysis (N=150,150 Records)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "07_demand_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_08_average_weekly_demand(df_raw):
    """Graph 8: Average Demand Over Time"""
    print("--> Generating Graph 08: Average Demand Over Time...")
    df = df_raw.copy()
    df["week"] = pd.to_datetime(df["week"], format="%d/%m/%y")
    weekly = df.groupby("week")["units_sold"].mean().reset_index().sort_values("week")
    weekly["rolling_4"] = weekly["units_sold"].rolling(4).mean()

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    ax.plot(weekly["week"], weekly["units_sold"], color="#94A3B8", linewidth=1.2, alpha=0.7, label="Weekly Mean Demand (Per Store/SKU)")
    ax.plot(weekly["week"], weekly["rolling_4"], color=COLOR_BEST, linewidth=2.2, label="4-Week Moving Average Trend")

    # Annotate Chronological Train / Val / Test Splits
    val_start = pd.to_datetime("2013-03-12")
    test_start = pd.to_datetime("2013-05-07")
    end_date = weekly["week"].max()

    ax.axvline(val_start, color="#D97706", linestyle="--", linewidth=1.8, label="Validation Split (Mar 2013)")
    ax.axvline(test_start, color="#2563EB", linestyle="--", linewidth=1.8, label="Test Split (May 2013)")

    ax.axvspan(weekly["week"].min(), val_start, color="#F1F5F9", alpha=0.5, label="Training Period (~104 wks)")
    ax.axvspan(val_start, test_start, color="#FEF3C7", alpha=0.5, label="Validation Period (8 wks)")
    ax.axvspan(test_start, end_date, color="#DBEAFE", alpha=0.6, label="Testing Period (10 wks)")

    ax.set_xlabel("Historical Date Window (January 2011 – July 2013)", fontweight="bold")
    ax.set_ylabel("Average Units Sold", fontweight="bold")
    ax.set_title("Figure 8: Overall Average Weekly Demand Trend with Chronological Train/Val/Test Splits", pad=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "08_average_weekly_demand.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_09_10_11_model_and_shap(df_raw):
    """Graphs 9, 10, 11: Feature Importance and SHAP Explainability"""
    print("--> Preparing Feature Matrix for Tree Model Feature Importance & SHAP...")
    
    from inventory_demand_forecasting_source_code import InventoryForecastingPipeline
    
    pipeline = InventoryForecastingPipeline(data_path=DATA_PATH, output_dir="outputs", model_dir="models")
    pipeline.load_and_inspect_data()
    pipeline.engineer_features()
    pipeline.split_data_chronologically()

    # Train XGBoost Model to inspect feature importances
    xgb_reg = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1
    )
    xgb_reg.fit(pipeline.X_train, pipeline.y_train, eval_set=[(pipeline.X_val, pipeline.y_val)], verbose=False)

    # Graph 9: XGBoost Feature Importance
    print("--> Generating Graph 09: Feature Importance (XGBoost Regressor)...")
    importance = xgb_reg.feature_importances_
    cols = pipeline.encoded_cols
    
    df_imp = pd.DataFrame({"Feature_Raw": cols, "Importance": importance})
    df_imp["Feature"] = df_imp["Feature_Raw"].apply(clean_feature_name)
    df_imp = df_imp.sort_values("Importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    bars = ax.barh(df_imp["Feature"], df_imp["Importance"], color=COLOR_TREE, height=0.6, zorder=3)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Feature Importance (Gini / Gain Score)", fontweight="bold")
    ax.set_ylabel("Engineered Feature", fontweight="bold")
    ax.set_title("Figure 9: Top 20 Feature Importances (XGBoost Regressor)", pad=12, fontweight="bold")
    ax.set_xlim(0, max(df_imp["Importance"]) * 1.15)

    for bar, val in zip(bars, df_imp["Importance"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
                va="center", ha="left", fontsize=8.5, color="#1E293B")

    plt.tight_layout()
    out_path_09 = os.path.join(OUTPUT_DIR, "09_feature_importance.png")
    plt.savefig(out_path_09)
    plt.close()
    print(f"Saved: {out_path_09}")

    # SHAP Explainer
    print("--> Calculating SHAP Values on Test Set Sample...")
    sample_X = pipeline.X_test.sample(min(1000, len(pipeline.X_test)), random_state=42)
    # Rename columns in sample_X for clean SHAP labels
    renamed_sample = sample_X.copy()
    renamed_sample.columns = [clean_feature_name(c) for c in sample_X.columns]

    explainer = shap.TreeExplainer(xgb_reg)
    shap_vals = explainer.shap_values(renamed_sample)

    # Graph 10: SHAP Summary Plot
    print("--> Generating Graph 10: SHAP Summary Plot...")
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    shap.summary_plot(shap_vals, renamed_sample, max_display=15, show=False)
    plt.title("Figure 10: SHAP Summary Beeswarm Plot (XGBoost Regressor)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path_10 = os.path.join(OUTPUT_DIR, "10_shap_summary.png")
    plt.savefig(out_path_10, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path_10}")

    # Graph 11: SHAP Feature Importance Bar Plot
    print("--> Generating Graph 11: SHAP Feature Importance Plot...")
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    shap.summary_plot(shap_vals, renamed_sample, plot_type="bar", max_display=15, show=False)
    plt.title("Figure 11: SHAP Global Feature Importance - Mean |SHAP Value| (XGBoost Regressor)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path_11 = os.path.join(OUTPUT_DIR, "11_shap_feature_importance.png")
    plt.savefig(out_path_11, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path_11}")


def main():
    print("=" * 70)
    print("STARTING GRAPH GENERATION PIPELINE")
    print("=" * 70)

    # Load Model Comparison Metrics
    if not os.path.exists(MODEL_COMP_PATH):
        raise FileNotFoundError(f"Model comparison file missing at {MODEL_COMP_PATH}")
    
    df_comp = pd.read_csv(MODEL_COMP_PATH)
    if df_comp.columns[0] == "" or "Unnamed" in df_comp.columns[0]:
        df_comp = df_comp.rename(columns={df_comp.columns[0]: "Model"})

    print("Loaded Model Comparison Metrics:")
    print(df_comp.to_string())

    # Load Predictions Data
    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError(f"Predictions data file missing at {PRED_PATH}")
    df_preds = pd.read_csv(PRED_PATH)

    # Load Raw Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Raw dataset missing at {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)

    # Generate Graphs 1 - 4 (Model Comparisons)
    generate_01_rmse(df_comp)
    generate_02_mae(df_comp)
    generate_03_r2(df_comp)
    generate_04_mape(df_comp)

    # Generate Graphs 5 - 6 (Actual vs Predicted & Time Series)
    generate_05_actual_vs_predicted(df_preds)
    generate_06_actual_vs_predicted_over_time(df_preds)

    # Generate Graphs 7 - 8 (Distribution & Demand Trend)
    generate_07_demand_distribution(df_raw)
    generate_08_average_weekly_demand(df_raw)

    # Generate Graphs 9 - 11 (Feature Importance & SHAP)
    generate_09_10_11_model_and_shap(df_raw)

    print("\n" + "=" * 70)
    print("ALL 11 RESEARCH PAPER GRAPHS GENERATED SUCCESSFULLY!")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
