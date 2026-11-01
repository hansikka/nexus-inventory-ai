"""
Inventory Demand Forecasting - Complete Machine Learning Pipeline
===================================================================
Author: Antigravity AI
Date: October 2026

This module implements an end-to-end, leak-free Inventory Demand Forecasting system.
It performs data preprocessing, feature engineering (lags, rolling stats, price discounts),
time-based train/validation/test splitting, target transformation (log1p), training of multiple
ML models (Baseline, RandomForest, XGBoost, LightGBM, ExtraTrees, Stacking, Hybrid), SHAP explainability,
and generates model evaluation reports, plots, and reusable forecasting utilities.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import shap

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# Set global random state
RANDOM_STATE = 42

class InventoryForecastingPipeline:
    """
    Complete pipeline for Inventory Demand Forecasting.
    """

    def __init__(self, data_path="data/train.csv", output_dir="outputs", model_dir="models"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.model_dir = model_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.df_raw = None
        self.df_clean = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.feature_cols = None
        self.encoded_cols = None

        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.best_model_name = None
        self.best_model = None

    def load_and_inspect_data(self):
        """Loads dataset and prints summary statistics."""
        print("=" * 60)
        print("STEP 1: LOADING & INSPECTING DATASET")
        print("=" * 60)

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        self.df_raw = pd.read_csv(self.data_path)
        print(f"Dataset Shape: {self.df_raw.shape[0]} rows, {self.df_raw.shape[1]} columns")
        print(f"Columns: {list(self.df_raw.columns)}")
        print("\nMissing Values:")
        print(self.df_raw.isnull().sum())

        # Handle missing total_price by replacing with base_price
        if self.df_raw["total_price"].isnull().sum() > 0:
            print("--> Imputing missing total_price values with base_price...")
            self.df_raw["total_price"] = self.df_raw["total_price"].fillna(self.df_raw["base_price"])

        # Parse week column to datetime safely
        self.df_raw["week"] = pd.to_datetime(self.df_raw["week"], format="%d/%m/%y")
        self.df_raw = self.df_raw.sort_values(["store_id", "sku_id", "week"]).reset_index(drop=True)

        print(f"\nDate Range: {self.df_raw['week'].min().strftime('%Y-%m-%d')} to {self.df_raw['week'].max().strftime('%Y-%m-%d')}")
        print(f"Unique Stores: {self.df_raw['store_id'].nunique()}")
        print(f"Unique SKUs: {self.df_raw['sku_id'].nunique()}")
        print(f"Total Unique Weeks: {self.df_raw['week'].nunique()}")

        # Plot Target Distribution
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.hist(self.df_raw["units_sold"], bins=50, color="navy", alpha=0.7)
        plt.title("Original Target Distribution (units_sold)")
        plt.xlabel("Units Sold")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.hist(np.log1p(self.df_raw["units_sold"]), bins=50, color="teal", alpha=0.7)
        plt.title("Log-Transformed Target (log1p)")
        plt.xlabel("log1p(Units Sold)")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "target_distribution.png"), dpi=300)
        plt.close()
        print(f"Saved target distribution plot to {os.path.join(self.output_dir, 'target_distribution.png')}")

    def engineer_features(self):
        """Creates calendar, price, and demand history lag/rolling features."""
        print("\n" + "=" * 60)
        print("STEP 2: FEATURE ENGINEERING (LEAK-FREE LAGS & PRICE METRICS)")
        print("=" * 60)

        df = self.df_raw.copy()

        # 1. Price and Discount Features
        df["discount"] = df["base_price"] - df["total_price"]
        df["discount_pct"] = np.where(df["base_price"] > 0, df["discount"] / df["base_price"], 0.0)
        df["price_ratio"] = np.where(df["base_price"] > 0, df["total_price"] / df["base_price"], 1.0)

        # 2. Date Features
        df["day"] = df["week"].dt.day
        df["month"] = df["week"].dt.month
        df["year"] = df["week"].dt.year
        df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["week"].dt.quarter
        df["day_of_week"] = df["week"].dt.dayofweek

        # 3. Time-Series History Features (Grouped by Store + SKU)
        # CRITICAL: We shift by at least 1 week BEFORE rolling calculations to avoid target leakage!
        grouped = df.groupby(["store_id", "sku_id"])["units_sold"]

        df["lag_1"] = grouped.shift(1)
        df["lag_2"] = grouped.shift(2)
        df["lag_3"] = grouped.shift(3)
        df["lag_4"] = grouped.shift(4)
        df["lag_8"] = grouped.shift(8)

        df["rolling_mean_4"] = grouped.transform(lambda x: x.shift(1).rolling(4).mean())
        df["rolling_mean_8"] = grouped.transform(lambda x: x.shift(1).rolling(8).mean())
        df["rolling_std_4"] = grouped.transform(lambda x: x.shift(1).rolling(4).std())
        df["rolling_std_8"] = grouped.transform(lambda x: x.shift(1).rolling(8).std())

        # Drop record_ID identifier
        if "record_ID" in df.columns:
            df = df.drop(columns=["record_ID"])

        # Filter out initial warmup rows where lag_8 is NaN
        self.df_clean = df.dropna(subset=["lag_8", "rolling_std_8"]).reset_index(drop=True)
        print(f"Clean Feature Matrix Shape (after dropping 8-week warmup): {self.df_clean.shape}")

    def split_data_chronologically(self):
        """Splits dataset into Train, Validation, and Test sets chronologically."""
        print("\n" + "=" * 60)
        print("STEP 3: CHRONOLOGICAL TIME-BASED TRAIN / VAL / TEST SPLIT")
        print("=" * 60)

        unique_weeks = np.sort(self.df_clean["week"].unique())
        total_weeks = len(unique_weeks)

        test_weeks = unique_weeks[-10:]      # Last 10 weeks
        val_weeks = unique_weeks[-18:-10]    # Prior 8 weeks
        train_weeks = unique_weeks[:-18]     # Earlier weeks (~104 weeks)

        def format_date(d):
            return pd.Timestamp(d).strftime('%Y-%m-%d')

        print(f"Total time steps: {total_weeks} weeks")
        print(f"Training Period:   {format_date(train_weeks[0])} to {format_date(train_weeks[-1])} ({len(train_weeks)} weeks)")
        print(f"Validation Period: {format_date(val_weeks[0])} to {format_date(val_weeks[-1])} ({len(val_weeks)} weeks)")
        print(f"Testing Period:    {format_date(test_weeks[0])} to {format_date(test_weeks[-1])} ({len(test_weeks)} weeks)")

        train_df = self.df_clean[self.df_clean["week"].isin(train_weeks)].copy()
        val_df = self.df_clean[self.df_clean["week"].isin(val_weeks)].copy()
        test_df = self.df_clean[self.df_clean["week"].isin(test_weeks)].copy()

        # Trim top 1% outliers ONLY from the training dataset
        q99 = train_df["units_sold"].quantile(0.99)
        train_df = train_df[train_df["units_sold"] <= q99].copy()
        print(f"--> Trimmed top 1% outliers from Train (> {q99:.1f} units). Train rows: {len(train_df)}")

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Define feature columns
        exclude_cols = ["week", "units_sold"]
        self.feature_cols = [c for c in self.df_clean.columns if c not in exclude_cols]

        # Categorical One-Hot Encoding
        def encode_frame(df_in, columns_ref=None):
            df_enc = pd.get_dummies(df_in[self.feature_cols], columns=["store_id", "sku_id"], drop_first=False)
            if columns_ref is None:
                return df_enc
            return df_enc.reindex(columns=columns_ref, fill_value=0)

        self.X_train = encode_frame(self.train_df)
        self.encoded_cols = list(self.X_train.columns)
        self.X_val = encode_frame(self.val_df, self.encoded_cols)
        self.X_test = encode_frame(self.test_df, self.encoded_cols)

        # Targets
        self.y_train = np.log1p(self.train_df["units_sold"].values)
        self.y_val = np.log1p(self.val_df["units_sold"].values)
        self.y_test = self.test_df["units_sold"].values  # Real units for evaluation

        print(f"Final Encoded Feature Columns: {len(self.encoded_cols)}")
        print(f"Train Shape: {self.X_train.shape}, Val Shape: {self.X_val.shape}, Test Shape: {self.X_test.shape}")

    def evaluate_predictions(self, model_name, y_pred_raw):
        """Calculates evaluation metrics on real demand scale."""
        pred_clipped = np.clip(y_pred_raw, 0, None)
        rmse = np.sqrt(mean_squared_error(self.y_test, pred_clipped))
        mae = mean_absolute_error(self.y_test, pred_clipped)
        r2 = r2_score(self.y_test, pred_clipped)
        mape = np.mean(np.abs((self.y_test - pred_clipped) / np.maximum(self.y_test, 1))) * 100

        self.predictions[model_name] = pred_clipped
        self.metrics[model_name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE (%)": mape
        }

        print(f"{model_name:20s} | RMSE: {rmse:7.2f} | MAE: {mae:6.2f} | R²: {r2:.4f} | MAPE: {mape:6.2f}%")

    def build_and_train_models(self):
        """Trains Baseline, Random Forest, XGBoost, LightGBM, ExtraTrees, Stacking, and Hybrid models."""
        print("\n" + "=" * 60)
        print("STEP 4: TRAINING & EVALUATING MACHINE LEARNING MODELS")
        print("=" * 60)

        # 1. Baseline Model (Naive Lag-1)
        pred_base = self.test_df["lag_1"].values
        self.evaluate_predictions("Baseline (Naive Lag-1)", pred_base)

        # 2. Random Forest Regressor
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        pred_rf_log = rf.predict(self.X_test)
        self.models["Single Model (Random Forest)"] = rf
        self.evaluate_predictions("Single Model (Random Forest)", np.expm1(pred_rf_log))

        # 3. XGBoost Regressor
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
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        xgb_reg.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        pred_xgb_log = xgb_reg.predict(self.X_test)
        self.models["XGBoost"] = xgb_reg
        self.evaluate_predictions("XGBoost", np.expm1(pred_xgb_log))

        # 4. LightGBM Regressor
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        lgb_reg.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        pred_lgb_log = lgb_reg.predict(self.X_test)
        self.models["LightGBM"] = lgb_reg
        self.evaluate_predictions("LightGBM", np.expm1(pred_lgb_log))

        # 5. ExtraTrees Regressor
        et = ExtraTreesRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        et.fit(self.X_train, self.y_train)
        pred_et_log = et.predict(self.X_test)
        self.models["ExtraTrees"] = et
        self.evaluate_predictions("ExtraTrees", np.expm1(pred_et_log))

        # 6. CatBoost Regressor
        val_preds_list = [xgb_reg.predict(self.X_val), lgb_reg.predict(self.X_val), et.predict(self.X_val)]
        test_preds_list = [xgb_reg.predict(self.X_test), lgb_reg.predict(self.X_test), et.predict(self.X_test)]

        if HAS_CATBOOST:
            cat_reg = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                random_seed=RANDOM_STATE,
                verbose=0
            )
            cat_reg.fit(self.X_train, self.y_train, eval_set=(self.X_val, self.y_val), early_stopping_rounds=50)
            pred_cat_log = cat_reg.predict(self.X_test)
            self.models["CatBoost"] = cat_reg
            self.evaluate_predictions("CatBoost", np.expm1(pred_cat_log))
            val_preds_list.append(cat_reg.predict(self.X_val))
            test_preds_list.append(pred_cat_log)

        # 7. Stacking Model (Time-Safe validation blending)
        val_stack = np.column_stack(val_preds_list)
        test_stack = np.column_stack(test_preds_list)

        meta_learner = Ridge(alpha=1.0, positive=True)
        meta_learner.fit(val_stack, self.y_val)
        pred_stack_log = meta_learner.predict(test_stack)
        self.models["Stacking (Ensemble)"] = meta_learner
        self.evaluate_predictions("Stacking (Ensemble)", np.expm1(pred_stack_log))

        # 7. Hybrid Model (Ridge Baseline + XGBoost Residuals)
        linear_base = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=10.0))
        linear_base.fit(self.X_train, self.y_train)

        res_train = self.y_train - linear_base.predict(self.X_train)
        res_val = self.y_val - linear_base.predict(self.X_val)

        xgb_res = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        xgb_res.fit(self.X_train, res_train, eval_set=[(self.X_val, res_val)], verbose=False)

        pred_hybrid_log = linear_base.predict(self.X_test) + xgb_res.predict(self.X_test)
        self.models["Hybrid (Ridge+XGB)"] = (linear_base, xgb_res)
        self.evaluate_predictions("Hybrid (Ridge+XGB)", np.expm1(pred_hybrid_log))

    def generate_comparison_and_plots(self):
        """Generates summary tables, comparison plots, and SHAP explainability charts."""
        print("\n" + "=" * 60)
        print("STEP 5: GENERATING MODEL COMPARISON & VISUALIZATIONS")
        print("=" * 60)

        # 1. Comparison DataFrame
        comp_df = pd.DataFrame(self.metrics).T.sort_values("RMSE")
        comp_df.to_csv(os.path.join(self.output_dir, "model_comparison.csv"))
        print("\n=== MODEL PERFORMANCE COMPARISON ===")
        print(comp_df.to_string())

        self.best_model_name = comp_df.index[0]
        print(f"\n[BEST MODEL] BEST MODEL IDENTIFIED: {self.best_model_name}")

        # Save predictions CSV
        test_out = self.test_df[["week", "store_id", "sku_id", "units_sold"]].copy()
        for name, preds in self.predictions.items():
            test_out[f"pred_{name}"] = preds
        test_out.to_csv(os.path.join(self.output_dir, "predictions.csv"), index=False)

        # 2. Bar Chart Model Comparison
        plt.figure(figsize=(10, 5))
        x_pos = np.arange(len(comp_df))
        plt.bar(x_pos - 0.2, comp_df["RMSE"], width=0.4, label="RMSE", color="royalblue")
        plt.bar(x_pos + 0.2, comp_df["MAE"], width=0.4, label="MAE", color="darkorange")
        plt.xticks(x_pos, comp_df.index, rotation=25, ha="right")
        plt.ylabel("Error Score (Lower is Better)")
        plt.title("Model Performance Comparison (RMSE & MAE)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_comparison.png"), dpi=300)
        plt.close()

        # 3. Actual vs Predicted Scatter Plot for Best Model
        best_preds = self.predictions[self.best_model_name]
        plt.figure(figsize=(7, 6))
        plt.scatter(self.y_test, best_preds, alpha=0.3, color="midnightblue", s=15)
        lim_max = max(self.y_test.max(), best_preds.max())
        plt.plot([0, lim_max], [0, lim_max], "r--", linewidth=2, label="Perfect 1:1 Line")
        plt.xlabel("Actual Units Sold")
        plt.ylabel("Predicted Units Sold")
        plt.title(f"Best Model ({self.best_model_name}): Actual vs Predicted Demand")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "actual_vs_predicted.png"), dpi=300)
        plt.close()

        # 4. Time-Series Demand Trend Plot
        # Sample top store-SKU combination in test period
        top_pair = self.test_df.groupby(["store_id", "sku_id"]).size().index[0]
        sample_df = self.test_df[(self.test_df["store_id"] == top_pair[0]) & (self.test_df["sku_id"] == top_pair[1])].sort_values("week")
        sample_idx = sample_df.index
        sample_actual = self.y_test[self.test_df.index.isin(sample_idx)]
        sample_pred = best_preds[self.test_df.index.isin(sample_idx)]

        plt.figure(figsize=(10, 4))
        plt.plot(sample_df["week"], sample_actual, label="Actual Demand", marker="o", linewidth=2, color="black")
        plt.plot(sample_df["week"], sample_pred, label=f"Predicted ({self.best_model_name})", marker="s", linestyle="--", linewidth=2, color="crimson")
        plt.title(f"Weekly Demand Forecast Trend (Store {top_pair[0]}, SKU {top_pair[1]})")
        plt.xlabel("Week")
        plt.ylabel("Units Sold")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "demand_trend.png"), dpi=300)
        plt.close()

        # 5. SHAP Explainability Plot (using XGBoost or LightGBM model)
        tree_model = self.models.get("XGBoost", self.models.get("LightGBM"))
        if tree_model is not None:
            print("\n--> Generating SHAP Explainability Plots...")
            sample_X = self.X_test.sample(min(1000, len(self.X_test)), random_state=RANDOM_STATE)
            explainer = shap.TreeExplainer(tree_model)
            shap_vals = explainer.shap_values(sample_X)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, sample_X, max_display=15, show=False)
            plt.title("SHAP Feature Summary (XGBoost)", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "shap_summary.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, sample_X, plot_type="bar", max_display=15, show=False)
            plt.title("SHAP Feature Importance (XGBoost)", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "shap_importance.png"), dpi=300)
            plt.close()
            print("Saved SHAP summary and importance plots to outputs/")

    def save_artifacts(self):
        """Saves best model and preprocessor metadata for future inference."""
        print("\n" + "=" * 60)
        print("STEP 6: SERIALIZING BEST MODEL & PREPROCESSOR ARTIFACTS")
        print("=" * 60)

        best_obj = self.models.get(self.best_model_name, self.models["XGBoost"])
        joblib.dump(best_obj, os.path.join(self.model_dir, "best_model.pkl"))

        preprocessor = {
            "feature_cols": self.feature_cols,
            "encoded_cols": self.encoded_cols,
            "store_ids": list(self.df_raw["store_id"].unique()),
            "sku_ids": list(self.df_raw["sku_id"].unique()),
            "best_model_name": self.best_model_name
        }
        joblib.dump(preprocessor, os.path.join(self.model_dir, "preprocessor.pkl"))
        print(f"Saved best model to {os.path.join(self.model_dir, 'best_model.pkl')}")
        print(f"Saved preprocessor to {os.path.join(self.model_dir, 'preprocessor.pkl')}")

    def run_all(self):
        """Executes full pipeline end-to-end."""
        self.load_and_inspect_data()
        self.engineer_features()
        self.split_data_chronologically()
        self.build_and_train_models()
        self.generate_comparison_and_plots()
        self.save_artifacts()
        print("\n[SUCCESS] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")

def forecast_demand(new_data, model_path="models/best_model.pkl", preprocessor_path="models/preprocessor.pkl"):
    """
    Inference utility function to forecast inventory demand for new data.

    Parameters:
    -----------
    new_data : pd.DataFrame
        Raw input dataframe containing required columns.
    model_path : str
        Path to saved best model pickle.
    preprocessor_path : str
        Path to saved preprocessor metadata pickle.

    Returns:
    --------
    np.ndarray : Array of predicted units_sold.
    """
    model = joblib.load(model_path)
    prep = joblib.load(preprocessor_path)

    df = new_data.copy()
    if "total_price" in df.columns and "base_price" in df.columns:
        df["total_price"] = df["total_price"].fillna(df["base_price"])
        df["discount"] = df["base_price"] - df["total_price"]
        df["discount_pct"] = np.where(df["base_price"] > 0, df["discount"] / df["base_price"], 0.0)
        df["price_ratio"] = np.where(df["base_price"] > 0, df["total_price"] / df["base_price"], 1.0)

    if "week" in df.columns:
        df["week"] = pd.to_datetime(df["week"], format="%d/%m/%y", errors="coerce")
        df["day"] = df["week"].dt.day
        df["month"] = df["week"].dt.month
        df["year"] = df["week"].dt.year
        df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["week"].dt.quarter
        df["day_of_week"] = df["week"].dt.dayofweek

    # Categorical encoding matching trained columns
    df_enc = pd.get_dummies(df, columns=["store_id", "sku_id"], drop_first=False)
    df_aligned = df_enc.reindex(columns=prep["encoded_cols"], fill_value=0)

    # Predict on log scale if model expects it
    if isinstance(model, tuple):  # Hybrid model
        pred_log = model[0].predict(df_aligned) + model[1].predict(df_aligned)
    else:
        pred_log = model.predict(df_aligned)

    pred_units = np.clip(np.expm1(pred_log), 0, None)
    return pred_units

if __name__ == "__main__":
    pipeline = InventoryForecastingPipeline(
        data_path="data/train.csv",
        output_dir="outputs",
        model_dir="models"
    )
    pipeline.run_all()
