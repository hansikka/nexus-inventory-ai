# backend/app.py
"""Simple Flask API to serve inventory demand forecasts.

Endpoints:
- GET / : serves the frontend HTML page (optional, can be served separately)
- POST /predict : accepts JSON payload with a list of records (same schema as training data) and returns predicted units_sold.

Run:
    pip install flask pandas numpy joblib
    python backend/app.py
"""

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os

# Import database module
try:
    from backend import database as db
except ImportError:
    import database as db

# Initialize database schema and seed data
db.init_db()

app = Flask(__name__, static_folder="../frontend", static_url_path="")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/predict", methods=["OPTIONS"])
def predict_options():
    return "", 204

# Try importing xgboost so unpickling works if installed
try:
    import xgboost
except ImportError:
    pass

# Load model and preprocessor paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "best_model.pkl"))
PREPROC_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "preprocessor.pkl"))

model = None
preproc = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROC_PATH):
        model = joblib.load(MODEL_PATH)
        preproc = joblib.load(PREPROC_PATH)
        print("[OK] ML Model and Preprocessor loaded successfully.")
    else:
        print("[!] Model file not found, using heuristic predictor.")
except Exception as e:
    print(f"[!] Notice: Could not load pickled model ({e}). Using heuristic predictor.")

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Replicate the feature engineering steps used in forecast_demand
    # Price features
    if "total_price" in df.columns and "base_price" in df.columns:
        df["total_price"] = df["total_price"].fillna(df["base_price"])
        df["discount"] = df["base_price"] - df["total_price"]
        df["discount_pct"] = np.where(df["base_price"] > 0, df["discount"] / df["base_price"], 0.0)
        df["price_ratio"] = np.where(df["base_price"] > 0, df["total_price"] / df["base_price"], 1.0)

    # Date features
    if "week" in df.columns:
        df["week"] = pd.to_datetime(df["week"], format="%d/%m/%y", errors="coerce")
        df["day"] = df["week"].dt.day
        df["month"] = df["week"].dt.month
        df["year"] = df["week"].dt.year
        df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["week"].dt.quarter
        df["day_of_week"] = df["week"].dt.dayofweek

    # Encode categorical columns to match training columns
    df_enc = pd.get_dummies(df, columns=["store_id", "sku_id"], drop_first=False)
    df_aligned = df_enc.reindex(columns=preproc["encoded_cols"], fill_value=0)
    return df_aligned

@app.route('/')
def serve_index():
    # Serve the static index.html from the frontend folder
    return send_from_directory(app.static_folder, "index.html")

@app.route('/login')
def serve_login():
    # Serve the login page
    return send_from_directory(app.static_folder, "login.html")

# ── Database API Endpoints ────────────────────────────────────

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    identifier = data.get("identifier", "").strip()
    password = data.get("password", "").strip()

    if not identifier or not password:
        return jsonify({"error": "Email/Username and password are required."}), 400

    user = db.verify_credentials(identifier, password)
    if user:
        return jsonify({
            "status": "success",
            "message": "Authentication successful",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"]
            }
        })
    else:
        return jsonify({"error": "Invalid credentials. Please verify your email and password."}), 401

@app.route('/api/dashboard', methods=['GET'])
def api_dashboard():
    try:
        metrics = db.fetch_dashboard_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/inventory', methods=['GET'])
def api_inventory():
    try:
        products = db.fetch_all_products()
        return jsonify({"products": products, "count": len(products)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "records" not in data:
        return jsonify({"error": "JSON body must contain a 'records' key with a list of data rows."}), 400
    try:
        df = pd.DataFrame(data["records"])
        preds = []
        if model is not None and preproc is not None:
            X = _prepare_features(df)
            # Prediction handling for hybrid model (tuple) vs single model
            if isinstance(model, tuple):
                pred_log = model[0].predict(X) + model[1].predict(X)
            else:
                pred_log = model.predict(X)
            preds = np.clip(np.expm1(pred_log), 0, None).tolist()
        else:
            # Resilient fallback if pickle model isn't active
            for _, row in df.iterrows():
                base = float(row.get("base_price", 100) or 100)
                tot = float(row.get("total_price", base) or base)
                discount = max(0.0, base - tot)
                est = max(10, int(120 - 0.15 * tot + 1.2 * discount))
                preds.append(est)

        # Log predictions to database
        for i, row in df.iterrows():
            p_val = preds[i] if i < len(preds) else 0.0
            try:
                db.log_prediction(
                    store_id=str(row.get("store_id", "S001")),
                    sku_id=str(row.get("sku_id", "SKU-001")),
                    base_price=float(row.get("base_price", 0.0) or 0.0),
                    total_price=float(row.get("total_price", 0.0) or 0.0),
                    predicted_units=float(p_val)
                )
            except Exception:
                pass

        return jsonify({"predictions": preds})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Support dynamic PORT for cloud deployment (Render, Heroku, Railway, etc.), default to 5005
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port, debug=False)
