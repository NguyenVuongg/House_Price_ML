# ============================================================
# src/train.py
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os
from preprocess import run_preprocessing, LOG_TARGET

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ────────────────────────────────────────────────────
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Models muốn thử ──────────────────────────────────────────
MODELS = {
    "LinearRegression" : LinearRegression(),
    "Ridge"            : Ridge(alpha=10),
    "Lasso"            : Lasso(alpha=100),
    "RandomForest"     : RandomForestRegressor(n_estimators=200, max_depth=10,
                                               random_state=42, n_jobs=-1),
    "GradientBoosting" : GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                   max_depth=4, random_state=42),
}


# ============================================================
# Metrics helper
# ============================================================
def calc_metrics(y_true, y_pred, log_target: bool = False) -> dict:
    """
    Nếu đã log1p(price) khi train → inverse_transform để tính metrics gốc.
    """
    if log_target:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE%": mape}


# ============================================================
# Train & compare tất cả models
# ============================================================
def train_all(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    results = []

    for name, model in MODELS.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_test  = model.predict(X_test)

        train_m = calc_metrics(y_train, pred_train, LOG_TARGET)
        test_m  = calc_metrics(y_test,  pred_test,  LOG_TARGET)

        print(f"   Train → R²={train_m['R2']:.4f}  RMSE={train_m['RMSE']:,.0f}")
        print(f"   Test  → R²={test_m['R2']:.4f}   RMSE={test_m['RMSE']:,.0f}  MAE={test_m['MAE']:,.0f}  MAPE={test_m['MAPE%']:.1f}%")

        # Detect overfitting
        gap = train_m["R2"] - test_m["R2"]
        if gap > 0.10:
            print(f"   ⚠️  Overfit gap R²: {gap:.3f} (train-test)")

        # Save model
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump(model, path)

        results.append({
            "Model"      : name,
            "Train_R2"   : round(train_m["R2"], 4),
            "Test_R2"    : round(test_m["R2"], 4),
            "Test_RMSE"  : round(test_m["RMSE"], 0),
            "Test_MAE"   : round(test_m["MAE"], 0),
            "Test_MAPE%" : round(test_m["MAPE%"], 2),
        })

    return pd.DataFrame(results).sort_values("Test_R2", ascending=False)


# ============================================================
# Feature Importance (Random Forest)
# ============================================================
def show_feature_importance(X_train: pd.DataFrame):
    rf_path = os.path.join(MODEL_DIR, "RandomForest.pkl")
    if not os.path.exists(rf_path):
        print("Chưa có RandomForest model.")
        return

    rf = joblib.load(rf_path)
    importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=False)

    print("\n📊 Feature Importance (Random Forest):")
    for feat, val in importance.items():
        bar = "█" * int(val * 40)
        print(f"  {feat:<20} {bar} {val:.4f}")

    return importance


# ============================================================
# Save best model
# ============================================================
def save_best_model(results: pd.DataFrame):
    best_name = results.iloc[0]["Model"]
    src  = os.path.join(MODEL_DIR, f"{best_name}.pkl")
    dest = os.path.join(MODEL_DIR, "house_price_model.pkl")

    model = joblib.load(src)
    joblib.dump(model, dest)
    print(f"\n🏆 Best model: {best_name}  (Test R²={results.iloc[0]['Test_R2']})")
    print(f"   Saved → {dest}")
    return best_name


# ============================================================
# Pipeline chính
# ============================================================
def run_training():
    print("\n" + "="*50)
    print("  TRAINING PIPELINE")
    print("="*50)

    X_train, X_test, y_train, y_test = run_preprocessing()

    results = train_all(X_train, X_test, y_train, y_test)

    print("\n" + "="*50)
    print("  BẢNG SO SÁNH MODELS")
    print("="*50)
    print(results.to_string(index=False))

    feat_imp = show_feature_importance(X_train)

    best = save_best_model(results)

    # Lưu bảng kết quả
    results.to_csv("output/reports/model_comparison.csv", index=False)
    print("\n✅ Training hoàn tất! Kết quả → output/reports/model_comparison.csv")

    return results, feat_imp


if __name__ == "__main__":
    run_training()