# ============================================================
# src/evaluate.py
# ============================================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from preprocess import LOG_TARGET, TARGET, MODEL_DIR

PLOT_DIR = "output/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def load_best_model():
    return joblib.load(os.path.join(MODEL_DIR, "house_price_model.pkl"))


def evaluate(X_test: pd.DataFrame, y_test: pd.Series):
    model = load_best_model()
    y_pred = model.predict(X_test)

    # Inverse log nếu cần
    if LOG_TARGET:
        y_true_plot = np.expm1(y_test)
        y_pred_plot = np.expm1(y_pred)
    else:
        y_true_plot = y_test
        y_pred_plot = y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Plot 1: Actual vs Predicted ──────────────────────────
    axes[0].scatter(y_true_plot, y_pred_plot, alpha=0.5, s=25, color="#4C72B0")
    mn = min(y_true_plot.min(), y_pred_plot.min())
    mx = max(y_true_plot.max(), y_pred_plot.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Price")
    axes[0].set_ylabel("Predicted Price")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()

    # ── Plot 2: Residuals ─────────────────────────────────────
    residuals = y_true_plot - y_pred_plot
    axes[1].scatter(y_pred_plot, residuals, alpha=0.5, s=25, color="#DD8452")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Predicted Price")
    axes[1].set_ylabel("Residual (Actual - Predicted)")
    axes[1].set_title("Residual Plot")

    plt.suptitle("Model Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_evaluation.png"), bbox_inches="tight")
    plt.show()

    print(f"\n📊 Residuals summary:")
    print(f"   Mean: {residuals.mean():,.0f}  (gần 0 là tốt)")
    print(f"   Std:  {residuals.std():,.0f}")


# ============================================================
# src/predict.py
# ============================================================
"""
Dùng để predict giá nhà cho 1 mẫu mới.

Usage:
    from predict import predict_price
    price = predict_price(area=5000, bedrooms=3, bathrooms=2,
                          stories=2, mainroad="yes", guestroom="no",
                          basement="no", hotwaterheating="no",
                          airconditioning="yes", parking=1,
                          prefarea="no", furnishingstatus="semi-furnished")
    print(f"Giá dự đoán: {price:,.0f} VND")
"""

import numpy as np
import pandas as pd
import joblib
import os

_MODEL_PATH  = os.path.join("models", "house_price_model.pkl")
_SCALER_PATH = os.path.join("models", "scaler.pkl")

_BINARY_COLS = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]
_FURNISH_MAP = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
_SCALE_COLS  = ["area"]
_LOG_TARGET  = True   # phải khớp với LOG_TARGET trong preprocess.py

# Feature order phải khớp lúc train
_FEATURE_ORDER = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus"
]


def predict_price(**kwargs) -> float:
    """
    Nhận các kwargs tương ứng với cột trong dataset,
    trả về giá nhà dự đoán (đơn vị gốc, không log).
    """
    model  = joblib.load(_MODEL_PATH)
    scaler = joblib.load(_SCALER_PATH)

    row = {}
    for col in _FEATURE_ORDER:
        val = kwargs.get(col)
        if val is None:
            raise ValueError(f"Thiếu feature: {col}")

        if col in _BINARY_COLS:
            row[col] = 1 if str(val).lower() == "yes" else 0
        elif col == "furnishingstatus":
            row[col] = _FURNISH_MAP.get(str(val).lower(), 0)
        else:
            row[col] = float(val)

    X = pd.DataFrame([row])[_FEATURE_ORDER]

    # Scale
    X[_SCALE_COLS] = scaler.transform(X[_SCALE_COLS])

    pred = model.predict(X)[0]

    # Inverse log nếu cần
    if _LOG_TARGET:
        pred = np.expm1(pred)

    return pred


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    price = predict_price(
        area=6000, bedrooms=3, bathrooms=2, stories=2,
        mainroad="yes", guestroom="no", basement="no",
        hotwaterheating="no", airconditioning="yes",
        parking=1, prefarea="no", furnishingstatus="semi-furnished"
    )
    print(f"\n🏠 Giá nhà dự đoán: {price:,.0f}")