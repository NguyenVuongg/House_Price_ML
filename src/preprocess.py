# ============================================================
# src/preprocess.py
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # chia dữ liệu thành train/test
from sklearn.preprocessing import StandardScaler  # kéo về cùng scale để model dễ học hơn
import joblib   # lưu scaler đã fit để dùng lại lúc predict
import os   # tạo thư mục nếu chưa có

# ── Paths ────────────────────────────────────────────────────
RAW_PATH       = "data/raw/Housing.csv"
PROCESSED_DIR  = "data/processed"
MODEL_DIR      = "models"
SCALER_PATH    = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Constants ────────────────────────────────────────────────
BINARY_COLS = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]
FURNISH_MAP = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
SCALE_COLS  = ["area"]          # chỉ scale các cột phân phối rộng
TARGET      = "price"
LOG_TARGET  = True              # log1p(price) → set False nếu không muốn


# ============================================================
# 1. Load
# ============================================================
def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] {df.shape[0]} mẫu, {df.shape[1]} cột")
    return df


# ============================================================
# 2. Encode categorical
# ============================================================
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 2a. Binary: yes → 1, no → 0
    for col in BINARY_COLS:
        df[col] = df[col].map({"yes": 1, "no": 0})
        print(f"[encode] {col}: yes/no → 1/0")

    # 2b. furnishingstatus: ordinal (furnished=2 > semi=1 > unfurnished=0)
    df["furnishingstatus"] = df["furnishingstatus"].map(FURNISH_MAP)
    print("[encode] furnishingstatus: ordinal 0/1/2")

    return df


# ============================================================
# 3. Xử lý outliers (IQR capping — giữ lại nhưng cap tại fence)
# ============================================================
def cap_outliers(df: pd.DataFrame,
                 cols: list = ["price", "area"]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df[(df[col] < lower) | (df[col] > upper)])
        df[col] = df[col].clip(lower, upper)
        print(f"[outlier] {col}: cap {before} điểm vào [{lower:,.0f}, {upper:,.0f}]")
    return df


# ============================================================
# 4. Log-transform target
# ============================================================
def transform_target(df: pd.DataFrame) -> pd.DataFrame:
    if LOG_TARGET:
        df = df.copy()
        df[TARGET] = np.log1p(df[TARGET])
        print(f"[transform] price → log1p(price)  skew={df[TARGET].skew():.3f}")
    return df


# ============================================================
# 5. Scale
# ============================================================
def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   fit: bool = True):
    """
    fit=True  → fit scaler trên X_train rồi transform cả hai (training pipeline)
    fit=False → chỉ load scaler đã lưu và transform (inference pipeline)
    """
    if fit:
        scaler = StandardScaler()
        X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
        X_test[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])
        joblib.dump(scaler, SCALER_PATH)
        print(f"[scale] StandardScaler fit & saved → {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        X_train[SCALE_COLS] = scaler.transform(X_train[SCALE_COLS])
        print(f"[scale] Scaler loaded from {SCALER_PATH}")
    return X_train, X_test


# ============================================================
# 6. Split
# ============================================================
def split_data(df: pd.DataFrame,
               test_size: float = 0.2,
               random_state: int = 42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[split] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ============================================================
# 7. Save processed data
# ============================================================
def save_processed(X_train, X_test, y_train, y_test):
    train = X_train.copy(); train[TARGET] = y_train.values
    test  = X_test.copy();  test[TARGET]  = y_test.values
    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR,  "test.csv"),  index=False)
    print(f"[save] → {PROCESSED_DIR}/train.csv  &  test.csv")


# ============================================================
# 8. Pipeline chính
# ============================================================
def run_preprocessing() -> tuple:
    print("\n" + "="*50)
    print("  PREPROCESSING PIPELINE")
    print("="*50)

    df = load_data()
    df = encode_features(df)
    df = cap_outliers(df)
    df = transform_target(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test = scale_features(X_train.copy(), X_test.copy(), fit=True)

    save_processed(X_train, X_test, y_train, y_test)

    print("\n✅ Preprocessing hoàn tất!")
    print(f"   Features ({X_train.shape[1]}): {list(X_train.columns)}")
    print("="*50 + "\n")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()