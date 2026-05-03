"""
automate_Dhea-Yuza-Fadiya.py
Pipeline preprocessing otomatis untuk NASA FIRMS Wildfire dataset.

Usage:
    python automate_Dhea-Yuza-Fadiya.py
    python automate_Dhea-Yuza-Fadiya.py --input_dir wildfire_data_raw --output_dir wildfire_preprocessing
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High", 3: "Extreme"}


def load_raw_data(input_dir: str) -> pd.DataFrame:
    """Load semua file CSV dari direktori raw data."""
    csv_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        raise FileNotFoundError(f"Tidak ada file CSV di: {input_dir}")

    print(f"Ditemukan {len(csv_files)} file CSV")
    all_dfs = []
    for fpath in csv_files:
        df = pd.read_csv(fpath, low_memory=False)
        df["source_file"] = os.path.basename(fpath)
        all_dfs.append(df)
        print(f"  ✔ {os.path.basename(fpath)}: {df.shape}")

    df_raw = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined shape: {df_raw.shape}")
    return df_raw


def clean_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Konversi confidence kategorikal ke numerik."""
    conf_map = {"low": 33, "nominal": 66, "high": 99}
    df = df.copy()
    df["confidence"] = df["confidence"].apply(
        lambda x: conf_map.get(str(x).strip().lower(), x)
    )
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


def create_intensity_label(df: pd.DataFrame) -> pd.DataFrame:
    """Buat label intensity_class dari kolom frp."""
    def categorize(frp):
        if pd.isna(frp):  return np.nan
        if frp < 10:      return 0   # Low
        if frp < 50:      return 1   # Moderate
        if frp < 200:     return 2   # High
        return 3                     # Extreme

    df = df.copy()
    df["intensity_class"] = df["frp"].apply(categorize)
    return df


def encode_features(df: pd.DataFrame):
    """Encode fitur kategorikal dan tentukan feature columns."""
    df = df.copy()

    if "daynight" in df.columns:
        df["daynight_enc"] = df["daynight"].map({"D": 1, "N": 0})

    instrument_cols = []
    if "instrument" in df.columns:
        dummies = pd.get_dummies(df["instrument"], prefix="instrument", dtype=int)
        df = pd.concat([df, dummies], axis=1)
        instrument_cols = list(dummies.columns)

    base_features = ["brightness", "scan", "track", "bright_t31", "confidence", "daynight_enc"]
    feature_cols = [c for c in base_features if c in df.columns] + instrument_cols

    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    return df, feature_cols


def handle_missing(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Drop baris tanpa target dan imputasi median untuk fitur."""
    df = df.copy()
    df = df.dropna(subset=["intensity_class"])
    df["intensity_class"] = df["intensity_class"].astype(int)

    for feat in feature_cols:
        n_miss = df[feat].isnull().sum()
        if n_miss > 0:
            median_val = df[feat].median()
            df[feat] = df[feat].fillna(median_val)
            print(f"  {feat}: {n_miss:,} missing → diisi median ({median_val:.4f})")

    print(f"Shape setelah handle missing: {df.shape}")
    return df


def cap_outliers(df: pd.DataFrame, feature_cols: list):
    """Outlier capping menggunakan IQR × 3.0."""
    df = df.copy()
    bounds = {}
    # Hanya fitur numerik kontinyu (skip binary encoded)
    cap_cols = [c for c in feature_cols
                if c != "daynight_enc" and not c.startswith("instrument_")]

    for col in cap_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 3.0 * IQR, Q3 + 3.0 * IQR
        n_cap = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        bounds[col] = (lower, upper)
        print(f"  {col:15s}: [{lower:.2f}, {upper:.2f}] → {n_cap:,} di-cap")

    return df, bounds


def split_and_scale(df: pd.DataFrame, feature_cols: list):
    """Train/Val/Test split 64/16/20 + StandardScaler."""
    X = df[feature_cols].values
    y = df["intensity_class"].values

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.20, random_state=SEED, stratify=y_tv)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    total = len(X)
    print(f"Train: {X_train.shape} ({len(X_train)/total*100:.1f}%)")
    print(f"Val  : {X_val.shape} ({len(X_val)/total*100:.1f}%)")
    print(f"Test : {X_test.shape} ({len(X_test)/total*100:.1f}%)")

    return (X_train_sc, y_train), (X_val_sc, y_val), (X_test_sc, y_test), scaler


def save_outputs(train, val, test, scaler, bounds, feature_cols, output_dir: str):
    """Simpan CSV splits dan artefak preprocessing."""
    os.makedirs(output_dir, exist_ok=True)
    TARGET_COL = "intensity_class"

    for split_name, (X_sc, y_split) in [("train", train), ("val", val), ("test", test)]:
        out_df = pd.DataFrame(X_sc, columns=feature_cols)
        out_df[TARGET_COL] = y_split
        out_path = os.path.join(output_dir, f"{split_name}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  ✔ {split_name}.csv → {out_df.shape}")

    joblib.dump(scaler,       os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(output_dir, "feature_cols.pkl"))
    joblib.dump(bounds,       os.path.join(output_dir, "outlier_bounds.pkl"))
    print("  ✔ scaler.pkl, feature_cols.pkl, outlier_bounds.pkl")


def main():
    parser = argparse.ArgumentParser(description="NASA FIRMS Wildfire Preprocessing Pipeline")
    parser.add_argument("--input_dir",  type=str, default="wildfire_data_raw",
                        help="Path folder raw dataset")
    parser.add_argument("--output_dir", type=str, default="wildfire_preprocessing",
                        help="Path folder output preprocessed")
    args = parser.parse_args()

    print("=" * 55)
    print("WILDFIRE PREPROCESSING PIPELINE")
    print("=" * 55)

    print("\n[1/7] Load raw data...")
    df = load_raw_data(args.input_dir)

    print("\n[2/7] Clean confidence...")
    df = clean_confidence(df)

    print("\n[3/7] Create intensity labels...")
    df = create_intensity_label(df)
    class_dist = df["intensity_class"].value_counts().sort_index()
    for cls, cnt in class_dist.items():
        print(f"  {int(cls)} ({LABEL_MAP[int(cls)]:10s}): {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

    print("\n[4/7] Encode features...")
    df, feature_cols = encode_features(df)

    print("\n[5/7] Handle missing values...")
    df = handle_missing(df, feature_cols)

    print("\n[6/7] Cap outliers (IQR × 3.0)...")
    df, bounds = cap_outliers(df, feature_cols)

    print("\n[7/7] Split, scale & save...")
    train, val, test, scaler = split_and_scale(df, feature_cols)
    save_outputs(train, val, test, scaler, bounds, feature_cols, args.output_dir)

    print(f"\n✔ Selesai! Output tersimpan di: {args.output_dir}")


if __name__ == "__main__":
    main()
