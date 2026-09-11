"""
Week 3 — Model Training & Validation

Trains an XGBoost classifier for flood susceptibility prediction:
  1. Load the engineered dataset
  2. Spatial train/test split (avoids leakage from neighboring pixels)
  3. Handle class imbalance
  4. Train XGBoost
  5. Evaluate: AUC-ROC, precision/recall, confusion matrix
  6. Generate full-region risk probability map
  7. Save model + metrics
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    confusion_matrix, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURES, LABEL_COL, OUTPUT_DIR, PROCESSED_DIR,
    XGB_PARAMS, RANDOM_STATE, TEST_SIZE,
)


def load_dataset(csv_path: Path = None) -> pd.DataFrame:
    """Load the engineered flood dataset."""
    if csv_path is None:
        csv_path = OUTPUT_DIR / "flood_dataset.csv"

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Check required columns
    missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")

    return df


def spatial_split(df: pd.DataFrame, test_size: float = TEST_SIZE, block_size: int = 50):
    """
    Spatial block split: divide the dataset into spatial blocks and assign
    entire blocks to train/test. This prevents leakage from spatial
    autocorrelation (neighboring pixels are not independent samples).

    block_size: number of consecutive rows per spatial block.
    For a properly spatial split, you'd want to use coordinates, but
    since our data is sampled row-by-row from a raster grid, blocking
    consecutive rows (which correspond to adjacent raster rows) is a
    reasonable approximation for the MVP.
    """
    n_blocks = len(df) // block_size
    block_ids = np.repeat(np.arange(n_blocks), block_size)
    block_ids = np.concatenate([block_ids, np.full(len(df) - len(block_ids), n_blocks - 1)])

    # Shuffle block IDs
    unique_blocks = np.unique(block_ids)
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unique_blocks)

    n_test_blocks = int(len(unique_blocks) * test_size)
    test_blocks = set(unique_blocks[:n_test_blocks])

    test_mask = np.isin(block_ids, list(test_blocks))

    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    print(f"  Spatial split: {len(train_df)} train / {len(test_df)} test")
    print(f"  Train positive: {(train_df[LABEL_COL]==1).sum()} | "
          f"Test positive: {(test_df[LABEL_COL]==1).sum()}")

    return train_df, test_df


def prepare_features(train_df, test_df):
    """Scale features and prepare X/y arrays."""
    available_features = [f for f in FEATURES if f in train_df.columns]

    X_train = train_df[available_features].values
    y_train = train_df[LABEL_COL].values
    X_test = test_df[available_features].values
    y_test = test_df[LABEL_COL].values

    # Standard scaling (helps XGBoost slightly, essential for feature importance comparison)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, available_features, scaler


def train_model(X_train, y_train, available_features):
    """Train XGBoost with automatic class imbalance handling."""
    # Handle class imbalance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    imbalance_ratio = n_neg / max(n_pos, 1)

    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = imbalance_ratio
    params.pop("use_label_encoder", None)  # deprecated in newer xgboost

    print(f"  Class imbalance ratio: 1:{imbalance_ratio:.1f} (pos:neg)")
    print(f"  scale_pos_weight: {imbalance_ratio:.1f}")

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=True)

    return model


def evaluate_model(model, X_test, y_test, available_features):
    """Full evaluation: metrics + plots."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Core metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, target_names=["Not Flooded", "Flooded"])

    print(f"\n  AUC-ROC:   {auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"\n{report}")

    # Save metrics
    metrics = {
        "auc_roc": auc,
        "average_precision": ap,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": report,
        "n_features": len(available_features),
        "features": available_features,
    }

    metrics_path = OUTPUT_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0].plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # 2. Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1].plot(recall, precision, "r-", linewidth=2, label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    # 3. Feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    axes[2].barh(range(len(available_features)),
                 importance[sorted_idx], color="steelblue")
    axes[2].set_yticks(range(len(available_features)))
    axes[2].set_yticklabels([available_features[i] for i in sorted_idx])
    axes[2].set_xlabel("Feature Importance (gain)")
    axes[2].set_title("Feature Importance")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "evaluation_plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plots saved: {plot_path}")

    return model, metrics


def generate_risk_map(model, scaler, available_features):
    """
    Generate a full-region flood susceptibility probability map.
    Predicts over every pixel in the aligned raster stack.
    """
    import rasterio

    print("\n── Generating Risk Map ──")
    raster_paths = {
        "elevation": PROCESSED_DIR / "dem_aligned.tif",
        "slope": PROCESSED_DIR / "slope.tif",
        "twi": PROCESSED_DIR / "twi.tif",
        "distance_to_river": PROCESSED_DIR / "distance_to_river.tif",
        "land_cover": PROCESSED_DIR / "landcover_aligned.tif",
        "rainfall_max": PROCESSED_DIR / "rainfall_max.tif",
        "rainfall_cumulative": PROCESSED_DIR / "rainfall_cumulative.tif",
    }

    # Check all exist
    missing = [k for k, p in raster_paths.items() if not p.exists()]
    if missing:
        print(f"  ERROR: Missing raster layers: {missing}")
        print("  Run feature_engineering.py first.")
        return None

    # Read and stack
    arrays = {}
    ref_meta = None
    for name, path in raster_paths.items():
        with rasterio.open(path) as src:
            if ref_meta is None:
                ref_meta = src.profile.copy()
            arrays[name] = src.read(1).flatten()

    # Build feature matrix (only features the model uses)
    stack = np.column_stack([arrays[f] for f in available_features])

    # Handle nodata: mask pixels where any feature is nodata
    valid_mask = ~np.any(np.isnan(stack), axis=1)
    valid_stack = stack[valid_mask]

    # Scale
    valid_stack_scaled = scaler.transform(valid_stack)

    # Predict probabilities
    print(f"  Predicting on {len(valid_stack)} valid pixels...")
    risk_proba = np.full(len(stack), np.nan)
    risk_proba[valid_mask] = model.predict_proba(valid_stack_scaled)[:, 1]

    # Reshape to 2D
    with rasterio.open(PROCESSED_DIR / "dem_aligned.tif") as src:
        shape = src.shape
        transform = src.transform
        crs = src.crs

    risk_map = risk_proba.reshape(shape)

    # Save as GeoTIFF
    risk_path = OUTPUT_DIR / "flood_risk_map.tif"
    profile = ref_meta.copy()
    profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)

    with rasterio.open(risk_path, "w", **profile) as dst:
        dst.write(risk_map.astype(np.float32), 1)

    print(f"  Risk map saved: {risk_path}")
    print(f"  Risk range: {np.nanmin(risk_map):.3f} - {np.nanmax(risk_map):.3f}")
    print(f"  Mean risk: {np.nanmean(risk_map):.3f}")

    # Quick visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    im = ax.imshow(risk_map, cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Flood Susceptibility Probability")
    ax.set_title("Flood Susceptibility Map — Sindh 2022")
    ax.axis("off")
    viz_path = OUTPUT_DIR / "risk_map_preview.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Preview saved: {viz_path}")

    return risk_path


def run_training():
    """Execute the full model training pipeline."""
    print("=" * 70)
    print("MODEL TRAINING — Week 3")
    print("=" * 70)

    # Load data
    df = load_dataset()

    # Spatial split
    print("\n── Spatial Train/Test Split ──")
    train_df, test_df = spatial_split(df)

    # Prepare features
    print("\n── Feature Preparation ──")
    X_train, y_train, X_test, y_test, avail_features, scaler = prepare_features(train_df, test_df)
    print(f"  Features used: {avail_features}")

    # Train
    print("\n── Training XGBoost ──")
    model = train_model(X_train, y_train, avail_features)

    # Evaluate
    print("\n── Evaluation ──")
    model, metrics = evaluate_model(model, X_test, y_test, avail_features)

    # Save model
    model_path = OUTPUT_DIR / "flood_model.json"
    model.save_model(str(model_path))
    print(f"\n  Model saved: {model_path}")

    # Generate risk map
    risk_path = generate_risk_map(model, scaler, avail_features)

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print(f"Model: {model_path}")
    print(f"Risk map: {risk_path}")
    print(f"Metrics: {OUTPUT_DIR / 'model_metrics.json'}")
    print("Next step: Run streamlit run app.py for dashboard (Week 4)")
    print("=" * 70)

    return model, metrics


if __name__ == "__main__":
    run_training()
