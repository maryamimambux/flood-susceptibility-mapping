"""
Prepare lightweight assets for deployment.
Downsamples the full risk map (~957MB) into a small JSON + PNG
that can be served by a Flask app on Vercel (<5MB total).

Run:  python prepare_deploy.py
"""
import json
import base64
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("data/output")
DEPLOY_DIR = Path("deploy/static")
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Downsample risk map ───────────────────────────────────────────
print("Downsampling risk map...")
import rasterio
from rasterio.transform import array_bounds

risk_path = OUTPUT_DIR / "flood_risk_map.tif"
with rasterio.open(risk_path) as src:
    full_data = src.read(1)
    bounds = src.bounds
    crs = src.crs.to_string()

# Downsample by factor of 20 (6170x4556 → ~309x228)
factor = 20
small = full_data[::factor, ::factor].copy()
# Replace nodata (NaN) with -1 for JSON
small = np.where(np.isnan(small), -1, small)
small = np.round(small, 4)  # 4 decimal places is plenty

print(f"  Original: {full_data.shape} -> Downsampled: {small.shape}")

# Save as JSON
risk_json = {
    "data": small.tolist(),
    "bounds": {
        "south": bounds.bottom,
        "north": bounds.top,
        "west": bounds.left,
        "east": bounds.right,
    },
    "shape": list(small.shape),
}
json_path = DEPLOY_DIR / "risk_data.json"
with open(json_path, "w") as f:
    json.dump(risk_json, f)
print(f"  Saved: {json_path} ({json_path.stat().st_size / 1024:.0f} KB)")

# ── 2. Copy model metrics ────────────────────────────────────────────
import shutil
metrics_src = OUTPUT_DIR / "model_metrics.json"
metrics_dst = DEPLOY_DIR / "model_metrics.json"
shutil.copy2(metrics_src, metrics_dst)
print(f"  Copied: {metrics_dst}")

# ── 3. Copy evaluation plots ─────────────────────────────────────────
plots_src = OUTPUT_DIR / "evaluation_plots.png"
plots_dst = DEPLOY_DIR / "evaluation_plots.png"
shutil.copy2(plots_src, plots_dst)
print(f"  Copied: {plots_dst}")

# ── 4. Copy risk map preview ─────────────────────────────────────────
preview_src = OUTPUT_DIR / "risk_map_preview.png"
preview_dst = DEPLOY_DIR / "risk_map_preview.png"
if preview_src.exists():
    shutil.copy2(preview_src, preview_dst)
    print(f"  Copied: {preview_dst}")

# ── Summary ──────────────────────────────────────────────────────────
total_size = sum(f.stat().st_size for f in DEPLOY_DIR.rglob("*") if f.is_file())
print(f"\nDeploy assets: {total_size / 1024 / 1024:.1f} MB total")
print("Done! Deploy folder is ready.")
