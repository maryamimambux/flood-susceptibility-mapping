"""
Flood Susceptibility Mapping — Sindh 2022 Floods
Configuration constants and paths.
"""
from pathlib import Path

# ── Study Area: Sindh province, Pakistan ──────────────────────────────
# Bounding box (lon/lat, WGS84) — covers the core flood-affected zone
# along the Indus River basin in Sindh.
STUDY_AREA_BBOX = {
    "lon_min": 67.0,
    "lon_max": 71.0,
    "lat_min": 23.5,
    "lat_max": 28.5,
}

# Target CRS for all analysis (UTM Zone 42N — covers Sindh, preserves distance)
TARGET_CRS = "EPSG:32642"
# Alternatively, keep geographic (WGS84) if you prefer working in degrees:
# TARGET_CRS = "EPSG:4326"

# Analysis resolution (meters) — 90m is a fast MVP default; SRTM native is ~30m
ANALYSIS_RESOLUTION_M = 90

# ── Data Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Source URLs ──────────────────────────────────────────────────
# Copernicus DEM 30m — hosted on AWS S3, NO API key or AWS account needed.
# Bucket: copernicus-dem-30m (eu-central-1 region)
COPERNICUS_DEM_S3_BUCKET = "copernicus-dem-30m"
COPERNICUS_DEM_HTTPS = "https://copernicus-dem-30m.s3.amazonaws.com"

# Copernicus EMS flood extent (manual download — instructions in download script)
EMS_URL_HINT = "https://mapping.emergency.copernicus.eu/activations/"

# CHIRPS rainfall data (pentad, 0.05° resolution, global)
# Actual file format: chirps-v2.0.{YEAR}.{MONTH}.{PENTAD_NUM}.tif.gz
# Pentad numbers: 1-6
CHIRPS_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_pentad/tifs/"

# ESA WorldCover land cover (2021, 10m resolution)
WORLD_COVER_URL = "https://esa-worldcover.org/en/data-access"

# HydroRIVERS (HydroSHEDS)
HYDROSHEDS_URL = "https://hydrosheds.org/downloads"

# ── Flood Susceptibility Factors ─────────────────────────────────────
# These are the features we engineer from raw data
FEATURES = [
    "elevation",           # from DEM
    "slope",               # derived from DEM
    "twi",                 # Topographic Wetness Index — derived from DEM
    "distance_to_river",   # from HydroRIVERS network
    "land_cover",          # from ESA WorldCover
    "rainfall_max",        # peak rainfall during flood period (Aug 2022)
    "rainfall_cumulative", # cumulative rainfall Aug 2022
]

LABEL_COL = "flooded"  # binary: 1 = flooded (from EMS ground truth)

# ── Model Hyperparameters (defaults) ─────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.25

# XGBoost defaults — good starting point for imbalanced binary classification
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": None,  # set dynamically based on class imbalance ratio
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
}
