"""
Week 2 — Feature Engineering & Dataset Construction

Transforms raw geospatial data into an ML-ready tabular dataset:
  1. Reproject & align all rasters to a common grid (DEM reference)
  2. Derive terrain features (slope, TWI) from DEM using Whitebox
  3. Compute distance-to-river from HydroRIVERS network
  4. Aggregate rainfall pentads into max + cumulative layers
  5. Rasterize EMS flood extent into binary labels
  6. Stack all aligned layers into a DataFrame
  7. Sample training data
"""
import numpy as np
import rasterio
from rasterio.warp import Resampling
from pathlib import Path
import geopandas as gpd
import whitebox
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DIR, PROCESSED_DIR, OUTPUT_DIR,
    TARGET_CRS, ANALYSIS_RESOLUTION_M, FEATURES, LABEL_COL,
)
from src.utils import (
    align_raster, get_reference_grid, rasterize_vector,
    sample_raster_stack, get_raster_info,
)


def setup_whitebox():
    """Initialize WhiteboxTools for terrain analysis."""
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    # Set working directory
    wbt.set_working_dir(str(PROCESSED_DIR))
    return wbt


def step1_align_dem():
    """Reproject DEM to target CRS and resolution. This becomes our reference grid."""
    print("\n── Step 1: Align DEM (reference grid) ──")
    dem_raw = RAW_DIR / "srtm_dem.tif"
    dem_aligned = PROCESSED_DIR / "dem_aligned.tif"

    if not dem_raw.exists():
        print("  ERROR: DEM not found. Run download_data.py first.")
        return None

    align_raster(
        dem_raw,
        dem_aligned,
        target_crs=TARGET_CRS,
        target_resolution=ANALYSIS_RESOLUTION_M,
        resampling=Resampling.bilinear,
    )

    ref_transform, ref_shape = get_reference_grid(dem_aligned)
    print(f"  Reference grid: {ref_shape[0]} x {ref_shape[1]} pixels")
    return dem_aligned


def step2_derive_terrain(dem_path: Path):
    """
    Compute slope and TWI (Topographic Wetness Index) from the aligned DEM.
    TWI = ln(As / tan(slope))  where As = upslope contributing area.
    High TWI = water accumulates here = more flood-prone.
    """
    print("\n── Step 2: Derive terrain features (slope, TWI) ──")
    wbt = setup_whitebox()

    slope_path = PROCESSED_DIR / "slope.tif"
    twi_path = PROCESSED_DIR / "twi.tif"
    fill_path = PROCESSED_DIR / "dem_filled.tif"  # pit-filled DEM (required for flow)

    if slope_path.exists() and twi_path.exists():
        print("  Slope and TWI already exist, skipping.")
        return slope_path, twi_path

    dem_name = dem_path.name
    dem_stem = dem_path.stem

    # Fill depressions (required for hydrologically correct flow analysis)
    print("  Filling DEM depressions...")
    wbt.fill_depressions(dem_name, f"{dem_stem}_filled.tif")

    # Slope (degrees — for the ML feature)
    print("  Computing slope...")
    wbt.slope(f"{dem_stem}_filled.tif", "slope.tif", units="degrees")

    # TWI = ln(As / tan(slope))
    # WhiteboxTools wetness_index needs: specific contributing area + slope in radians
    print("  Computing flow accumulation (for TWI)...")
    wbt.d8_flow_accumulation(f"{dem_stem}_filled.tif", "sca.tif", out_type="Specific Catchment Area")

    print("  Computing slope in radians (for TWI)...")
    wbt.slope(f"{dem_stem}_filled.tif", "slope_rad.tif", units="radians")

    print("  Computing Topographic Wetness Index...")
    wbt.wetness_index("sca.tif", "slope_rad.tif", "twi.tif")

    print(f"  Slope saved: {slope_path}")
    print(f"  TWI saved: {twi_path}")
    return slope_path, twi_path


def step3_distance_to_river(ref_transform, ref_shape: tuple):
    """
    Compute Euclidean distance from each pixel to the nearest river/stream.
    Uses HydroRIVERS shapefile rasterized onto the reference grid, then
    computes distance transform.
    """
    print("\n── Step 3: Distance to river network ──")
    from scipy.ndimage import distance_transform_edt

    river_shp = RAW_DIR / "hydrorivers"
    dist_path = PROCESSED_DIR / "distance_to_river.tif"

    if dist_path.exists():
        print("  Distance-to-river already exists, skipping.")
        return dist_path

    # Find the shapefile (check subdirectories too)
    shp_files = []
    if river_shp.exists():
        shp_files = list(river_shp.rglob("*.shp"))
        print(f"  Scanned: {river_shp}")
        print(f"  Found {len(shp_files)} shapefile(s): {[f.name for f in shp_files[:5]]}")
        if not shp_files:
            all_files = list(river_shp.rglob("*"))
            exts = set(f.suffix.lower() for f in all_files if f.is_file())
            print(f"  Directory contains {len(all_files)} items. Extensions: {exts}")
    if not shp_files:
        print("  WARNING: HydroRIVERS shapefile not found.")
        print(f"  Place .shp file in: {river_shp}")
        print("  Creating placeholder zero-distance raster.")
        # Create a zero-filled raster as placeholder
        _create_placeholder_raster(dist_path, ref_transform, ref_shape, fill_val=0)
        return dist_path

    wbt = setup_whitebox()

    # Rasterize rivers onto reference grid
    river_raster = PROCESSED_DIR / "rivers_rasterized.tif"
    rasterize_vector(
        shp_files[0], river_raster,
        ref_transform, ref_shape, reference_crs=TARGET_CRS, burn_value=1,
    )

    # Euclidean distance from each non-river pixel to nearest river pixel
    print("  Computing Euclidean distance transform...")
    with rasterio.open(river_raster) as src:
        river_arr = src.read(1)

    # distance_transform_edt returns distance in pixels; multiply by resolution
    dist_pixels = distance_transform_edt(river_arr == 0)
    dist_meters = dist_pixels * ANALYSIS_RESOLUTION_M

    _write_array_as_raster(dist_meters, dist_path, ref_transform, ref_shape,
                           dtype="float32", nodata=-9999)
    print(f"  Distance-to-river saved: {dist_path}")
    print(f"  Max distance: {dist_meters.max():.0f} m")
    return dist_path


def step4_aggregate_rainfall(ref_transform, ref_shape: tuple):
    """
    Aggregate CHIRPS pentad rainfall data into:
      - rainfall_max: peak pentad rainfall during Aug 2022
      - rainfall_cumulative: total Aug 2022 rainfall
    Both reprojected to the reference grid.
    """
    print("\n── Step 4: Aggregate rainfall layers ──")
    import xarray as xr

    chirps_dir = RAW_DIR / "chirps_aug2022"
    max_path = PROCESSED_DIR / "rainfall_max.tif"
    cum_path = PROCESSED_DIR / "rainfall_cumulative.tif"

    if max_path.exists() and cum_path.exists():
        print("  Rainfall layers already exist, skipping.")
        return max_path, cum_path

    tif_files = sorted(chirps_dir.glob("*.tif")) if chirps_dir.exists() else []
    if not tif_files:
        print("  WARNING: No CHIRPS rainfall files found.")
        print(f"  Run download_data.py or place files in: {chirps_dir}")
        _create_placeholder_raster(max_path, ref_transform, ref_shape, fill_val=0)
        _create_placeholder_raster(cum_path, ref_transform, ref_shape, fill_val=0)
        return max_path, cum_path

    # Read and stack all pentads
    pentad_arrays = []
    for f in tif_files:
        aligned_p = PROCESSED_DIR / f"f_{f.name}"
        align_raster(
            f, aligned_p,
            target_crs=TARGET_CRS,
            reference_transform=ref_transform,
            reference_shape=ref_shape,
            resampling=Resampling.bilinear,
        )
        with rasterio.open(aligned_p) as src:
            arr = src.read(1).astype(np.float32)
            arr[arr < 0] = np.nan  # CHIRPS uses negative for nodata
            pentad_arrays.append(arr)

    stack = np.stack(pentad_arrays, axis=0)  # shape: (pentads, rows, cols)

    rainfall_max = np.nanmax(stack, axis=0)
    rainfall_cum = np.nansum(stack, axis=0)

    _write_array_as_raster(rainfall_max, max_path, ref_transform, ref_shape,
                           dtype="float32", nodata=-9999)
    _write_array_as_raster(rainfall_cum, cum_path, ref_transform, ref_shape,
                           dtype="float32", nodata=-9999)

    print(f"  Rainfall max: {rainfall_max[~np.isnan(rainfall_max)].max():.1f} mm/pentad")
    print(f"  Rainfall cumulative: {rainfall_cum[~np.isnan(rainfall_cum)].max():.1f} mm")
    return max_path, cum_path


def step5_rasterize_flood_extent(ref_transform, ref_shape: tuple):
    """
    Convert Copernicus EMS flood extent shapefiles into a binary label raster.
    Merges all shapefiles found in the ems_flood_extent directory (supports
    multiple AOIs from different activations like EMSR629 + EMSR631).
    """
    print("\n── Step 5: Rasterize flood extent (labels) ──")
    ems_dir = RAW_DIR / "ems_flood_extent"
    label_path = PROCESSED_DIR / "flood_label.tif"

    if label_path.exists():
        print("  Flood label already exists, skipping.")
        return label_path

    # Find ALL vector files (shapefiles, geojsons, geopackages)
    all_vector_files = []
    if ems_dir.exists():
        for ext in ["*.shp", "*.geojson", "*.gpkg"]:
            all_vector_files.extend(ems_dir.glob(ext))
            all_vector_files.extend(ems_dir.rglob(ext))
        # Deduplicate
        all_vector_files = list(set(all_vector_files))
        # Diagnostic: show what we found
        print(f"  Scanned: {ems_dir}")
        print(f"  Found {len(all_vector_files)} total vector file(s)")
        if not all_vector_files:
            # Show what IS in the directory
            all_files = list(ems_dir.rglob("*"))
            print(f"  Directory contains {len(all_files)} items. Extensions found:")
            exts = set(f.suffix.lower() for f in all_files if f.is_file())
            print(f"    {exts}")

    # Filter to only flood extent files (observedEventA = observed flood polygons)
    # EMS products include roads, buildings, hydrography etc. — we only want flood extent
    shp_files = [f for f in all_vector_files if "observedEvent" in f.stem]
    if not shp_files and all_vector_files:
        print(f"  WARNING: No 'observedEvent' files found among {len(all_vector_files)} vectors.")
        print(f"  File names: {sorted([f.name for f in all_vector_files[:5]])}")
        print(f"  Falling back to ALL files (may include non-flood features).")
        shp_files = all_vector_files
    elif shp_files:
        print(f"  Filtered to {len(shp_files)} flood extent file(s): {[f.name for f in shp_files]}")
    
    if not shp_files:
        print("  WARNING: EMS flood extent file not found.")
        print(f"  Place .shp or .geojson in: {ems_dir}")
        _create_placeholder_raster(label_path, ref_transform, ref_shape,
                                   fill_val=0, dtype="uint8")
        return label_path

    # Load and merge all shapefiles into one GeoDataFrame
    import geopandas as gpd
    gdfs = []
    for f in shp_files:
        try:
            gdf = gpd.read_file(f)
            if not gdf.empty:
                gdfs.append(gdf)
                print(f"  Loaded {len(gdf)} features from: {f.name}")
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")

    if not gdfs:
        print("  WARNING: No valid features found in EMS files.")
        _create_placeholder_raster(label_path, ref_transform, ref_shape,
                                   fill_val=0, dtype="uint8")
        return label_path

    # Merge all into one GeoDataFrame
    merged = gpd.pd.concat(gdfs, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=gdfs[0].crs)
    print(f"  Merged {len(merged)} total flood features from {len(gdfs)} files")

    rasterize_vector(
        merged, label_path,
        ref_transform, ref_shape, reference_crs=TARGET_CRS, burn_value=1,
    )
    return label_path


def step6_build_dataset(raster_paths: dict, output_csv: Path = None, max_samples: int = 500_000):
    """
    Stack all aligned rasters and sample into an ML-ready DataFrame.
    Smart sampling: keeps ALL flooded pixels (rare & valuable) and samples
    non-flooded pixels to maintain a reasonable class ratio (up to 5:1).
    """
    print("\n── Step 6: Build ML dataset ──")
    if output_csv is None:
        output_csv = OUTPUT_DIR / "flood_dataset.csv"

    if output_csv.exists():
        print(f"  Dataset already exists: {output_csv}")
        return pd.read_csv(output_csv)

    df = sample_raster_stack(raster_paths, n_samples=None)

    # Smart sampling to keep dataset manageable
    if LABEL_COL in df.columns and len(df) > max_samples:
        n_pos = (df[LABEL_COL] == 1).sum()
        n_neg = (df[LABEL_COL] == 0).sum()
        print(f"  Raw: {n_pos} flooded / {n_neg} not-flooded")

        if n_pos > 0:
            # Keep all flooded, sample non-flooded at 5:1 ratio (or fill remaining budget)
            target_neg = min(n_neg, max(n_pos * 5, max_samples - n_pos))
            df_neg = df[df[LABEL_COL] == 0].sample(n=int(target_neg), random_state=42)
            df_pos = df[df[LABEL_COL] == 1]
            df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"  Sampled: {len(df_pos)} flooded / {len(df_neg)} not-flooded")
        else:
            # No flood pixels — just random sample
            df = df.sample(n=max_samples, random_state=42)
            print(f"  Random sample: {len(df)} rows (no flood pixels found)")

    # Report final class balance
    if LABEL_COL in df.columns:
        n_pos = (df[LABEL_COL] == 1).sum()
        n_neg = (df[LABEL_COL] == 0).sum()
        ratio = n_neg / max(n_pos, 1)
        print(f"  Final: {n_pos} flooded / {n_neg} not-flooded (ratio 1:{ratio:.0f})")

    df.to_csv(output_csv, index=False)
    print(f"  Dataset saved: {output_csv} ({len(df)} rows, {len(df.columns)} columns)")
    return df


# ── Helper functions ──────────────────────────────────────────────────

def _write_array_as_raster(array, path, transform, shape, dtype="float32", nodata=-9999):
    """Write a 2D numpy array as a GeoTIFF using reference grid parameters."""
    from rasterio.transform import from_bounds
    from pyproj import CRS

    # Use appropriate nodata for the dtype
    if dtype == "uint8":
        nodata = 255
    elif dtype == "int16":
        nodata = -9999

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "width": shape[1],
        "height": shape[0],
        "transform": transform,
        "crs": CRS.from_user_input(TARGET_CRS),
        "compress": "lzw",
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def _create_placeholder_raster(path, transform, shape, fill_val=0, dtype="float32"):
    """Create a placeholder raster filled with a constant value."""
    arr = np.full(shape, fill_val, dtype=dtype)
    _write_array_as_raster(arr, path, transform, shape, dtype=dtype)
    print(f"  Created placeholder: {path}")


# ── Main Pipeline ─────────────────────────────────────────────────────

def run_feature_engineering():
    """Execute the full feature engineering pipeline."""
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE — Week 2")
    print("=" * 70)

    # Step 1: Align DEM (establishes reference grid)
    dem_path = step1_align_dem()
    if dem_path is None:
        return

    ref_transform, ref_shape = get_reference_grid(dem_path)

    # Step 2: Derive terrain
    slope_path, twi_path = step2_derive_terrain(dem_path)

    # Step 3: Distance to river
    dist_river_path = step3_distance_to_river(ref_transform, ref_shape)

    # Step 4: Rainfall aggregation
    rainfall_max_path, rainfall_cum_path = step4_aggregate_rainfall(ref_transform, ref_shape)

    # Step 5: Flood extent labels
    label_path = step5_rasterize_flood_extent(ref_transform, ref_shape)

    # Step 6: Align land cover (if available)
    landcover_path = PROCESSED_DIR / "landcover_aligned.tif"
    lc_raw = list((RAW_DIR / "worldcover").glob("*.tif")) if (RAW_DIR / "worldcover").exists() else []
    if lc_raw and not landcover_path.exists():
        print("\n── Aligning land cover ──")
        align_raster(
            lc_raw[0], landcover_path,
            target_crs=TARGET_CRS,
            reference_transform=ref_transform,
            reference_shape=ref_shape,
            resampling=Resampling.nearest,  # categorical data → nearest neighbor
        )
    elif not landcover_path.exists():
        print("\n  WARNING: No WorldCover data found. Using placeholder.")
        _create_placeholder_raster(landcover_path, ref_transform, ref_shape, fill_val=0)

    # Step 7: Build dataset
    raster_map = {
        "elevation": str(dem_path),
        "slope": str(slope_path),
        "twi": str(twi_path),
        "distance_to_river": str(dist_river_path),
        "land_cover": str(landcover_path),
        "rainfall_max": str(rainfall_max_path),
        "rainfall_cumulative": str(rainfall_cum_path),
        LABEL_COL: str(label_path),
    }

    df = step6_build_dataset(raster_map)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print(f"Dataset: {OUTPUT_DIR / 'flood_dataset.csv'}")
    print("Next step: Run src/model.py for model training (Week 3)")
    print("=" * 70)
    return df


if __name__ == "__main__":
    run_feature_engineering()
