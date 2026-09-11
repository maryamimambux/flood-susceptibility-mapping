"""
Shared utility functions for geospatial operations.
"""
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from pyproj import CRS
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TARGET_CRS, ANALYSIS_RESOLUTION_M


def get_raster_info(path: Path) -> dict:
    """Print and return key metadata about a raster file."""
    with rasterio.open(path) as src:
        info = {
            "crs": src.crs,
            "bounds": src.bounds,
            "resolution": src.res,
            "shape": src.shape,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "band_count": src.count,
        }
    print(f"  CRS:        {info['crs']}")
    print(f"  Bounds:     {info['bounds']}")
    print(f"  Resolution: {info['resolution']}")
    print(f"  Shape:      {info['shape']}")
    print(f"  Dtype:      {info['dtype']}")
    print(f"  NoData:     {info['nodata']}")
    return info


def align_raster(
    source_path: Path,
    output_path: Path,
    target_crs: str = TARGET_CRS,
    target_resolution: float = ANALYSIS_RESOLUTION_M,
    reference_transform=None,
    reference_shape: tuple = None,
    resampling: Resampling = Resampling.bilinear,
):
    """
    Reproject and resample a raster to match the target CRS, resolution,
    and optionally a reference grid (for pixel-perfect alignment).

    This is the core function that prevents the "layers silently misalign"
    problem — every raster in the analysis must pass through this.
    """
    with rasterio.open(source_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_width = src.width
        src_height = src.height

        if reference_transform is not None and reference_shape is not None:
            # Match exact grid of reference raster
            dst_transform = reference_transform
            dst_height, dst_width = reference_shape
        else:
            # Compute transform for target CRS + resolution
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs,
                target_crs,
                src_width,
                src_height,
                *src.bounds,
                resolution=target_resolution,
            )

        dst_profile = {
            "driver": "GTiff",
            "dtype": src.dtypes[0],
            "count": src.count,
            "width": dst_width,
            "height": dst_height,
            "transform": dst_transform,
            "crs": CRS.from_user_input(target_crs),
            "compress": "lzw",
            "nodata": src.nodata,
        }

        with rasterio.open(output_path, "w", **dst_profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )

    print(f"  Aligned raster saved: {output_path}")
    print(f"  Shape: {dst_height} x {dst_width}, CRS: {target_crs}")
    return output_path


def get_reference_grid(
    dem_path: Path,
) -> tuple:
    """
    Get the reference transform and shape from the DEM raster.
    All other layers will be aligned to this grid.
    """
    with rasterio.open(dem_path) as src:
        return src.transform, src.shape


def rasterize_vector(
    vector_path,
    output_path: Path,
    reference_transform,
    reference_shape: tuple,
    reference_crs: str = TARGET_CRS,
    burn_value: int = 1,
):
    """
    Rasterize a vector file (shapefile/geojson) onto the reference grid.
    Accepts either a Path to a single file, or a pre-loaded GeoDataFrame
    (useful when merging multiple shapefiles before rasterizing).
    """
    import geopandas as gpd
    from rasterio.features import rasterize as rio_rasterize

    if isinstance(vector_path, gpd.GeoDataFrame):
        gdf = vector_path
    else:
        gdf = gpd.read_file(vector_path)

    # Reproject to target CRS if needed
    if gdf.crs and gdf.crs.to_epsg() != int(reference_crs.split(":")[1]):
        gdf = gdf.to_crs(reference_crs)

    # Create binary raster: 1 = flooded, 0 = not flooded
    shapes = [(geom, burn_value) for geom in gdf.geometry if geom is not None]

    out_array = rio_rasterize(
        shapes=shapes,
        out_shape=reference_shape,
        transform=reference_transform,
        fill=0,
        dtype="uint8",
    )

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": reference_shape[1],
        "height": reference_shape[0],
        "transform": reference_transform,
        "crs": CRS.from_user_input(reference_crs),
        "compress": "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_array, 1)

    flooded_pct = out_array.sum() / out_array.size * 100
    print(f"  Flood label raster saved: {output_path}")
    print(f"  Flooded pixels: {out_array.sum()} / {out_array.size} ({flooded_pct:.1f}%)")
    return output_path


def sample_raster_stack(raster_paths: dict, n_samples: int = None) -> "pd.DataFrame":
    """
    Sample aligned raster layers into a tabular DataFrame.
    Each raster_path maps to a feature name; values are read at every pixel
    (or a random subset if n_samples is set).

    Returns a DataFrame with one row per pixel, columns = feature names.
    """
    import pandas as pd

    arrays = {}
    ref_shape = None

    for name, path in raster_paths.items():
        with rasterio.open(path) as src:
            arr = src.read(1).flatten()
            if ref_shape is None:
                ref_shape = src.shape
            arrays[name] = arr

    df = pd.DataFrame(arrays)

    # Drop nodata rows (where any feature is NaN)
    df = df.dropna()

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)

    print(f"  Sampled {len(df)} valid pixels with {len(df.columns)} columns")
    return df
