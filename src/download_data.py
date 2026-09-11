"""
Week 1 — Data Acquisition Script

Downloads / guides manual download of all required datasets for the
Sindh 2022 flood susceptibility study.

Datasets:
  1. Copernicus DEM 30m (elevation) — from AWS S3, NO API key needed
  2. CHIRPS pentad rainfall — programmatic download (auto-decompressed)
  3. Copernicus EMS flood extent — manual download (instructions printed)
  4. ESA WorldCover land cover — manual download (instructions printed)
  5. HydroRIVERS network — manual download (instructions printed)
"""
import gzip
import shutil
import requests
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    STUDY_AREA_BBOX, RAW_DIR,
    COPERNICUS_DEM_HTTPS,
    CHIRPS_BASE_URL, EMS_URL_HINT, WORLD_COVER_URL, HYDROSHEDS_URL,
)


# ── 1. DEM Download (Copernicus, no API key) ─────────────────────────

def _dem_tile_url(lat: int, lon: int) -> str:
    """
    Build HTTPS URL for a single Copernicus DEM 30m tile.
    Tile naming: Copernicus_DSM_COG_10_N{lat}_00_E{lon}_00_DEM
    (resolution=10 arcsec = 30m; N/S and E/W prefix based on sign)
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    tile_name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    return f"{COPERNICUS_DEM_HTTPS}/{tile_name}/{tile_name}.tif"


def _dem_tile_list() -> list:
    """
    Generate list of (lat, lon) for all 1°x1° tiles covering the study area.
    Copernicus tiles are indexed by their SW corner coordinate.
    """
    bbox = STUDY_AREA_BBOX
    lat_start = int(bbox["lat_min"])
    lat_end = int(bbox["lat_max"])
    lon_start = int(bbox["lon_min"])
    lon_end = int(bbox["lon_max"])

    tiles = []
    for lat in range(lat_start, lat_end):
        for lon in range(lon_start, lon_end):
            tiles.append((lat, lon))
    return tiles


def _validate_tile(tile_path: Path) -> bool:
    """Check if a GeoTIFF tile is readable (not corrupted/partial)."""
    import rasterio
    try:
        with rasterio.open(tile_path) as src:
            src.read(1)  # attempt to read band 1
        return True
    except Exception:
        return False


def download_dem(output_path: Path = None):
    """
    Download Copernicus DEM 30m tiles covering the study area from AWS S3.
    NO API key or AWS account required. Downloads individual 1°x1° tiles
    and mosaics them into a single GeoTIFF.
    Supports resume: validates cached tiles, re-downloads corrupted ones.
    """
    if output_path is None:
        output_path = RAW_DIR / "srtm_dem.tif"  # keep same name for pipeline compat

    tiles_dir = RAW_DIR / "dem_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    tile_coords = _dem_tile_list()
    expected_count = len(tile_coords)

    # Check which tiles are already downloaded AND valid (not corrupted)
    valid_tiles = []
    missing_coords = []
    for lat, lon in tile_coords:
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        tile_name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
        tile_path = tiles_dir / f"{tile_name}.tif"
        if tile_path.exists():
            if _validate_tile(tile_path):
                valid_tiles.append(tile_path)
            else:
                # Corrupted/partial file — delete and re-download
                print(f"  ⚠ Corrupted tile detected: {tile_name}, deleting for re-download")
                tile_path.unlink()
                missing_coords.append((lat, lon))
        else:
            missing_coords.append((lat, lon))

    # If all tiles present and mosaic exists, skip entirely
    if len(valid_tiles) == expected_count and output_path.exists():
        print(f"[DEM] All {expected_count} tiles + mosaic already exist: {output_path}")
        return output_path

    # If some tiles missing (or mosaic deleted), re-download missing ones
    if not missing_coords:
        print(f"[DEM] All {expected_count} tiles valid, but mosaic missing or stale. Re-mosaicking...")
    else:
        print(f"[DEM] {len(valid_tiles)}/{expected_count} tiles valid. "
              f"Downloading {len(missing_coords)} remaining...")
        print(f"  Study area bbox: {STUDY_AREA_BBOX}")

        for lat, lon in missing_coords:
            url = _dem_tile_url(lat, lon)
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile_name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
            tile_path = tiles_dir / f"{tile_name}.tif"

            try:
                resp = requests.get(url, stream=True, timeout=180)
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(tile_path, "wb") as f:
                    for chunk in tqdm(resp.iter_content(chunk_size=65536),
                                      total=max(total // 65536, 1),
                                      desc=f"  Tile {ns}{abs(lat):02d}{ew}{abs(lon):03d}",
                                      leave=False):
                        f.write(chunk)

                # Validate the download succeeded by reading the file
                if _validate_tile(tile_path):
                    print(f"  ✓ {ns}{abs(lat):02d}{ew}{abs(lon):03d} "
                          f"({tile_path.stat().st_size / 1e6:.1f} MB)")
                else:
                    print(f"  ✗ {ns}{abs(lat):02d}{ew}{abs(lon):03d}: downloaded but corrupt, deleting")
                    tile_path.unlink()
            except requests.HTTPError as e:
                # Clean up partial file on failure
                if tile_path.exists():
                    tile_path.unlink()
                print(f"  ✗ {ns}{abs(lat):02d}{ew}{abs(lon):03d}: not available (ocean/missing)")
            except Exception as e:
                # Clean up partial file on failure
                if tile_path.exists():
                    tile_path.unlink()
                print(f"  ✗ {ns}{abs(lat):02d}{ew}{abs(lon):03d}: {e}")

    # Collect all available tiles (existing valid + newly downloaded)
    all_tiles = []
    for lat, lon in tile_coords:
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        tile_name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
        tile_path = tiles_dir / f"{tile_name}.tif"
        if tile_path.exists():
            all_tiles.append(tile_path)

    if not all_tiles:
        print("[DEM] ERROR: No tiles downloaded.")
        return None

    print(f"\n[DEM] Mosaicking {len(all_tiles)} of {expected_count} tiles...")
    if len(all_tiles) < expected_count:
        print(f"  WARNING: {expected_count - len(all_tiles)} tiles missing — mosaic will have gaps.")
        print(f"  Re-run 'python run.py download' later to fill gaps.")
    from rasterio.merge import merge as rio_merge
    import rasterio

    src_files = [rasterio.open(p) for p in all_tiles]
    mosaic, transform = rio_merge(src_files)

    profile = src_files[0].profile.copy()
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)

    for f in src_files:
        f.close()

    print(f"[DEM] Mosaic saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Shape: {mosaic.shape[1]} x {mosaic.shape[2]} pixels")
    return output_path


# ── 2. CHIRPS Rainfall Download ───────────────────────────────────────

def download_chirps_rainfall(output_path: Path = None):
    """
    Download CHIRPS pentad rainfall data for August 2022.
    File format: chirps-v2.0.{YEAR}.{MONTH}.{PENTAD_1-6}.tif.gz
    Pentad numbers: 1=days 1-5, 2=days 6-10, ..., 6=days 26-31
    Files are gzipped — auto-decompressed after download.
    """
    if output_path is None:
        output_path = RAW_DIR / "chirps_aug2022"
    output_path.mkdir(parents=True, exist_ok=True)

    # August 2022: pentads 1-6
    pentads = [1, 2, 3, 4, 5, 6]
    files = []

    for pentad in pentads:
        # Correct filename format (verified from CHIRPS server directory listing)
        filename_gz = f"chirps-v2.0.2022.08.{pentad}.tif.gz"
        filename_tif = f"chirps-v2.0.2022.08.{pentad}.tif"
        url = f"{CHIRPS_BASE_URL}{filename_gz}"
        local_gz = output_path / filename_gz
        local_tif = output_path / filename_tif

        if local_tif.exists():
            print(f"  [CHIRPS] Already have: {filename_tif}")
            files.append(local_tif)
            continue

        print(f"  [CHIRPS] Downloading {filename_gz}...")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            # Download .gz file
            with open(local_gz, "wb") as f:
                for chunk in tqdm(resp.iter_content(chunk_size=65536),
                                  total=max(total // 65536, 1),
                                  desc=f"    Pentad {pentad}",
                                  leave=False):
                    f.write(chunk)

            # Decompress .gz → .tif
            with gzip.open(local_gz, "rb") as f_in:
                with open(local_tif, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove .gz to save space
            local_gz.unlink()
            files.append(local_tif)
            print(f"  [CHIRPS] ✓ {filename_tif} ({local_tif.stat().st_size / 1e6:.1f} MB)")

        except requests.HTTPError as e:
            print(f"  [CHIRPS] ✗ Pentad {pentad}: {e}")
        except Exception as e:
            print(f"  [CHIRPS] ✗ Pentad {pentad}: {e}")

    print(f"[CHIRPS] Downloaded {len(files)} pentad files to {output_path}")
    return files


# ── 3. Manual Download Instructions ──────────────────────────────────

def print_manual_download_instructions():
    """
    Print instructions for datasets that require manual download.
    These 3 datasets need free registration on their respective portals.
    """
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)

    print(f"""
[1] COPERNICUS EMS FLOOD EXTENT (Ground Truth Labels)
    Two activations cover the study area — download BOTH:

    [1a] EMSR629 (Aug 2022, northern Sindh — Larkana, Shikarpur, Jacobabad)
         URL: https://mapping.emergency.copernicus.eu/activations/EMSR629/
         Download: "Delineation Product" > Vector Package (.zip) for each AOI

    [1b] EMSR631 (Sep 2022, central Sindh — Sanghar, Khipro)
         URL: https://mapping.emergency.copernicus.eu/activations/EMSR631/
         Download: "Delineation Product" > Vector Package (.zip) for each AOI

    Extract ALL zips into: {RAW_DIR / 'ems_flood_extent'}
    (The script picks up any .shp or .geojson in that folder.)

[2] ESA WORLDCOVER (Land Cover)
    URL: {WORLD_COVER_URL}
    Steps:
      1. Go to the data download page
      2. Select WorldCover 2021 product
      3. Download tiles covering Sindh, Pakistan (tiles N20E060, N20E070)
      4. Place GeoTIFF(s) in: {RAW_DIR / 'worldcover'}
    Alternative: Use the AWS Open Data S3 bucket (instructions on the page).

[3] HYDROSHEDS — HydroRIVERS (River Network)
    URL: {HYDROSHEDS_URL}
    Steps:
      1. Register for a free HydroSHEDS account
      2. Download HydroRIVERS v10 for South Asia (or global)
      3. Extract shapefile and place in: {RAW_DIR / 'hydrorivers'}
    This gives you the river/stream network for distance calculations.
""")


# ── Main ──────────────────────────────────────────────────────────────

def run_all():
    """Run all automated downloads, then print manual instructions."""
    print("=" * 70)
    print("FLOOD SUSCEPTIBILITY MVP — DATA ACQUISITION")
    print("=" * 70)

    print("\n── Automated Downloads (no API keys needed) ──")
    download_dem()
    download_chirps_rainfall()

    print_manual_download_instructions()

    print("\n" + "=" * 70)
    print("DATA ACQUISITION COMPLETE")
    print("Next step: Run 'python run.py process' for Week 2 processing")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
