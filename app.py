"""
Week 4 — Interactive Dashboard

Streamlit + Folium dashboard for the flood susceptibility map.
Features:
  - Toggle between risk map / actual flood extent / rainfall layer
  - Click a location to see risk score and contributing factors
  - Model performance metrics display
"""
import streamlit as st
import folium
from streamlit_folium import st_folium, folium_static
import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, PROCESSED_DIR, STUDY_AREA_BBOX, FEATURES

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Susceptibility — Sindh 2022",
    page_icon="🌊",
    layout="wide",
)

st.title(" Flood Susceptibility Mapping — Sindh 2022 Floods")
st.markdown(
    "ML-based flood risk assessment for the Sindh province, Pakistan. "
    "Trained on Copernicus EMS ground truth from the August 2022 flood event."
)


# ── Load Data ─────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    """Load model evaluation metrics."""
    metrics_path = OUTPUT_DIR / "model_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


@st.cache_data
def load_risk_data():
    """Load risk map as numpy array."""
    import rasterio
    risk_path = OUTPUT_DIR / "flood_risk_map.tif"
    if not risk_path.exists():
        return None, None, None
    with rasterio.open(risk_path) as src:
        data = src.read(1)
        bounds = src.bounds
        transform = src.transform
    return data, bounds, transform


# ── Sidebar Controls ──────────────────────────────────────────────────
st.sidebar.header("Layer Controls")

show_risk_map = st.sidebar.checkbox("Flood Risk Map", value=True)
show_flood_extent = st.sidebar.checkbox("Actual Flood Extent (EMS)", value=False)
show_rainfall = st.sidebar.checkbox("Rainfall Layer", value=False)
show_rivers = st.sidebar.checkbox("River Network", value=False)

risk_threshold = st.sidebar.slider(
    "Risk Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Pixels above this probability are highlighted as high-risk"
)

colormap = st.sidebar.selectbox(
    "Color Scheme", ["RdYlGn_r", "viridis", "plasma", "YlOrRd", "Blues"],
    index=0
)


# ── Metrics Panel ─────────────────────────────────────────────────────
metrics = load_metrics()
if metrics:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Performance")
    st.sidebar.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
    st.sidebar.metric("Avg Precision", f"{metrics['average_precision']:.3f}")
else:
    st.sidebar.warning("Model metrics not found. Run model training first.")


# ── Map ───────────────────────────────────────────────────────────────
st.markdown("## Interactive Risk Map")

# Center map on study area
center_lat = (STUDY_AREA_BBOX["lat_min"] + STUDY_AREA_BBOX["lat_max"]) / 2
center_lon = (STUDY_AREA_BBOX["lon_min"] + STUDY_AREA_BBOX["lon_max"]) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=7,
    tiles="OpenStreetMap",
)

# Risk map overlay
risk_data, bounds, transform = load_risk_data()
if risk_data is not None and show_risk_map:
    from folium.raster_layers import ImageOverlay
    from matplotlib import cm
    import matplotlib.colors as mcolors

    # Convert risk array to RGBA image
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap(colormap)
    risk_rgba = cmap(norm(np.ma.masked_where(np.isnan(risk_data), risk_data)))

    # Convert to uint8 image for folium
    risk_img = (risk_rgba[:, :, :3] * 255).astype(np.uint8)
    # Set low-risk pixels to semi-transparent
    risk_img_rgba = np.zeros((*risk_img.shape[:2], 4), dtype=np.uint8)
    risk_img_rgba[:, :, :3] = risk_img
    alpha = np.where(np.isnan(risk_data), 0,
                     np.where(risk_data < risk_threshold, 50, 180)).astype(np.uint8)
    risk_img_rgba[:, :, 3] = alpha

    ImageOverlay(
        image=risk_img_rgba,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        name="Flood Risk",
        opacity=0.7,
    ).add_to(m)

# Add base layers (all free, no API key needed)
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>',
    name="Light Base",
).add_to(m)
folium.TileLayer("Esri WorldImagery", name="Satellite").add_to(m)

# ── Study Area Boundary ──────────────────────────────────────────────
study_group = folium.FeatureGroup(name="Study Area")
folium.Rectangle(
    bounds=[
        [STUDY_AREA_BBOX["lat_min"], STUDY_AREA_BBOX["lon_min"]],
        [STUDY_AREA_BBOX["lat_max"], STUDY_AREA_BBOX["lon_max"]],
    ],
    color="#FF4500",
    weight=3,
    dashArray="8, 4",
    fill=False,
    tooltip="Study Area — Sindh 2022 Flood Susceptibility",
).add_to(study_group)
study_group.add_to(m)

# ── Key Cities in Sindh ──────────────────────────────────────────────
cities = [
    ("Hyderabad",   25.396, 68.358),
    ("Sukkur",      27.705, 68.857),
    ("Larkana",     27.555, 68.214),
    ("Nawabshah",   26.248, 68.410),
    ("Mirpur Khas", 25.528, 69.013),
    ("Jacobabad",   28.283, 68.438),
    ("Khairpur",    27.530, 68.759),
    ("Shikarpur",   27.956, 68.638),
    ("Thatta",      24.747, 67.924),
    ("Badin",       24.656, 68.837),
]

city_group = folium.FeatureGroup(name="Cities")
for name, lat, lon in cities:
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color="#1a1a2e",
        fillColor="#FF4500",
        fillOpacity=0.9,
        weight=2,
        tooltip=f"<b>{name}</b>",
    ).add_to(city_group)
    folium.map.Marker(
        [lat, lon],
        icon=folium.DivIcon(
            html=f'<div style="font-size:11px;font-weight:bold;color:#1a1a2e;'
                 f'text-shadow:1px 1px 2px white,-1px -1px 2px white,'
                 f'1px -1px 2px white,-1px 1px 2px white;'
                 f'white-space:nowrap">{name}</div>',
            icon_size=(100, 20),
            icon_anchor=(-8, 10),
        ),
    ).add_to(city_group)
city_group.add_to(m)

# Layer control
folium.LayerControl().add_to(m)

# Click handler
m.add_child(folium.LatLngPopup())

# Render map
map_data = st_folium(m, width=1200, height=600, returned_objects=["last_clicked"])


# ── Click Analysis ────────────────────────────────────────────────────
st.markdown("## Location Analysis")

clicked = map_data.get("last_clicked")
if clicked:
    lat, lon = clicked["lat"], clicked["lng"]
    st.info(f"📍 Clicked: {lat:.4f}°N, {lon:.4f}°E")

    if risk_data is not None and transform is not None:
        # Convert lat/lon (WGS84) to UTM Zone 42N before pixel lookup
        from rasterio.transform import rowcol
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32642", always_xy=True)
        x_utm, y_utm = transformer.transform(lon, lat)
        row, col = rowcol(transform, x_utm, y_utm)

        if 0 <= row < risk_data.shape[0] and 0 <= col < risk_data.shape[1]:
            risk_score = risk_data[row, col]
            if np.isnan(risk_score):
                st.warning("No data at this location (outside coverage area).")
            else:
                # Risk gauge
                risk_level = (
                    "🟢 Low" if risk_score < 0.3
                    else "🟡 Moderate" if risk_score < 0.6
                    else "🟠 High" if risk_score < 0.8
                    else "🔴 Very High"
                )

                col1, col2 = st.columns(2)
                col1.metric("Risk Score", f"{risk_score:.3f}", risk_level)
                col2.metric("Risk Level", risk_level)

                # Contributing factors (if raster data available)
                st.subheader("Contributing Factors")
                factor_data = {}
                for feature in FEATURES:
                    raster_map = {
                        "elevation": PROCESSED_DIR / "dem_aligned.tif",
                        "slope": PROCESSED_DIR / "slope.tif",
                        "twi": PROCESSED_DIR / "twi.tif",
                        "distance_to_river": PROCESSED_DIR / "distance_to_river.tif",
                        "land_cover": PROCESSED_DIR / "landcover_aligned.tif",
                        "rainfall_max": PROCESSED_DIR / "rainfall_max.tif",
                        "rainfall_cumulative": PROCESSED_DIR / "rainfall_cumulative.tif",
                    }
                    path = raster_map.get(feature)
                    if path and path.exists():
                        import rasterio
                        with rasterio.open(path) as src:
                            val = src.read(1)[row, col]
                            if not np.isnan(val):
                                factor_data[feature] = val

                if factor_data:
                    factor_cols = st.columns(len(factor_data))
                    for i, (name, val) in enumerate(factor_data.items()):
                        units = {
                            "elevation": "m", "slope": "°", "twi": "",
                            "distance_to_river": "m", "land_cover": "class",
                            "rainfall_max": "mm/pentad", "rainfall_cumulative": "mm",
                        }
                        unit = units.get(name, "")
                        factor_cols[i].metric(
                            name.replace("_", " ").title(),
                            f"{val:.1f} {unit}" if unit != "class" else f"Class {int(val)}"
                        )
        else:
            st.warning("Clicked location is outside the study area.")
else:
    st.info("👆 Click on the map to see flood risk details for a specific location.")


# ── Bottom Section: Methodology & Limitations ─────────────────────────
st.markdown("---")
st.markdown("""
### Methodology
- **Features**: Elevation, slope, Topographic Wetness Index (TWI), distance to river,
  land cover, peak & cumulative rainfall (August 2022)
- **Ground Truth**: Copernicus EMS Rapid Mapping flood extent polygons (EMSR631)
- **Model**: XGBoost classifier with spatial block cross-validation
- **Data Sources**: SRTM DEM, CHIRPS rainfall, ESA WorldCover, HydroRIVERS

### Limitations
- This is a **susceptibility model**, not a real-time forecast. It estimates the
  *propensity* of an area to flood given environmental conditions.
- Temporal dynamics (flood wave propagation, dam releases) are not captured.
- Resolution: ~90m analysis grid — individual buildings/roads not resolved.
- Trained on a single event (2022) — may not generalize to different flood mechanisms.
""")
