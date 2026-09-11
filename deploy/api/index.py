"""
Flood Susceptibility Mapping — Lightweight Flask App for Vercel Deployment

Serves an interactive Leaflet.js map with downsampled flood risk overlay,
model metrics, and location click-to-analyze.
"""
from flask import Flask, render_template, jsonify, send_from_directory
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)

# ── Cache loaded data ────────────────────────────────────────────────
_risk_data = None
_metrics = None


def _load_risk_data():
    global _risk_data
    if _risk_data is None:
        with open(STATIC_DIR / "risk_data.json") as f:
            _risk_data = json.load(f)
    return _risk_data


def _load_metrics():
    global _metrics
    if _metrics is None:
        path = STATIC_DIR / "model_metrics.json"
        if path.exists():
            with open(path) as f:
                _metrics = json.load(f)
    return _metrics


# ── Routes ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/risk")
def get_risk():
    """Return downsampled risk map data."""
    return jsonify(_load_risk_data())


@app.route("/api/metrics")
def get_metrics():
    """Return model evaluation metrics."""
    m = _load_metrics()
    return jsonify(m if m else {"error": "metrics not found"})


@app.route("/api/analyze")
def analyze_location():
    """Given lat/lon, return risk score from downsampled grid."""
    from flask import request
    import numpy as np

    lat = float(request.args.get("lat", 0))
    lon = float(request.args.get("lon", 0))

    data = _load_risk_data()
    bounds = data["bounds"]
    grid = np.array(data["data"])
    rows, cols = grid.shape

    # Map lat/lon to grid indices
    row_frac = (bounds["north"] - lat) / (bounds["north"] - bounds["south"])
    col_frac = (lon - bounds["west"]) / (bounds["east"] - bounds["west"])

    row = int(row_frac * rows)
    col = int(col_frac * cols)

    if 0 <= row < rows and 0 <= col < cols:
        score = float(grid[row, col])
        if score < 0:
            return jsonify({"error": "No data at this location"})

        level = (
            "Low" if score < 0.3
            else "Moderate" if score < 0.6
            else "High" if score < 0.8
            else "Very High"
        )
        return jsonify({
            "lat": lat, "lon": lon,
            "risk_score": round(score, 4),
            "risk_level": level,
        })
    else:
        return jsonify({"error": "Outside study area"})


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)


# ── Entry point for local dev ────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
