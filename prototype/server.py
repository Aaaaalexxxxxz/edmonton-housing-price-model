"""FastAPI server for the Edmonton housing price prototype."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from prototype.predictor import PropertyPricePredictor

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
LRT_ROUTES_PATH = ROOT / "edmonton_lrt_routes.geojson"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "EdmontonPropertyPricePrototype/0.1 (local research prototype)"
EDMONTON_VIEWBOX = "-113.85,53.72,-113.25,53.30"
EDMONTON_BOUNDS = {
    "latitude": (53.30, 53.72),
    "longitude": (-113.85, -113.25),
}
_last_geocode_at = 0.0

app = FastAPI(title="Edmonton Housing Price Prototype", version="0.1.0")
predictor = PropertyPricePredictor()


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=53.3, le=53.7, description="Property latitude within Edmonton area")
    longitude: float = Field(..., ge=-113.8, le=-113.3, description="Property longitude within Edmonton area")
    year_built: int = Field(..., ge=1800, le=2030)
    lot_size: float = Field(..., gt=0, le=50_000, description="Lot size in square meters")
    garage: str = Field(..., pattern="^[YyNn]$")
    zoning: str = Field(..., min_length=1, max_length=20)
    assessment_year: int = Field(default=2023, ge=2012, le=2023)


class GeocodeRequest(BaseModel):
    address: str = Field(..., min_length=3, max_length=200)


def _within_edmonton(latitude: float, longitude: float) -> bool:
    lat_min, lat_max = EDMONTON_BOUNDS["latitude"]
    lon_min, lon_max = EDMONTON_BOUNDS["longitude"]
    return lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max


def _normalize_query(address: str) -> str:
    query = address.strip()
    if "edmonton" not in query.lower():
        query = f"{query}, Edmonton, Alberta, Canada"
    return query


def _respect_rate_limit() -> None:
    global _last_geocode_at
    elapsed = time.monotonic() - _last_geocode_at
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)


def _query_nominatim(query: str, *, bounded: bool, limit: int = 1) -> list[dict]:
    global _last_geocode_at

    _respect_rate_limit()

    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "limit": limit,
            "countrycodes": "ca",
            "viewbox": EDMONTON_VIEWBOX,
            "bounded": "1" if bounded else "0",
            "addressdetails": "1",
        }
    )
    request = urllib.request.Request(
        f"{NOMINATIM_URL}?{params}",
        headers={"User-Agent": NOMINATIM_USER_AGENT},
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            _last_geocode_at = time.monotonic()
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Geocoder returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail="Geocoder is unavailable") from exc


def _format_geocode_hit(hit: dict, query: str) -> dict:
    address = hit.get("address", {})
    short_label = address.get("house_number")
    if short_label and address.get("road"):
        short_label = f"{short_label} {address['road']}"
    elif address.get("road"):
        short_label = address["road"]
    else:
        short_label = hit.get("display_name", query).split(",")[0]

    neighbourhood = address.get("neighbourhood") or address.get("suburb") or address.get("city_district")
    if neighbourhood and neighbourhood not in short_label:
        short_label = f"{short_label}, {neighbourhood}"

    return {
        "latitude": float(hit["lat"]),
        "longitude": float(hit["lon"]),
        "display_name": hit.get("display_name", query),
        "short_label": short_label,
    }


def _edmonton_hits(results: list[dict]) -> list[dict]:
    seen: set[tuple[float, float]] = set()
    hits: list[dict] = []
    for hit in results:
        latitude = float(hit["lat"])
        longitude = float(hit["lon"])
        if not _within_edmonton(latitude, longitude):
            continue
        key = (round(latitude, 5), round(longitude, 5))
        if key in seen:
            continue
        seen.add(key)
        hits.append(hit)
    return hits


@app.get("/api/geocode/suggest")
def geocode_suggest(q: str = Query(..., min_length=3, max_length=200)) -> dict:
    query = _normalize_query(q)
    results = _query_nominatim(query, bounded=True, limit=5)
    if not results:
        results = _query_nominatim(query, bounded=False, limit=8)

    suggestions = [_format_geocode_hit(hit, query) for hit in _edmonton_hits(results)[:5]]
    return {"suggestions": suggestions}


@app.post("/api/geocode")
def geocode(body: GeocodeRequest) -> dict:
    query = _normalize_query(body.address)

    results = _query_nominatim(query, bounded=True, limit=1)
    if not results:
        results = _query_nominatim(query, bounded=False, limit=1)

    if not results:
        raise HTTPException(status_code=404, detail="Address not found. Try adding a street, city, or postal code.")

    edmonton_results = _edmonton_hits(results)
    if not edmonton_results:
        raise HTTPException(
            status_code=422,
            detail="Address resolved outside Edmonton. Please enter an Edmonton address.",
        )

    result = _format_geocode_hit(edmonton_results[0], query)
    return {**result, "query": query}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
def meta() -> dict:
    zones = sorted(predictor.known_zones)
    if "OTHER" not in zones:
        zones.append("OTHER")

    lrt_stations = [
        {
            "name": row["LRT Stop Description"],
            "latitude": float(row["Latitude"]),
            "longitude": float(row["Longitude"]),
        }
        for _, row in predictor.lrt_df.iterrows()
    ]

    return {
        "model_metrics": predictor.metrics,
        "zoning_options": zones,
        "defaults": {
            "latitude": 53.5461,
            "longitude": -113.4938,
            "year_built": 1995,
            "lot_size": 450,
            "garage": "Y",
            "zoning": "RF1",
            "assessment_year": 2023,
        },
        "lrt_stations": lrt_stations,
        "map_center": {"latitude": 53.5461, "longitude": -113.4938},
    }


@app.get("/api/lrt-routes")
def lrt_routes() -> dict:
    if not LRT_ROUTES_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="LRT route data not found. Download edmonton_lrt_routes.geojson from City of Edmonton open data.",
        )
    return json.loads(LRT_ROUTES_PATH.read_text())


@app.post("/api/predict")
def predict(body: PredictRequest) -> dict:
    try:
        result = predictor.predict(
            latitude=body.latitude,
            longitude=body.longitude,
            year_built=body.year_built,
            lot_size=body.lot_size,
            garage=body.garage,
            zoning=body.zoning,
            assessment_year=body.assessment_year,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "predicted_price": round(result.predicted_price),
        "predicted_price_formatted": f"${result.predicted_price:,.0f}",
        "location": {"latitude": result.latitude, "longitude": result.longitude},
        "spatial_features": result.spatial_features,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
