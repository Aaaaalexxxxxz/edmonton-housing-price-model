"""
PyTorch neural network for Edmonton residential property price prediction.

Uses historical assessment records, property coordinates, proximity to LRT,
University of Alberta, West Edmonton Mall, EPSB school catchment areas,
and recreation facilities as features.
Data sources (in `data/`):
  - Property_Assessment_Data.csv
  - Edmonton_lrt_stations.csv
  - edmonton_school_catchments.csv (extracted locations + catchment polygons)
  - edmonton_recreation_facilities.csv
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import wkt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROPERTY_CSV = DATA_DIR / "Property_Assessment_Data.csv"
LRT_CSV = DATA_DIR / "Edmonton_lrt_stations.csv"
SCHOOL_CSV = DATA_DIR / "edmonton_school_catchments.csv"
SCHOOL_LOCATIONS_CSV = DATA_DIR / "edmonton_school_locations.csv"
SCHOOL_POLYGONS_CSV = DATA_DIR / "edmonton_school_catchment_polygons.csv"
REC_CSV = DATA_DIR / "edmonton_recreation_facilities.csv"
DEFAULT_MODEL_PATH = DATA_DIR / "housing_price_pytorch.pt"
DEFAULT_ARTIFACTS_PATH = DATA_DIR / "housing_price_pytorch_artifacts.json"

MIN_PRICE = 50_000
MAX_PRICE = 2_500_000
UTM_EPSG = 32612
TOP_ZONING_CATEGORIES = 40
SCHOOL_TYPE_BUCKETS = {
    "elementary": {"EL"},
    "junior": {"EJ", "JR", "JS", "EJS"},
    "senior": {"SR"},
}
REC_FACILITY_BUCKETS = {
    "Recreation_Centre": {"Recreation Centre"},
    "Arena": {"Arena"},
    "Park": {"River Valley Park", "City Park"},
    "Pool": {"Outdoor Pool"},
    "Sports": {"Staffed Sports Field", "Tennis Court", "Golf Course"},
}
# WGS84 coordinates for key Edmonton destinations
UOFa_COORDS = (53.5232, -113.5263)  # University of Alberta main campus
WEST_EDMONTON_MALL_COORDS = (53.5230, -113.6242)  # West Edmonton Mall


@dataclass
class TrainConfig:
    sample_size: int | None = 500_000
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    batch_size: int = 4096
    epochs: int = 100
    patience: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.15
    lrt_radii_m: tuple[int, ...] = (500, 1000, 2000)
    school_radii_m: tuple[int, ...] = (500, 1000, 2000)
    rec_radii_m: tuple[int, ...] = (500, 1000, 2000)
    price_quantile_low: float = 0.01
    price_quantile_high: float = 0.99


class HousingPriceNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def parse_assessed_value(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)


def parse_lot_size(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^\d.]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def load_property_data(path: Path = PROPERTY_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["Assessed Value"] = parse_assessed_value(df["Assessed Value"])
    df["Lot Size"] = parse_lot_size(df["Lot Size"])

    residential = df["Assessment Class 1"] == "RESIDENTIAL"
    in_range = (df["Assessed Value"] >= MIN_PRICE) & (df["Assessed Value"] <= MAX_PRICE)
    df = df[residential & in_range].copy()

    df = df.dropna(subset=["Latitude", "Longitude", "Lot Size", "Assessment Year"])
    median_year = df["Actual Year Built"].median()
    df["Actual Year Built"] = df["Actual Year Built"].fillna(median_year)

    drop_cols = [
        "Account Number",
        "Suite",
        "House Number",
        "Street Name",
        "Legal Description",
        "Point Location",
        "Assessment Class 1",
        "Assessment Class % 1",
        "Assessment Class 2",
        "Assessment Class % 2",
        "Assessment Class 3",
        "Assessment Class % 3",
        "Neighbourhood",
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def filter_price_outliers(
    df: pd.DataFrame,
    low_q: float,
    high_q: float,
) -> pd.DataFrame:
    lower = df["Assessed Value"].quantile(low_q)
    upper = df["Assessed Value"].quantile(high_q)
    filtered = df[(df["Assessed Value"] >= lower) & (df["Assessed Value"] <= upper)].copy()
    print(f"  Price filter ({low_q:.0%}-{high_q:.0%}): ${lower:,.0f} to ${upper:,.0f} -> {len(filtered):,} rows")
    return filtered


def collapse_rare_zoning(df: pd.DataFrame, top_n: int = TOP_ZONING_CATEGORIES) -> pd.DataFrame:
    df = df.copy()
    top_zones = df["Zoning"].value_counts().head(top_n).index
    df["Zoning"] = df["Zoning"].where(df["Zoning"].isin(top_zones), "OTHER")
    return df


def load_lrt_stations(path: Path = LRT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df.drop_duplicates(subset=["Latitude", "Longitude"], keep="first")
    return df.reset_index(drop=True)


def fix_lat_lon(lat_series: pd.Series, lon_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Normalize EPSB coordinates where latitude/longitude columns are occasionally swapped."""
    lat = pd.to_numeric(lat_series, errors="coerce")
    lon = pd.to_numeric(lon_series, errors="coerce")
    swapped = lat.abs() > 90
    fixed_lat = lat.where(~swapped, lon)
    fixed_lon = lon.where(~swapped, lat)
    return fixed_lat, fixed_lon


def extract_school_catchment_data(
    source_path: Path = SCHOOL_CSV,
    locations_path: Path = SCHOOL_LOCATIONS_CSV,
    polygons_path: Path = SCHOOL_POLYGONS_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract corrected school coordinates and catchment polygons from the EPSB export."""
    raw = pd.read_csv(source_path, low_memory=False)

    if "school_name" in raw.columns and "catchment_wkt" in raw.columns:
        locations = raw.dropna(subset=["latitude", "longitude"]).copy()
        locations["latitude"], locations["longitude"] = fix_lat_lon(
            locations["latitude"], locations["longitude"]
        )
        polygons = raw[raw["catchment_wkt"].notna() & raw["catchment_wkt"].astype(str).str.len().gt(0)].copy()
        if not locations_path.exists():
            locations.to_csv(locations_path, index=False)
        if not polygons_path.exists():
            polygons[["year", "epsb_id", "school_name", "school_type", "catchment_wkt"]].to_csv(
                polygons_path, index=False
            )
        return locations, polygons[["year", "epsb_id", "school_name", "school_type", "catchment_wkt"]]

    # Original EPSB column names: latitude/longitude are swapped.
    locations = pd.DataFrame(
        {
            "year": raw["Year"],
            "epsb_id": raw["EPSB_ID"],
            "school_name": raw["School_Nam"],
            "school_type": raw["Sch_Type"],
            "grades": raw["Grades"],
            "latitude": pd.to_numeric(raw["Longitude"], errors="coerce"),
            "longitude": pd.to_numeric(raw["Latitude"], errors="coerce"),
            "sector": raw["Sector"],
            "epsb_ward": raw["epsb_ward"],
        }
    )
    locations["latitude"], locations["longitude"] = fix_lat_lon(
        locations["latitude"], locations["longitude"]
    )
    locations = locations.dropna(subset=["latitude", "longitude"]).drop_duplicates(subset=["epsb_id"], keep="first")

    polygons = pd.DataFrame(
        {
            "year": raw["Year"],
            "epsb_id": raw["EPSB_ID"],
            "school_name": raw["School_Nam"],
            "school_type": raw["Sch_Type"],
            "catchment_wkt": raw["Catchment Polygon"],
        }
    )
    polygons = polygons[
        polygons["catchment_wkt"].notna() & polygons["catchment_wkt"].astype(str).str.len().gt(0)
    ].drop_duplicates(subset=["epsb_id"], keep="first")

    combined = locations.merge(
        polygons.drop(columns=["school_name", "school_type", "year"], errors="ignore"),
        on="epsb_id",
        how="left",
    )
    locations.to_csv(locations_path, index=False)
    polygons.to_csv(polygons_path, index=False)
    combined.to_csv(source_path, index=False)
    return locations, polygons


def load_school_data(
    locations_path: Path = SCHOOL_LOCATIONS_CSV,
    polygons_path: Path = SCHOOL_POLYGONS_CSV,
    source_path: Path = SCHOOL_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if locations_path.exists() and polygons_path.exists():
        locations = pd.read_csv(locations_path)
        polygons = pd.read_csv(polygons_path)
        locations["latitude"], locations["longitude"] = fix_lat_lon(
            locations["latitude"], locations["longitude"]
        )
        locations = locations.dropna(subset=["latitude", "longitude"])
        polygons = polygons.dropna(subset=["catchment_wkt", "epsb_id", "school_type"])
        return locations, polygons
    return extract_school_catchment_data(source_path, locations_path, polygons_path)


def _school_bucket(school_type: str) -> str | None:
    for bucket, types in SCHOOL_TYPE_BUCKETS.items():
        if school_type in types:
            return bucket
    return None


def _add_nearest_school_distances(
    df: pd.DataFrame,
    schools: pd.DataFrame,
    bucket: str,
    prop_x: np.ndarray,
    prop_y: np.ndarray,
    radii_m: tuple[int, ...],
) -> pd.DataFrame:
    subset = schools[schools["school_bucket"] == bucket].dropna(subset=["latitude", "longitude"])
    prefix = bucket.capitalize()

    if subset.empty:
        df[f"Distance_to_nearest_{prefix}_School_m"] = np.nan
        df[f"Log_Distance_to_nearest_{prefix}_School"] = np.nan
        for radius in radii_m:
            df[f"{prefix}_Schools_within_{radius}m"] = 0
        return df

    school_x, school_y = project_to_utm(subset["longitude"].to_numpy(), subset["latitude"].to_numpy())
    tree = cKDTree(np.column_stack([school_x, school_y]))
    prop_points = np.column_stack([prop_x, prop_y])

    distances, _ = tree.query(prop_points, k=1)
    df[f"Distance_to_nearest_{prefix}_School_m"] = distances
    df[f"Log_Distance_to_nearest_{prefix}_School"] = np.log1p(distances)

    for radius in radii_m:
        df[f"{prefix}_Schools_within_{radius}m"] = tree.query_ball_point(
            prop_points, r=radius, return_length=True
        )
    return df


def _add_catchment_assignments(
    df: pd.DataFrame,
    polygons: pd.DataFrame,
    locations: pd.DataFrame,
    prop_x: np.ndarray,
    prop_y: np.ndarray,
) -> pd.DataFrame:
    if polygons.empty:
        for bucket in SCHOOL_TYPE_BUCKETS:
            prefix = bucket.capitalize()
            df[f"In_{prefix}_Catchment"] = 0
            df[f"Distance_to_Assigned_{prefix}_School_m"] = np.nan
        return df

    polygons = polygons.copy()
    polygons["school_bucket"] = polygons["school_type"].map(_school_bucket)
    polygons = polygons[polygons["school_bucket"].notna()].copy()
    polygons["geometry"] = polygons["catchment_wkt"].map(wkt.loads)

    location_lookup = locations.set_index("epsb_id")[["latitude", "longitude", "school_bucket"]]
    gdf_props = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )

    for bucket in SCHOOL_TYPE_BUCKETS:
        prefix = bucket.capitalize()
        bucket_polys = polygons[polygons["school_bucket"] == bucket]
        in_col = f"In_{prefix}_Catchment"
        dist_col = f"Distance_to_Assigned_{prefix}_School_m"

        df[in_col] = 0
        df[dist_col] = np.nan

        if bucket_polys.empty:
            continue

        gdf_polys = gpd.GeoDataFrame(bucket_polys, geometry="geometry", crs="EPSG:4326")
        joined = gpd.sjoin(gdf_props, gdf_polys[["epsb_id", "geometry"]], how="left", predicate="within")

        if joined["index_right"].notna().any():
            matched = joined[joined["index_right"].notna()].copy()
            matched = matched[~matched.index.duplicated(keep="first")]
            df.loc[matched.index, in_col] = 1

            school_coords = location_lookup.loc[matched["epsb_id"]]
            school_x, school_y = project_to_utm(
                school_coords["longitude"].to_numpy(),
                school_coords["latitude"].to_numpy(),
            )
            prop_idx = matched.index.to_numpy()
            dx = prop_x[df.index.get_indexer(prop_idx)] - school_x
            dy = prop_y[df.index.get_indexer(prop_idx)] - school_y
            df.loc[prop_idx, dist_col] = np.hypot(dx, dy)

    return df


def add_school_features(
    df: pd.DataFrame,
    locations: pd.DataFrame,
    polygons: pd.DataFrame,
    radii_m: tuple[int, ...] = (500, 1000, 2000),
) -> pd.DataFrame:
    df = df.copy()
    locations = locations.copy()
    locations["school_bucket"] = locations["school_type"].map(_school_bucket)
    locations = locations[locations["school_bucket"].notna()]

    prop_x, prop_y = project_to_utm(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())

    for bucket in SCHOOL_TYPE_BUCKETS:
        df = _add_nearest_school_distances(df, locations, bucket, prop_x, prop_y, radii_m)

    df = _add_catchment_assignments(df, polygons, locations, prop_x, prop_y)

    school_distance_cols = [c for c in df.columns if "Distance_to" in c and "School" in c]
    for col in school_distance_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def project_to_utm(lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
    x, y = transformer.transform(lons, lats)
    return np.asarray(x), np.asarray(y)


def add_lrt_features(
    df: pd.DataFrame,
    lrt_df: pd.DataFrame,
    radii_m: tuple[int, ...] = (500, 1000, 2000),
) -> pd.DataFrame:
    prop_x, prop_y = project_to_utm(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())
    lrt_x, lrt_y = project_to_utm(lrt_df["Longitude"].to_numpy(), lrt_df["Latitude"].to_numpy())

    prop_points = np.column_stack([prop_x, prop_y])
    lrt_points = np.column_stack([lrt_x, lrt_y])
    tree = cKDTree(lrt_points)

    distances, _ = tree.query(prop_points, k=1)
    df = df.copy()
    df["Distance_to_LRT_m"] = distances
    df["Log_Distance_to_LRT"] = np.log1p(distances)

    for radius in radii_m:
        counts = tree.query_ball_point(prop_points, r=radius, return_length=True)
        df[f"LRT_stations_within_{radius}m"] = counts

    return df


def add_landmark_distances(
    df: pd.DataFrame,
    prop_x: np.ndarray | None = None,
    prop_y: np.ndarray | None = None,
) -> pd.DataFrame:
    """Distance in meters to U of A and West Edmonton Mall."""
    df = df.copy()
    if prop_x is None or prop_y is None:
        prop_x, prop_y = project_to_utm(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())

    landmarks = {
        "UofA": UOFa_COORDS,
        "West_Edmonton_Mall": WEST_EDMONTON_MALL_COORDS,
    }
    for name, (lat, lon) in landmarks.items():
        landmark_x, landmark_y = project_to_utm(np.array([lon]), np.array([lat]))
        distances = np.hypot(prop_x - landmark_x[0], prop_y - landmark_y[0])
        df[f"Distance_to_{name}_m"] = distances
        df[f"Log_Distance_to_{name}"] = np.log1p(distances)

    return df


def extract_recreation_facility_data(source_path: Path = REC_CSV, dest_path: Path = REC_CSV) -> pd.DataFrame:
    """Clean recreation facility coordinates and normalize column names."""
    raw = pd.read_csv(source_path, low_memory=False)

    if "facility_name" in raw.columns:
        cleaned = raw.copy()
    else:
        cleaned = pd.DataFrame(
            {
                "category": raw["Category"].astype(str).str.strip(),
                "facility_type": raw["Facility Type"].astype(str).str.strip(),
                "facility_name": raw["Facility Name"].astype(str).str.strip(),
                "address": raw["Address"],
                "latitude": pd.to_numeric(raw["latitude"], errors="coerce"),
                "longitude": pd.to_numeric(raw["longitude"], errors="coerce"),
                "website": raw["Website"],
            }
        )

    cleaned["latitude"], cleaned["longitude"] = fix_lat_lon(cleaned["latitude"], cleaned["longitude"])
    cleaned = cleaned.dropna(subset=["latitude", "longitude"])
    cleaned = cleaned.drop_duplicates(subset=["facility_name", "latitude", "longitude"], keep="first")
    cleaned = cleaned.reset_index(drop=True)
    cleaned.to_csv(dest_path, index=False)
    return cleaned


def load_recreation_facilities(path: Path = REC_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Recreation facility data not found: {path}")
    facilities = pd.read_csv(path)
    if "facility_name" not in facilities.columns:
        facilities = extract_recreation_facility_data(path, path)
    else:
        facilities["latitude"], facilities["longitude"] = fix_lat_lon(
            facilities["latitude"], facilities["longitude"]
        )
        facilities = facilities.dropna(subset=["latitude", "longitude"])
    facilities["facility_bucket"] = facilities["facility_type"].map(_rec_facility_bucket)
    return facilities.reset_index(drop=True)


def _rec_facility_bucket(facility_type: str) -> str | None:
    if not isinstance(facility_type, str):
        return None
    for bucket, types in REC_FACILITY_BUCKETS.items():
        if facility_type in types:
            return bucket
    return None


def _add_nearest_facility_distances(
    df: pd.DataFrame,
    facilities: pd.DataFrame,
    label: str,
    prop_x: np.ndarray,
    prop_y: np.ndarray,
    radii_m: tuple[int, ...],
) -> pd.DataFrame:
    prefix = label
    subset = facilities.dropna(subset=["latitude", "longitude"])

    if subset.empty:
        df[f"Distance_to_nearest_{prefix}_m"] = np.nan
        df[f"Log_Distance_to_nearest_{prefix}"] = np.nan
        for radius in radii_m:
            df[f"{prefix}s_within_{radius}m"] = 0
        return df

    facility_x, facility_y = project_to_utm(subset["longitude"].to_numpy(), subset["latitude"].to_numpy())
    tree = cKDTree(np.column_stack([facility_x, facility_y]))
    prop_points = np.column_stack([prop_x, prop_y])

    distances, _ = tree.query(prop_points, k=1)
    df[f"Distance_to_nearest_{prefix}_m"] = distances
    df[f"Log_Distance_to_nearest_{prefix}"] = np.log1p(distances)

    for radius in radii_m:
        df[f"{prefix}s_within_{radius}m"] = tree.query_ball_point(prop_points, r=radius, return_length=True)
    return df


def add_recreation_features(
    df: pd.DataFrame,
    facilities: pd.DataFrame,
    radii_m: tuple[int, ...] = (500, 1000, 2000),
) -> pd.DataFrame:
    df = df.copy()
    facilities = facilities.copy()
    prop_x, prop_y = project_to_utm(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())

    df = _add_nearest_facility_distances(df, facilities, "Rec_Facility", prop_x, prop_y, radii_m)

    for bucket in REC_FACILITY_BUCKETS:
        bucket_facilities = facilities[facilities["facility_bucket"] == bucket]
        df = _add_nearest_facility_distances(df, bucket_facilities, bucket, prop_x, prop_y, radii_m)

    distance_cols = [col for col in df.columns if col.startswith("Distance_to_nearest_")]
    for col in distance_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = df.copy()
    df["Property_Age_at_Assessment"] = (df["Assessment Year"] - df["Actual Year Built"]).clip(lower=0)
    df["Log_Lot_Size"] = np.log1p(df["Lot Size"])

    categorical_cols = ["Zoning", "Garage"]
    encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    target = encoded["Assessed Value"].astype(float)
    feature_cols = [col for col in encoded.columns if col != "Assessed Value"]
    features = encoded[feature_cols].astype(float)
    return features, target, feature_cols


def maybe_sample(df: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or len(df) <= sample_size:
        return df
    return df.sample(n=sample_size, random_state=random_state)


def prepare_datasets(
    config: TrainConfig,
    property_path: Path = PROPERTY_CSV,
    lrt_path: Path = LRT_CSV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    print("Loading property assessment data...")
    df = load_property_data(property_path)
    print(f"  {len(df):,} residential records after filtering")

    df = filter_price_outliers(df, config.price_quantile_low, config.price_quantile_high)
    df = collapse_rare_zoning(df)

    print("Loading LRT stations and computing spatial features...")
    lrt_df = load_lrt_stations(lrt_path)
    df = add_lrt_features(df, lrt_df, radii_m=config.lrt_radii_m)
    prop_x, prop_y = project_to_utm(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())
    df = add_landmark_distances(df, prop_x, prop_y)

    print("Loading school catchments and computing school features...")
    school_locations, school_polygons = load_school_data()
    df = add_school_features(df, school_locations, school_polygons, radii_m=config.school_radii_m)
    catchment_rate = df.filter(like="In_").mean().mean()
    print(f"  Average catchment match rate: {catchment_rate:.1%}")

    print("Loading recreation facilities and computing amenity features...")
    rec_facilities = load_recreation_facilities()
    df = add_recreation_features(df, rec_facilities, radii_m=config.rec_radii_m)
    print(f"  Loaded {len(rec_facilities):,} recreation facilities")

    df = maybe_sample(df, config.sample_size, config.random_state)
    print(f"  Using {len(df):,} records for training")

    X_df, y_series, feature_cols = build_feature_matrix(df)

    X_temp, X_test_df, y_temp, y_test = train_test_split(
        X_df,
        y_series,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    val_ratio = config.val_size / (1.0 - config.test_size)
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        random_state=config.random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_val = scaler.transform(X_val_df)
    X_test = scaler.transform(X_test_df)

    y_train_log = np.log1p(y_train.to_numpy())
    y_val_log = np.log1p(y_val.to_numpy())
    y_test_log = np.log1p(y_test.to_numpy())

    return X_train, X_val, X_test, y_train_log, y_val_log, y_test_log, scaler, feature_cols


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_rows = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if is_train:
            optimizer.zero_grad()

        preds = model(batch_x)
        loss = criterion(preds, batch_y)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_rows = len(batch_x)
        total_loss += loss.item() * batch_rows
        total_rows += batch_rows

    return total_loss / total_rows


@torch.no_grad()
def evaluate_mae_dollars(
    model: nn.Module,
    X: np.ndarray,
    y_log: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    preds: list[np.ndarray] = []
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for (batch_x,) in loader:
        batch_pred = model(batch_x.to(device)).cpu().numpy()
        preds.append(batch_pred)

    y_pred = np.expm1(np.concatenate(preds))
    y_true = np.expm1(y_log)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, r2


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> HousingPriceNet:
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = HousingPriceNet(
        input_dim=X_train.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )
    criterion = nn.SmoothL1Loss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        val_mae, val_r2 = evaluate_mae_dollars(model, X_val, y_val, device, config.batch_size)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            marker = "*"
        else:
            stale_epochs += 1
            marker = ""

        if epoch == 1 or epoch % 5 == 0 or stale_epochs == 0 or epoch == config.epochs:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:>3}/{config.epochs}{marker} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"val MAE ${val_mae:,.0f} | val R2 {val_r2:.3f} | lr {lr:.1e}"
            )

        if stale_epochs >= config.patience:
            print(f"Early stopping at epoch {epoch} (best val MAE ${best_val_mae:,.0f})")
            break

    model.load_state_dict(best_state)
    return model


def save_artifacts(
    model: HousingPriceNet,
    scaler: StandardScaler,
    feature_cols: list[str],
    config: TrainConfig,
    metrics: dict[str, float],
    model_path: Path,
    artifacts_path: Path,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_cols),
            "hidden_dims": config.hidden_dims,
            "dropout": config.dropout,
            "feature_cols": feature_cols,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "metrics": metrics,
        },
        model_path,
    )

    artifacts = {
        "feature_columns": feature_cols,
        "metrics": metrics,
        "model_path": str(model_path),
        "lrt_radii_m": list(config.lrt_radii_m),
        "school_radii_m": list(config.school_radii_m),
        "rec_radii_m": list(config.rec_radii_m),
        "price_quantiles": [config.price_quantile_low, config.price_quantile_high],
    }
    artifacts_path.write_text(json.dumps(artifacts, indent=2))


def predict_price(
    model: HousingPriceNet,
    scaler: StandardScaler,
    features: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
) -> np.ndarray:
    aligned = features.reindex(columns=feature_cols, fill_value=0.0).astype(float)
    scaled = scaler.transform(aligned)
    model.eval()
    with torch.no_grad():
        preds_log = model(torch.tensor(scaled, dtype=torch.float32).to(device)).cpu().numpy()
    return np.expm1(preds_log)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PyTorch housing price model for Edmonton.")
    parser.add_argument("--sample-size", type=int, default=500_000, help="Rows to train on (use 0 for all data).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--artifacts-path", type=Path, default=DEFAULT_ARTIFACTS_PATH)
    parser.add_argument(
        "--extract-school-data",
        action="store_true",
        help="Extract corrected school coordinates and catchment polygons, then exit.",
    )
    parser.add_argument(
        "--extract-rec-data",
        action="store_true",
        help="Clean recreation facility coordinates and save edmonton_recreation_facilities.csv, then exit.",
    )
    return parser


def extract_school_data_cli() -> None:
    locations, polygons = extract_school_catchment_data()
    print(f"Saved {len(locations)} school locations -> {SCHOOL_LOCATIONS_CSV}")
    print(f"Saved {len(polygons)} catchment polygons -> {SCHOOL_POLYGONS_CSV}")
    print(f"Updated combined file -> {SCHOOL_CSV}")


def extract_rec_data_cli() -> None:
    facilities = extract_recreation_facility_data()
    print(f"Saved {len(facilities)} recreation facilities -> {REC_CSV}")


def main() -> None:
    args = build_parser().parse_args()
    if args.extract_school_data:
        extract_school_data_cli()
        return
    if args.extract_rec_data:
        extract_rec_data_cli()
        return

    sample_size = None if args.sample_size == 0 else args.sample_size

    config = TrainConfig(
        sample_size=sample_size,
        test_size=args.test_size,
        random_state=args.random_state,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    device = get_device()
    print(f"Using device: {device}")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols = prepare_datasets(config)
    print(f"Feature count: {len(feature_cols)}")
    print(f"Train/val/test: {len(y_train):,} / {len(y_val):,} / {len(y_test):,}")

    model = train_model(X_train, y_train, X_val, y_val, config, device)
    mae, r2 = evaluate_mae_dollars(model, X_test, y_test, device, config.batch_size)

    print("\n--- PyTorch Model Results (test set) ---")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"R-squared: {r2:.3f}")

    metrics = {"mae": mae, "r2": r2}
    save_artifacts(
        model,
        scaler,
        feature_cols,
        config,
        metrics,
        args.model_path,
        args.artifacts_path,
    )
    print(f"\nSaved model to {args.model_path}")
    print(f"Saved artifacts to {args.artifacts_path}")


if __name__ == "__main__":
    main()
