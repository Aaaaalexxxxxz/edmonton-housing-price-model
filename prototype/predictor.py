"""Load trained model and predict price for a single Edmonton property."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from housing_price_pytorch import (  # noqa: E402
    DEFAULT_ARTIFACTS_PATH,
    DEFAULT_MODEL_PATH,
    HousingPriceNet,
    add_landmark_distances,
    add_lrt_features,
    add_recreation_features,
    add_school_features,
    build_feature_matrix,
    get_device,
    load_lrt_stations,
    load_recreation_facilities,
    load_school_data,
    predict_price,
    project_to_utm,
)


@dataclass
class PredictionResult:
    predicted_price: float
    latitude: float
    longitude: float
    spatial_features: dict[str, float | int]


class PropertyPricePredictor:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        artifacts_path: Path = DEFAULT_ARTIFACTS_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        artifacts = json.loads(artifacts_path.read_text())

        self.feature_cols: list[str] = checkpoint["feature_cols"]
        self.known_zones = {
            col.replace("Zoning_", "")
            for col in self.feature_cols
            if col.startswith("Zoning_") and col != "Zoning_OTHER"
        }
        self.lrt_radii_m = tuple(artifacts.get("lrt_radii_m", (500, 1000, 2000)))
        self.school_radii_m = tuple(artifacts.get("school_radii_m", (500, 1000, 2000)))
        self.rec_radii_m = tuple(artifacts.get("rec_radii_m", (500, 1000, 2000)))
        self.metrics = artifacts.get("metrics", {})

        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(checkpoint["scaler_mean"], dtype=float)
        self.scaler.scale_ = np.array(checkpoint["scaler_scale"], dtype=float)
        self.scaler.n_features_in_ = len(self.feature_cols)

        self.device = get_device()
        self.model = HousingPriceNet(
            input_dim=checkpoint["input_dim"],
            hidden_dims=tuple(checkpoint["hidden_dims"]),
            dropout=checkpoint["dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self.lrt_df = load_lrt_stations()
        self.school_locations, self.school_polygons = load_school_data()
        self.rec_facilities = load_recreation_facilities()

    def _normalize_zoning(self, zoning: str) -> str:
        zoning = zoning.strip().upper()
        if zoning not in self.known_zones:
            return "OTHER"
        return zoning

    def _build_property_frame(
        self,
        latitude: float,
        longitude: float,
        year_built: int,
        lot_size: float,
        garage: str,
        zoning: str,
        assessment_year: int,
    ) -> pd.DataFrame:
        garage = garage.strip().upper()
        if garage not in {"Y", "N"}:
            raise ValueError("Garage must be 'Y' or 'N'")

        df = pd.DataFrame(
            [
                {
                    "Latitude": latitude,
                    "Longitude": longitude,
                    "Actual Year Built": year_built,
                    "Lot Size": lot_size,
                    "Garage": garage,
                    "Zoning": self._normalize_zoning(zoning),
                    "Assessment Year": assessment_year,
                    "Assessed Value": 0.0,
                }
            ]
        )

        df = add_lrt_features(df, self.lrt_df, radii_m=self.lrt_radii_m)
        prop_x, prop_y = project_to_utm(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())
        df = add_landmark_distances(df, prop_x, prop_y)
        df = add_school_features(
            df,
            self.school_locations,
            self.school_polygons,
            radii_m=self.school_radii_m,
        )
        df = add_recreation_features(df, self.rec_facilities, radii_m=self.rec_radii_m)

        distance_cols = [c for c in df.columns if c.startswith("Distance_to") or c.startswith("Log_Distance")]
        for col in distance_cols:
            df[col] = df[col].fillna(10_000.0)

        return df

    def predict(
        self,
        latitude: float,
        longitude: float,
        year_built: int,
        lot_size: float,
        garage: str,
        zoning: str,
        assessment_year: int = 2023,
    ) -> PredictionResult:
        df = self._build_property_frame(
            latitude=latitude,
            longitude=longitude,
            year_built=year_built,
            lot_size=lot_size,
            garage=garage,
            zoning=zoning,
            assessment_year=assessment_year,
        )
        features, _, _ = build_feature_matrix(df)
        price = float(predict_price(self.model, self.scaler, features, self.feature_cols, self.device)[0])

        row = df.iloc[0]
        spatial = {
            "distance_to_lrt_m": round(float(row["Distance_to_LRT_m"]), 0),
            "distance_to_uofa_m": round(float(row["Distance_to_UofA_m"]), 0),
            "distance_to_west_edmonton_mall_m": round(float(row["Distance_to_West_Edmonton_Mall_m"]), 0),
            "lrt_stations_within_1000m": int(row["LRT_stations_within_1000m"]),
            "in_elementary_catchment": int(row["In_Elementary_Catchment"]),
            "in_junior_catchment": int(row["In_Junior_Catchment"]),
            "in_senior_catchment": int(row["In_Senior_Catchment"]),
            "rec_facilities_within_1000m": int(row["Rec_Facilitys_within_1000m"]),
        }

        return PredictionResult(
            predicted_price=price,
            latitude=latitude,
            longitude=longitude,
            spatial_features=spatial,
        )
