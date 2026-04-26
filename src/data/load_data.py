from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Literal, Optional

import pandas as pd

from src.config import DataConfig


DatasetLevel = Literal["features", "time_series", "both"]


@dataclass
class LoadedFeatures:
    features: pd.DataFrame
    metadata: pd.DataFrame
    raw_df: pd.DataFrame
    source_path: Path


def _find_hrdata2_file(cfg: DataConfig) -> Path:
    """Locate the HRdata2 CSV file under data/raw/HRdata2/."""
    hr_dir = cfg.raw_dir / cfg.hrdata2_subdir
    if not hr_dir.exists():
        raise FileNotFoundError(f"HRdata2 directory not found: {hr_dir}")

    matches = sorted(hr_dir.glob(cfg.hrdata2_pattern))
    if not matches:
        raise FileNotFoundError(
            f"No CSV files matching pattern '{cfg.hrdata2_pattern}' in {hr_dir}"
        )
    if len(matches) > 1:
        # Later you can extend this to merge multiple files if needed
        print(f"[load_data] Multiple HRdata2 files found, using first: {matches[0].name}")
    return matches[0]


def load_feature_data(cfg: Optional[DataConfig] = None) -> LoadedFeatures:
    """
    Load aggregated feature data from HRdata2.

    Returns:
        LoadedFeatures with:
          - features: only feature columns (HR_TD_*, TEMP_TD_*, EDA_TD_*, etc.)
          - metadata: columns describing subject/phase/IDs and questionnaires
          - raw_df: original DataFrame with all columns (except dropped index)
          - source_path: path to the loaded CSV
    """
    cfg = cfg or DataConfig()

    csv_path = _find_hrdata2_file(cfg)
    df = pd.read_csv(csv_path)

    # Drop the auto index column if present
    if cfg.index_col in df.columns:
        df = df.drop(columns=[cfg.index_col])

    # Split metadata vs feature columns
    metadata_cols = cfg.metadata_columns()
    metadata_cols_present = [c for c in metadata_cols if c in df.columns]

    metadata_df = df[metadata_cols_present].copy()

    feature_cols = [c for c in df.columns if c not in metadata_cols_present]
    features_df = df[feature_cols].copy()

    return LoadedFeatures(
        features=features_df,
        metadata=metadata_df,
        raw_df=df.copy(),
        source_path=csv_path,
    )


def load_time_series_data(
    cfg: Optional[DataConfig] = None,
):
    """
    Placeholder for loading raw time-series data from emopaircompete_raw.

    Later you can implement:
      - walking through emopaircompete_raw/
      - reading biosignal.csv and response.csv
      - returning a structure keyed by (subject, round, phase) or similar.

    For now this function raises NotImplementedError to keep the API explicit.
    """
    cfg = cfg or DataConfig()
    raw_ts_dir = cfg.raw_dir / "emopaircompete_raw"

    if not raw_ts_dir.exists():
        raise FileNotFoundError(f"Time-series directory not found: {raw_ts_dir}")

    raise NotImplementedError(
        "Time-series loading is not implemented yet. "
        "Implement when you decide how to represent the sequences."
    )


def load_dataset(
    level: DatasetLevel = "features",
    cfg: Optional[DataConfig] = None,
):
    """
    High-level dataset loader used by pipelines.

    Args:
        level: "features" | "time_series" | "both"
        cfg: DataConfig override

    Returns:
        If level == "features": LoadedFeatures
        If level == "time_series": NotImplemented (for now)
        If level == "both": (LoadedFeatures, <time_series_struct>)
    """
    cfg = cfg or DataConfig()

    if level == "features":
        return load_feature_data(cfg)

    if level == "time_series":
        ts_data = load_time_series_data(cfg)
        return ts_data

    if level == "both":
        features = load_feature_data(cfg)
        ts_data = load_time_series_data(cfg)
        return features, ts_data

    raise ValueError(f"Unknown level: {level}")
