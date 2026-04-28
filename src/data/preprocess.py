from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.config import DataConfig
from src.data.load_data import LoadedFeatures


@dataclass
class PreprocessConfig:
  imputation_strategy: str = "median"   # "median" | "mean" | "drop_rows"
  scale_features: bool = True
  scaler: str = "standard"             # "standard" | "minmax"
  subject_center: bool = False          # added subject centering flag

  target_type: str = "phase"           # "phase" | "frustration_binary" | "none"
  phase_mapping: Dict[str, int] = None
  frustration_threshold: float = 5.0
  drop_panas_from_features: bool = True

  def __post_init__(self):
      if self.phase_mapping is None:
          # Adjusted to actual Phase values from HRdata2
          self.phase_mapping = {
              "phase1": 0,
              "phase2": 1,
              "phase3": 2,
          }


def load_preprocess_config_from_yaml(path: Path) -> PreprocessConfig:
  """
  Optional helper if you want to read preprocessing config from YAML later.
  For now you can instantiate PreprocessConfig directly in notebooks.
  """
  import yaml

  with open(path, "r") as f:
      cfg_dict = yaml.safe_load(f)

  pp = cfg_dict.get("preprocessing", {})
  return PreprocessConfig(
      imputation_strategy=pp.get("imputation_strategy", "median"),
      scale_features=pp.get("scale_features", True),
      scaler=pp.get("scaler", "standard"),
      target_type=pp.get("target", {}).get("type", "phase"),
      phase_mapping=pp.get("target", {}).get("phase_mapping", None),
      frustration_threshold=pp.get("target", {}).get("frustration_threshold", 5.0),
      drop_panas_from_features=pp.get("drop_panas_from_features", True),
  )


@dataclass
class PreprocessResult:
  X: np.ndarray
  y: Optional[np.ndarray]
  feature_names: list
  metadata: pd.DataFrame
  info: Dict


def _impute_features(
  features_df: pd.DataFrame,
  strategy: str = "median",
) -> pd.DataFrame:
  if strategy == "drop_rows":
      return features_df.dropna(axis=0)

  if strategy not in {"median", "mean"}:
      raise ValueError(f"Unknown imputation strategy: {strategy}")

  if strategy == "median":
      fill_values = features_df.median()
  else:
      fill_values = features_df.mean()

  return features_df.fillna(fill_values)


def _scale_features(
  X: np.ndarray,
  scaler_name: str = "standard",
) -> Tuple[np.ndarray, object]:
  if scaler_name == "standard":
      scaler = StandardScaler()
  elif scaler_name == "minmax":
      scaler = MinMaxScaler()
  else:
      raise ValueError(f"Unknown scaler: {scaler_name}")

  X_scaled = scaler.fit_transform(X)
  return X_scaled, scaler


def _build_target(
  loaded: LoadedFeatures,
  cfg: DataConfig,
  pp_cfg: PreprocessConfig,
) -> Optional[np.ndarray]:
  meta = loaded.metadata

  if pp_cfg.target_type == "none":
      return None

  if pp_cfg.target_type == "phase":
      if cfg.phase_col not in meta.columns:
          raise KeyError(f"Phase column '{cfg.phase_col}' not in metadata")

      phases = meta[cfg.phase_col].astype(str)
      mapping = pp_cfg.phase_mapping
      y = phases.map(mapping)

      if y.isna().any():
          unknown = sorted(phases[y.isna()].unique())
          raise ValueError(
              f"Some Phase values not in mapping: {unknown}. "
              f"Current mapping: {mapping}"
          )
      return y.to_numpy(dtype=int)

  if pp_cfg.target_type == "frustration_binary":
      if cfg.frustration_col not in meta.columns:
          raise KeyError(f"Frustration column '{cfg.frustration_col}' not in metadata")

      vals = meta[cfg.frustration_col].astype(float)
      y = (vals >= pp_cfg.frustration_threshold).astype(int)
      return y.to_numpy(dtype=int)

  raise ValueError(f"Unknown target_type: {pp_cfg.target_type}")

def _subject_center(
    features_df: pd.DataFrame,
    individual_col: pd.Series,
) -> pd.DataFrame:
    """
    for each individual, subtracting their mean feature vector
    this should remove inter-subject baseline differences
    """
    df = features_df.copy()
    df["_individual"] = individual_col.values
    centered = df.groupby("_individual").transform(lambda x: x - x.mean())
    return centered


def preprocess_features(
  loaded: LoadedFeatures,
  data_cfg: Optional[DataConfig] = None,
  pp_cfg: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
  """
  Main preprocessing entry-point for HRdata2 features.

  Steps:
    1) Optionally drop PANAS columns from feature matrix
    2) Impute missing values in features
    3) Optionally scale features
    4) Build target y (phase or frustration-based) if required

  Returns:
    PreprocessResult with X (np.ndarray), y (np.ndarray or None),
    feature_names, metadata (unchanged), and info dict.
  """
  data_cfg = data_cfg or DataConfig()
  pp_cfg = pp_cfg or PreprocessConfig()

  features_df = loaded.features.copy()
  metadata_df = loaded.metadata.copy()

  # Optionally remove PANAS from feature set, keep them only in metadata
  if pp_cfg.drop_panas_from_features:
      panas_cols_present = [c for c in data_cfg.panas_cols if c in features_df.columns]
      if panas_cols_present:
          features_df = features_df.drop(columns=panas_cols_present)

  # 1) Imputation
  features_imputed = _impute_features(features_df, strategy=pp_cfg.imputation_strategy)

  # 2) Subject centering
  if pp_cfg.subject_center:
    individual_col = metadata_df[data_cfg.individual_col]
    features_imputed = _subject_center(features_imputed, individual_col)

  # 2) Scaling
  X = features_imputed.to_numpy(dtype=float)
  scaler_obj = None
  if pp_cfg.scale_features:
      X, scaler_obj = _scale_features(X, scaler_name=pp_cfg.scaler)

  # 3) Target
  y = _build_target(loaded, data_cfg, pp_cfg)

  info = {
      "imputation_strategy": pp_cfg.imputation_strategy,
      "scale_features": pp_cfg.scale_features,
      "scaler": pp_cfg.scaler,
      "target_type": pp_cfg.target_type,
      "phase_mapping": pp_cfg.phase_mapping,
      "frustration_threshold": pp_cfg.frustration_threshold,
      "drop_panas_from_features": pp_cfg.drop_panas_from_features,
      "scaler_object": scaler_obj,
  }

  return PreprocessResult(
      X=X,
      y=y,
      feature_names=list(features_imputed.columns),
      metadata=metadata_df,
      info=info,
  )


def save_processed_features(
    result: PreprocessResult,
    data_cfg: Optional[DataConfig] = None,
    prefix: str = "hrdata2_phase",
) -> Dict[str, Path]:
    """
    Save processed features, targets and metadata to data/processed/.

    Creates:
      - {prefix}_X.npy
      - {prefix}_y.npy   (if y is not None)
      - {prefix}_metadata.csv
      - {prefix}_info.json

    Returns:
      dict with created file paths.
    """
    import json

    data_cfg = data_cfg or DataConfig()
    out_dir = data_cfg.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X_path = out_dir / f"{prefix}_X.npy"
    y_path = out_dir / f"{prefix}_y.npy"
    meta_path = out_dir / f"{prefix}_metadata.csv"
    info_path = out_dir / f"{prefix}_info.json"

    # Save X
    np.save(X_path, result.X)

    # Save y if present
    if result.y is not None:
        np.save(y_path, result.y)

    # Save metadata
    result.metadata.to_csv(meta_path, index=False)

    # Save info (remove non-serializable objects, e.g. scaler)
    info_serializable = dict(result.info)
    if "scaler_object" in info_serializable:
        scaler = info_serializable.pop("scaler_object")
        if scaler is not None:
            info_serializable["scaler_class"] = scaler.__class__.__name__

    with open(info_path, "w") as f:
        json.dump(info_serializable, f, indent=2)

    return {
        "X_path": X_path,
        "y_path": y_path if result.y is not None else None,
        "metadata_path": meta_path,
        "info_path": info_path,
    }
