"""
PCA analysis module for EmoPairCompete biosignal features.

functions added:
  - fit_pca: fit PCA and return scores, loadings, explained variance
  - pca_scree_data: extract scree plot data
  - pca_top_loadings: extract top contributing features per PC
  - save_pca_results: persist PCA outputs to results/embeddings/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class PCAConfig:
    """configuration for PCA analysis"""
    n_components: Optional[int] = None      # None = keep all
    random_state: int = 42


@dataclass
class PCAResult:
    """container for PCA outputs"""
    scores: np.ndarray                      # (n_samples, n_components)
    loadings: np.ndarray                    # (n_features, n_components)
    explained_variance_ratio: np.ndarray    # (n_components,)
    cumulative_variance: np.ndarray         # (n_components,)
    feature_names: List[str]
    pca_model: PCA                          # fitted sklearn PCA object
    config: PCAConfig


def fit_pca(
    X: np.ndarray,
    feature_names: List[str],
    cfg: Optional[PCAConfig] = None,
) -> PCAResult:
    """
    fit PCA on preprocessed feature matrix

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Preprocessed (scaled) feature matrix.
    feature_names : list of str
        features names (columns).
    cfg : PCAConfig, optional
        configuration. uses defaults if None.

    Returns
    -------
    PCAResult
        Contains scores, loadings, explained variance, etc.
    """
    cfg = cfg or PCAConfig()

    pca = PCA(n_components=cfg.n_components, random_state=cfg.random_state)
    scores = pca.fit_transform(X)
    loadings = pca.components_.T  # shape (n_features, n_components)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    return PCAResult(
        scores=scores,
        loadings=loadings,
        explained_variance_ratio=explained,
        cumulative_variance=cumulative,
        feature_names=feature_names,
        pca_model=pca,
        config=cfg,
    )


def pca_top_loadings(
    result: PCAResult,
    pc_index: int = 0,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get the top contributing features for a given principal component.

    Parameters
    ----------
    result : PCAResult
    pc_index : int
        Which PC (0-indexed).
    top_n : int
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Columns: feature, loading, abs_loading
    """
    loadings_pc = result.loadings[:, pc_index]

    df = pd.DataFrame({
        "feature": result.feature_names,
        "loading": loadings_pc,
        "abs_loading": np.abs(loadings_pc),
    })

    df = df.sort_values("abs_loading", ascending=False).head(top_n)
    df = df.reset_index(drop=True)

    return df


def pca_scree_data(result: PCAResult, max_components: int = 20) -> pd.DataFrame:
    """
    Extract scree plot data.

    Returns
    -------
    pd.DataFrame
        Columns: PC, explained_variance, cumulative_variance
    """
    n = min(max_components, len(result.explained_variance_ratio))

    return pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(n)],
        "PC_number": np.arange(1, n + 1),
        "explained_variance": result.explained_variance_ratio[:n],
        "cumulative_variance": result.cumulative_variance[:n],
    })


def save_pca_results(
    result: PCAResult,
    metadata: pd.DataFrame,
    output_dir: Path,
    prefix: str = "pca",
) -> dict:
    """
    Save PCA scores, loadings, and variance info to disk.

    Parameters
    ----------
    result : PCAResult
    metadata : pd.DataFrame
        Aligned metadata for the observations.
    output_dir : Path
        Directory to save into (e.g., results/embeddings/).
    prefix : str
        File name prefix.

    Returns
    -------
    dict of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # scores with metadata
    scores_df = pd.DataFrame(
        result.scores,
        columns=[f"PC{i+1}" for i in range(result.scores.shape[1])],
    )
    scores_df = pd.concat([metadata.reset_index(drop=True), scores_df], axis=1)
    scores_path = output_dir / f"{prefix}_scores.csv"
    scores_df.to_csv(scores_path, index=False)

    # loadings
    loadings_df = pd.DataFrame(
        result.loadings,
        index=result.feature_names,
        columns=[f"PC{i+1}" for i in range(result.loadings.shape[1])],
    )
    loadings_path = output_dir / f"{prefix}_loadings.csv"
    loadings_df.to_csv(loadings_path)

    # vaiance inforation
    scree_df = pca_scree_data(result)
    scree_path = output_dir / f"{prefix}_variance.csv"
    scree_df.to_csv(scree_path, index=False)

    return {
        "scores": scores_path,
        "loadings": loadings_path,
        "variance": scree_path,
    }