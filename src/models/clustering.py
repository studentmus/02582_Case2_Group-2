"""
Clustering module for EmoPairCompete biosignal features

functions added:
  - fit_kmeans: run k-means with multiple K values
  - fit_gmm: run Gaussian Mixture Models with multiple K values
  - evaluate_clustering: compute silhouette, ARI, NMI against true labels
  - select_best_k: find optimal K using silhouette or BIC
  - save_clustering_results: persist outputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


@dataclass
class ClusteringConfig:
    """configuration for clustering experiments"""
    k_range: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    n_init: int = 50
    random_state: int = 42
    gmm_covariance_types: List[str] = field(
        default_factory=lambda: ["full", "diag"]
    )


@dataclass
class ClusteringResult:
    """container for a single clustering run"""
    method: str                     # "kmeans" or "gmm"
    k: int                          # number of clusters
    labels: np.ndarray              # (n_samples,) cluster assignments
    silhouette: float               # average silhouette score
    silhouette_samples: np.ndarray  # per-sample silhouette values
    inertia: Optional[float]        # k-means inertia (None for GMM)
    bic: Optional[float]            # GMM BIC (None for k-means)
    aic: Optional[float]            # GMM AIC (None for k-means)
    extra: Dict                     # any additional info


@dataclass
class ClusterEvaluation:
    """evaluation of clustering against known labels"""
    method: str
    k: int
    silhouette: float
    ari: float                      # Adjusted Rand Index
    nmi: float                      # Normalized Mutual Information
    contingency: pd.DataFrame       # crosstab of clusters vs true labels


def fit_kmeans(
    X: np.ndarray,
    cfg: Optional[ClusteringConfig] = None,
) -> List[ClusteringResult]:
    """
    run k-means for each K in k_range.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    cfg : ClusteringConfig

    Returns
    -------
    list of ClusteringResult, one per K.
    """
    cfg = cfg or ClusteringConfig()
    results = []

    for k in cfg.k_range:
        km = KMeans(
            n_clusters=k,
            n_init=cfg.n_init,
            random_state=cfg.random_state,
        )
        labels = km.fit_predict(X)
        sil_samples = silhouette_samples(X, labels)
        sil_avg = float(np.mean(sil_samples))

        results.append(ClusteringResult(
            method="kmeans",
            k=k,
            labels=labels,
            silhouette=sil_avg,
            silhouette_samples=sil_samples,
            inertia=float(km.inertia_),
            bic=None,
            aic=None,
            extra={"centroids": km.cluster_centers_},
        ))

    return results


def fit_gmm(
    X: np.ndarray,
    cfg: Optional[ClusteringConfig] = None,
    covariance_type: str = "full",
) -> List[ClusteringResult]:
    """
    run GMM for each K in k_range

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    cfg : ClusteringConfig
    covariance_type : str
        "full", "diag", "tied", or "spherical"

    Returns
    -------
    list of ClusteringResult, one per K.
    """
    cfg = cfg or ClusteringConfig()
    results = []

    for k in cfg.k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=cfg.n_init,
            random_state=cfg.random_state,
        )
        labels = gmm.fit_predict(X)
        sil_samples_arr = silhouette_samples(X, labels)
        sil_avg = float(np.mean(sil_samples_arr))

        results.append(ClusteringResult(
            method=f"gmm_{covariance_type}",
            k=k,
            labels=labels,
            silhouette=sil_avg,
            silhouette_samples=sil_samples_arr,
            inertia=None,
            bic=float(gmm.bic(X)),
            aic=float(gmm.aic(X)),
            extra={
                "covariance_type": covariance_type,
                "means": gmm.means_,
                "converged": gmm.converged_,
            },
        ))

    return results


def evaluate_clustering(
    result: ClusteringResult,
    true_labels: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> ClusterEvaluation:
    """
    evaluate a clustering result against known true labels.

    Parameters
    ----------
    result : ClusteringResult
    true_labels : np.ndarray
        ground truth labels (e.g., phase).
    label_names : list of str, optional
        display names for true labels.

    Returns
    -------
    ClusterEvaluation
    """
    ari = adjusted_rand_score(true_labels, result.labels)
    nmi = normalized_mutual_info_score(true_labels, result.labels)

    # Contingency table
    if label_names is not None:
        ct = pd.crosstab(
            pd.Series(result.labels, name="Cluster"),
            pd.Series(true_labels, name="True_Label").map(
                {i: n for i, n in enumerate(label_names)}
                if isinstance(true_labels[0], (int, np.integer))
                else {n: n for n in label_names}
            ),
        )
    else:
        ct = pd.crosstab(
            pd.Series(result.labels, name="Cluster"),
            pd.Series(true_labels, name="True_Label"),
        )

    return ClusterEvaluation(
        method=result.method,
        k=result.k,
        silhouette=result.silhouette,
        ari=ari,
        nmi=nmi,
        contingency=ct,
    )


def select_best_k(
    results: List[ClusteringResult],
    criterion: str = "silhouette",
) -> ClusteringResult:
    """
    Select the best K from a list of results.

    Parameters
    ----------
    results : list of ClusteringResult
    criterion : str
        "silhouette" (maximize) or "bic" (minimize).

    Returns
    -------
    ClusteringResult with optimal K.
    """
    if criterion == "silhouette":
        return max(results, key=lambda r: r.silhouette)
    elif criterion == "bic":
        return min(results, key=lambda r: r.bic)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def save_clustering_results(
    results: List[ClusteringResult],
    evaluations: List[ClusterEvaluation],
    metadata: pd.DataFrame,
    output_dir: Path,
    prefix: str = "clustering",
) -> Dict[str, Path]:
    """
    save clustering results to disk
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # saving summary table
    summary_rows = []
    for r, e in zip(results, evaluations):
        row = {
            "method": r.method,
            "k": r.k,
            "silhouette": r.silhouette,
            "ari": e.ari,
            "nmi": e.nmi,
        }
        if r.inertia is not None:
            row["inertia"] = r.inertia
        if r.bic is not None:
            row["bic"] = r.bic
            row["aic"] = r.aic
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"{prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # saving best cluster labels with metadata
    best = select_best_k(results, criterion="silhouette")
    labels_df = metadata.copy()
    labels_df["cluster"] = best.labels
    labels_path = output_dir / f"{prefix}_best_labels.csv"
    labels_df.to_csv(labels_path, index=False)

    return {
        "summary": summary_path,
        "labels": labels_path,
    }