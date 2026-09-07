"""
Data drift detection module for CreditRiskML.
Calculates Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from app.config import settings

logger = logging.getLogger("credit_risk_ml.drift")

def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-4
) -> float:
    """
    Computes Population Stability Index (PSI) between reference and current feature distributions.
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    """
    # Remove NaN values
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Determine bin edges from reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges = np.unique(bin_edges)  # In case of duplicate quantiles

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (len(cur) + eps * len(cur_counts))

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(np.round(psi_val, 4))

def calculate_categorical_psi(
    reference: pd.Series,
    current: pd.Series,
    eps: float = 1e-4
) -> float:
    """Computes PSI for categorical features based on category frequency shares."""
    categories = list(set(reference.dropna().unique()).union(set(current.dropna().unique())))
    if not categories:
        return 0.0

    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    ref_total = len(reference.dropna())
    cur_total = len(current.dropna())

    if ref_total == 0 or cur_total == 0:
        return 0.0

    psi_val = 0.0
    for cat in categories:
        ref_pct = (ref_counts.get(cat, 0) + eps) / (ref_total + eps * len(categories))
        cur_pct = (cur_counts.get(cat, 0) + eps) / (cur_total + eps * len(categories))
        psi_val += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

    return float(np.round(psi_val, 4))

def load_reference_dataset() -> pd.DataFrame:
    """Loads baseline training distribution from data/reference."""
    ref_path = Path(settings.DATA_DIR) / "reference" / "baseline_features.parquet"
    if not ref_path.exists():
        raw_path = Path(settings.DATA_DIR) / "raw" / "german_credit.csv"
        if raw_path.exists():
            return pd.read_csv(raw_path)
        raise FileNotFoundError(f"Reference dataset not found at {ref_path}")
    return pd.read_parquet(ref_path)

def compute_dataset_drift(
    current_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Computes drift metrics (PSI and KS) across all features between reference and current data.
    """
    if reference_df is None:
        reference_df = load_reference_dataset()

    features_report = []
    total_features = 0
    drifted_features_count = 0

    for col in reference_df.columns:
        if col == "credit_risk" or col not in current_df.columns:
            continue

        total_features += 1
        ref_col = reference_df[col]
        cur_col = current_df[col]

        is_numeric = pd.api.types.is_numeric_dtype(ref_col)

        if is_numeric:
            psi_score = calculate_psi(ref_col.values, cur_col.values)
            # Two-sample KS test
            ks_res = ks_2samp(ref_col.dropna(), cur_col.dropna())
            ks_stat = float(np.round(ks_res.statistic, 4))
            ks_pval = float(np.round(ks_res.pvalue, 4))
            is_drifted = (psi_score >= settings.DRIFT_PSI_THRESHOLD) or (ks_pval < settings.DRIFT_KS_PVALUE_THRESHOLD)
        else:
            psi_score = calculate_categorical_psi(ref_col, cur_col)
            ks_stat = None
            ks_pval = None
            is_drifted = (psi_score >= settings.DRIFT_PSI_THRESHOLD)

        if is_drifted:
            drifted_features_count += 1

        # Drift severity interpretation
        if psi_score < 0.10:
            status = "stable"
        elif psi_score < 0.25:
            status = "moderate"
        else:
            status = "significant"

        features_report.append({
            "feature": col,
            "type": "numeric" if is_numeric else "categorical",
            "psi": psi_score,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "status": status,
            "drift_detected": is_drifted
        })

    drift_share = (drifted_features_count / max(total_features, 1))
    dataset_drift_detected = drift_share >= 0.20  # Overall drift if >= 20% features drift

    return {
        "dataset_drift_detected": dataset_drift_detected,
        "drift_share": round(drift_share, 4),
        "total_features": total_features,
        "drifted_features_count": drifted_features_count,
        "psi_threshold": settings.DRIFT_PSI_THRESHOLD,
        "ks_pvalue_threshold": settings.DRIFT_KS_PVALUE_THRESHOLD,
        "features": features_report
    }
