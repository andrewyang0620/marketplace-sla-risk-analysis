from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd


def build_feature_list(
    windows: Sequence[int] = (7, 14, 30),
    include_trends: bool = True,
) -> List[str]:
    """
    Return the feature list used in 03_early_warning.ipynb so that
    04_intervention_roi.ipynb can reproduce the deployment score consistently.

    Parameters
    ----------
    windows : sequence of int, default (7, 14, 30)
        Rolling windows used in early-warning feature engineering.
    include_trends : bool, default True
        Whether to include the 7-vs-30 trend features.

    Returns
    -------
    list of str
        Feature names expected to exist in the early warning panel.
    """
    feature_cols = [
        "seller_tenure_days",
        "cum_delivered_orders",
        "cum_delivered_gmv",
        "lifetime_violation_rate",
        "lifetime_severe_violation_rate",
        "lifetime_violation_gmv_share",
        "lifetime_severe_violation_gmv_share",
        "violation_rate",
        "severe_violation_rate",
        "violation_gmv_share",
        "severe_violation_gmv_share",
        "avg_delay_days",
    ]

    for w in windows:
        feature_cols.extend(
            [
                f"delivered_{w}d",
                f"delivered_gmv_{w}d",
                f"violation_rate_{w}d",
                f"severe_violation_rate_{w}d",
                f"violation_gmv_share_{w}d",
                f"severe_violation_gmv_share_{w}d",
                f"avg_delay_{w}d",
            ]
        )

    if include_trends:
        feature_cols.extend(
            [
                "violation_rate_trend_7v30",
                "severe_rate_trend_7v30",
                "gmv_share_trend_7v30",
                "delay_trend_7v30",
            ]
        )

    return feature_cols


def ensure_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the trend features used in 03 exist in the early warning panel.

    If they are missing, they are created as:
      - violation_rate_trend_7v30 = violation_rate_7d - violation_rate_30d
      - severe_rate_trend_7v30 = severe_violation_rate_7d - severe_violation_rate_30d
      - gmv_share_trend_7v30 = severe_violation_gmv_share_7d - severe_violation_gmv_share_30d
      - delay_trend_7v30 = avg_delay_7d - avg_delay_30d

    Parameters
    ----------
    df : pd.DataFrame
        Early warning panel.

    Returns
    -------
    pd.DataFrame
        DataFrame with the required trend features present.
    """
    out = df.copy()

    if "violation_rate_trend_7v30" not in out.columns:
        out["violation_rate_trend_7v30"] = (
            out["violation_rate_7d"] - out["violation_rate_30d"]
        )

    if "severe_rate_trend_7v30" not in out.columns:
        out["severe_rate_trend_7v30"] = (
            out["severe_violation_rate_7d"] - out["severe_violation_rate_30d"]
        )

    if "gmv_share_trend_7v30" not in out.columns:
        out["gmv_share_trend_7v30"] = (
            out["severe_violation_gmv_share_7d"] - out["severe_violation_gmv_share_30d"]
        )

    if "delay_trend_7v30" not in out.columns:
        out["delay_trend_7v30"] = out["avg_delay_7d"] - out["avg_delay_30d"]

    return out
