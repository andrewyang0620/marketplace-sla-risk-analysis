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


def time_split_and_trim(
    df: pd.DataFrame,
    date_col: str = "date",
    train_frac: float = 0.70,
    max_horizon_days: int = 21,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Split a time-indexed panel into chronological train and test sets,
    then trim the tail of each split by `max_horizon_days`
    to avoid label leakage.
    """
    out = df.copy()
    out = out.sort_values(date_col)

    all_dates = np.sort(out[date_col].unique())
    split_idx = int(len(all_dates) * train_frac)
    train_end_date = all_dates[split_idx]

    train_df = out[out[date_col] <= train_end_date].copy()
    test_df = out[out[date_col] > train_end_date].copy()

    train_end_valid = train_df[date_col].max() - pd.Timedelta(days=max_horizon_days)
    test_end_valid = test_df[date_col].max() - pd.Timedelta(days=max_horizon_days)

    train_df = train_df[train_df[date_col] <= train_end_valid].copy()
    test_df = test_df[test_df[date_col] <= test_end_valid].copy()

    return train_df, test_df, train_end_date


def derive_h2_harm_coefficients(
    dose_summary: pd.DataFrame,
    reference_bucket: str = "on_time_or_early",
    moderate_bucket: str = "3-5_days_late",
    severe_bucket: str = "6+_days_late",
) -> pd.DataFrame:
    """
    Derive customer-harm coefficients from the H2 dose-response table.

    The function compares each target bucket against the reference bucket
    (typically on-time or early delivery) and computes:
      - delta_review_loss
      - delta_low_rating
      - delta_cancel_rate
      - delta_repeat_rate

    Parameters
    ----------
    dose_summary : pd.DataFrame
        Output of H2 dose-response analysis. Must contain:
          - delay_bucket
          - mean_review
          - low_rating_rate
          - cancel_rate
          - repeat_rate
    reference_bucket : str, default "on_time_or_early"
        Reference bucket used as the no-harm baseline.
    moderate_bucket : str, default "3-5_days_late"
        Moderate harm proxy.
    severe_bucket : str, default "6+_days_late"
        Severe harm proxy; this is the default bridge to H3 severe-event labels.

    Returns
    -------
    pd.DataFrame
        A small table with rows for moderate and severe buckets and columns:
          - bucket_type
          - bucket
          - reference_bucket
          - delta_review_loss
          - delta_low_rating
          - delta_cancel_rate
          - delta_repeat_rate
          - ref_* and bucket_* columns for auditability
    """
    required_cols = [
        "delay_bucket",
        "mean_review",
        "low_rating_rate",
        "cancel_rate",
        "repeat_rate",
    ]
    missing = [c for c in required_cols if c not in dose_summary.columns]
    if missing:
        raise ValueError(f"dose_summary is missing columns: {missing}")

    ref = dose_summary.loc[dose_summary["delay_bucket"] == reference_bucket]
    if ref.empty:
        raise ValueError(f"Reference bucket '{reference_bucket}' not found.")
    ref = ref.iloc[0]

    rows = []
    for bucket_type, bucket_name in [
        ("moderate", moderate_bucket),
        ("severe", severe_bucket),
    ]:
        cur = dose_summary.loc[dose_summary["delay_bucket"] == bucket_name]
        if cur.empty:
            raise ValueError(f"Bucket '{bucket_name}' not found.")
        cur = cur.iloc[0]

        rows.append(
            {
                "bucket_type": bucket_type,
                "bucket": bucket_name,
                "reference_bucket": reference_bucket,
                "ref_mean_review": ref["mean_review"],
                "ref_low_rating_rate": ref["low_rating_rate"],
                "ref_cancel_rate": ref["cancel_rate"],
                "ref_repeat_rate": ref["repeat_rate"],
                "bucket_mean_review": cur["mean_review"],
                "bucket_low_rating_rate": cur["low_rating_rate"],
                "bucket_cancel_rate": cur["cancel_rate"],
                "bucket_repeat_rate": cur["repeat_rate"],
                "delta_review_loss": ref["mean_review"] - cur["mean_review"],
                "delta_low_rating": cur["low_rating_rate"] - ref["low_rating_rate"],
                "delta_cancel_rate": cur["cancel_rate"] - ref["cancel_rate"],
                "delta_repeat_rate": ref["repeat_rate"] - cur["repeat_rate"],
            }
        )

    return pd.DataFrame(rows)