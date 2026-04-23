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


def compute_no_intervention_baseline(
    scored_df: pd.DataFrame,
    future_event_count_col: str,
    future_event_gmv_col: str,
    severe_harm_row: Dict,
    assumption_profiles: Dict[str, Dict],
    current_gmv_proxy_col: str = "delivered_gmv_14d",
    gmv_window_days: int = 14,
) -> pd.DataFrame:
    """
    Compute the explicit no-intervention baseline for each assumption profile.

    This formalises the benchmark that H4 scenarios are compared against.

    Economic proxy design
    ---------------------
    - Compensation cost proxy:
        future severe-event GMV × compensation_rate_on_prevented_gmv
    - Reputation cost proxy:
        incremental low ratings × cost_per_incremental_low_rating_proxy_brl
    - Review-score loss:
        kept as a non-monetised secondary KPI

    Important note on GMV proxy
    ---------------------------
    `current_gmv_proxy_col` is usually a rolling 14-day GMV proxy (e.g. delivered_gmv_14d).
    To align it with seller-day decisions, the no-intervention baseline stores a
    daily-aligned proxy:
        current_gmv_proxy_brl = sum(current_gmv_proxy_col / gmv_window_days)

    Parameters
    ----------
    scored_df : pd.DataFrame
        Deployment window with future event columns.
    future_event_count_col : str
        Future severe-event count column for the chosen horizon.
    future_event_gmv_col : str
        Future severe-event GMV column for the chosen horizon.
    severe_harm_row : dict
        Severe-harm coefficients from H2.
    assumption_profiles : dict
        Output of `build_assumption_profiles()`.
    current_gmv_proxy_col : str, default 'delivered_gmv_14d'
        Current-period rolling GMV exposure proxy.
    gmv_window_days : int, default 14
        Number of days represented by the rolling GMV proxy.

    Returns
    -------
    pd.DataFrame
        One row per assumption profile.
    """
    total_future_events = scored_df[future_event_count_col].sum()
    total_future_gmv = scored_df[future_event_gmv_col].sum()

    if current_gmv_proxy_col not in scored_df.columns:
        raise ValueError(f"Missing column: {current_gmv_proxy_col}")

    total_current_gmv_proxy = (
        scored_df[current_gmv_proxy_col].fillna(0.0) / gmv_window_days
    ).sum()

    incremental_low_ratings_proxy = (
        total_future_events * severe_harm_row["delta_low_rating"]
    )
    review_points_lost = (
        total_future_events * severe_harm_row["delta_review_loss"]
    )

    rows = []
    for profile_name, profile in assumption_profiles.items():
        compensation_cost_proxy = (
            total_future_gmv * profile["compensation_rate_on_prevented_gmv"]
        )
        reputation_cost_proxy = (
            incremental_low_ratings_proxy
            * profile["cost_per_incremental_low_rating_proxy_brl"]
        )
        total_harm_proxy = compensation_cost_proxy + reputation_cost_proxy

        rows.append(
            {
                "assumption_profile": profile_name,
                "seller_days": len(scored_df),
                "unique_sellers": scored_df["seller_id"].nunique()
                if "seller_id" in scored_df.columns else np.nan,
                "total_future_events": total_future_events,
                "total_future_gmv_brl": total_future_gmv,
                "incremental_low_ratings_proxy": incremental_low_ratings_proxy,
                "review_points_lost": review_points_lost,
                "compensation_cost_proxy_brl": compensation_cost_proxy,
                "reputation_cost_proxy_brl": reputation_cost_proxy,
                "total_harm_proxy_brl": total_harm_proxy,
                "current_gmv_proxy_brl": total_current_gmv_proxy,
            }
        )

    return pd.DataFrame(rows)


def build_assumption_profiles() -> Dict[str, Dict]:
    """
    Return conservative / base / aggressive assumption profiles for H4.
    """
    return {
        "conservative": {
            "compensation_rate_on_prevented_gmv": 0.010,
            "cost_per_incremental_low_rating_proxy_brl": 6.0,
            "margin_rate_on_gmv": 0.10,
            "actions": {
                "monitor": {
                    "efficacy": 0.05,
                    "ops_cost_per_flag": 1.0,
                    "throttle_loss_rate": 0.00,
                },
                "standard_support": {
                    "efficacy": 0.15,
                    "ops_cost_per_flag": 3.0,
                    "throttle_loss_rate": 0.00,
                },
                "intensive_support": {
                    "efficacy": 0.30,
                    "ops_cost_per_flag": 7.0,
                    "throttle_loss_rate": 0.00,
                },
                "throttle": {
                    "efficacy": 0.65,
                    "ops_cost_per_flag": 5.0,
                    "throttle_loss_rate": 0.25,
                },
            },
        },
        "base": {
            "compensation_rate_on_prevented_gmv": 0.020,
            "cost_per_incremental_low_rating_proxy_brl": 12.0,
            "margin_rate_on_gmv": 0.20,
            "actions": {
                "monitor": {
                    "efficacy": 0.10,
                    "ops_cost_per_flag": 1.5,
                    "throttle_loss_rate": 0.00,
                },
                "standard_support": {
                    "efficacy": 0.25,
                    "ops_cost_per_flag": 3.5,
                    "throttle_loss_rate": 0.00,
                },
                "intensive_support": {
                    "efficacy": 0.45,
                    "ops_cost_per_flag": 8.0,
                    "throttle_loss_rate": 0.00,
                },
                "throttle": {
                    "efficacy": 0.80,
                    "ops_cost_per_flag": 6.0,
                    "throttle_loss_rate": 0.20,
                },
            },
        },
        "aggressive": {
            "compensation_rate_on_prevented_gmv": 0.040,
            "cost_per_incremental_low_rating_proxy_brl": 20.0,
            "margin_rate_on_gmv": 0.30,
            "actions": {
                "monitor": {
                    "efficacy": 0.15,
                    "ops_cost_per_flag": 2.0,
                    "throttle_loss_rate": 0.00,
                },
                "standard_support": {
                    "efficacy": 0.35,
                    "ops_cost_per_flag": 4.0,
                    "throttle_loss_rate": 0.00,
                },
                "intensive_support": {
                    "efficacy": 0.60,
                    "ops_cost_per_flag": 9.0,
                    "throttle_loss_rate": 0.00,
                },
                "throttle": {
                    "efficacy": 0.90,
                    "ops_cost_per_flag": 7.0,
                    "throttle_loss_rate": 0.15,
                },
            },
        },
    }
    
    
def build_default_roi_scenarios(
    default_topk: float = 0.05,
) -> List[Dict]:
    """
    Return the default scenario library for H4.
    """
    return [
        {
            "scenario_family": "default",
            "scenario_name": "A_throttle_top_1pct",
            "description": "Hard throttle the top 1% highest-risk seller-days.",
            "tiers": [
                {
                    "tier_name": "top_1pct_throttle",
                    "start_frac": 0.00,
                    "end_frac": 0.01,
                    "action": "throttle",
                }
            ],
        },
        {
            "scenario_family": "default",
            "scenario_name": f"B_tiered_top_{int(default_topk*100)}pct",
            "description": "Top 1% intensive support; next band standard support.",
            "tiers": [
                {
                    "tier_name": "top_1pct_intensive",
                    "start_frac": 0.00,
                    "end_frac": 0.01,
                    "action": "intensive_support",
                },
                {
                    "tier_name": f"next_{int((default_topk-0.01)*100)}pct_standard",
                    "start_frac": 0.01,
                    "end_frac": default_topk,
                    "action": "standard_support",
                },
            ],
        },
        {
            "scenario_family": "default",
            "scenario_name": "C_monitor_top_10pct",
            "description": "Low-touch monitoring / warning for the top 10%.",
            "tiers": [
                {
                    "tier_name": "top_10pct_monitor",
                    "start_frac": 0.00,
                    "end_frac": 0.10,
                    "action": "monitor",
                }
            ],
        },
    ]


def build_k_sensitivity_scenarios(
    k_values: Sequence[float] = (0.01, 0.03, 0.05, 0.10),
    intensive_frac: float = 0.01,
) -> List[Dict]:
    """
    Build a K-sensitivity scenario family.
    """
    scenarios = []

    for k in k_values:
        if k <= intensive_frac:
            tiers = [
                {
                    "tier_name": f"top_{int(k*100)}pct_intensive",
                    "start_frac": 0.00,
                    "end_frac": k,
                    "action": "intensive_support",
                }
            ]
            desc = f"Intensive support on top {int(k*100)}% seller-days."
        else:
            tiers = [
                {
                    "tier_name": f"top_{int(intensive_frac*100)}pct_intensive",
                    "start_frac": 0.00,
                    "end_frac": intensive_frac,
                    "action": "intensive_support",
                },
                {
                    "tier_name": f"next_{int((k-intensive_frac)*100)}pct_standard",
                    "start_frac": intensive_frac,
                    "end_frac": k,
                    "action": "standard_support",
                },
            ]
            desc = (
                f"Top {int(intensive_frac*100)}% intensive support; "
                f"next {int((k-intensive_frac)*100)}% standard support."
            )

        scenarios.append(
            {
                "scenario_family": "k_sensitivity",
                "scenario_name": f"K_tiered_top_{int(k*100)}pct",
                "description": desc,
                "tiers": tiers,
            }
        )

    return scenarios


def _slice_rank_band(
    scored_df: pd.DataFrame,
    score_col: str,
    start_frac: float,
    end_frac: float,
) -> pd.DataFrame:
    """
    Slice a ranked score table into a non-overlapping top-K band.
    """
    sort_cols = [score_col]
    ascending = [False]

    for col in ["date", "seller_id"]:
        if col in scored_df.columns:
            sort_cols.append(col)
            ascending.append(True)

    ranked = scored_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    n = len(ranked)
    start_idx = int(np.floor(n * start_frac))
    end_idx = int(np.floor(n * end_frac))
    if end_frac > start_frac:
        end_idx = max(end_idx, start_idx + 1)

    return ranked.iloc[start_idx:end_idx].copy()


def _compute_action_exposure_and_ops_units(
    band: pd.DataFrame,
    action_name: str,
    current_gmv_proxy_col: str,
    gmv_window_days: int = 14,
    seller_col: str = "seller_id",
    throttle_duration_days: int = 1,
) -> Tuple[float, int]:
    """
    Compute the action-level GMV exposure proxy and the operational unit count.

    Why this helper exists
    ----------------------
    The early-warning panel is at the seller-day level, but not all interventions
    should be costed at the seller-day level:

    - For light-touch actions such as monitoring / support, seller-day costing is reasonable.
    - For restrictive actions such as throttle, seller-day costing can severely overstate
      business loss because:
        1) the same seller may be flagged on many consecutive days;
        2) `delivered_gmv_14d` is a rolling 14-day GMV proxy, so summing it across
           consecutive seller-days creates overlap.

    Design
    ------
    - For non-throttle actions:
        exposure proxy = sum(current_gmv_proxy_col / gmv_window_days) across rows
        ops units       = number of flagged seller-days
    - For throttle:
        exposure proxy = sum(max(current_gmv_proxy_col) / gmv_window_days) across unique sellers
                         × throttle_duration_days
        ops units       = number of unique flagged sellers

    Important assumption
    --------------------
    For throttle, exposure is computed as a single-day GMV estimate by converting
    the rolling GMV proxy into a daily proxy (e.g. delivered_gmv_14d / 14).
    If the throttle decision is assumed to persist for N days, multiply the
    exposure output by N or parameterise the duration explicitly via
    `throttle_duration_days`.

    Parameters
    ----------
    band : pd.DataFrame
        Ranked subset for one scenario tier.
    action_name : str
        Intervention action name, e.g. 'monitor', 'standard_support',
        'intensive_support', or 'throttle'.
    current_gmv_proxy_col : str
        Rolling GMV proxy column, e.g. 'delivered_gmv_14d'.
    gmv_window_days : int, default 14
        Number of days represented by the rolling GMV proxy.
    seller_col : str, default 'seller_id'
        Seller identifier column.
    throttle_duration_days : int, default 1
        Assumed number of days the throttle decision remains in effect.
        Increasing this value scales up the throttle GMV exposure linearly.
        Use for sensitivity analysis over {1, 3, 7} days.

    Returns
    -------
    (current_gmv_proxy_brl, ops_units)
        current_gmv_proxy_brl : float
            Current exposure proxy in BRL, aligned to the action unit.
        ops_units : int
            Operational unit count used to compute intervention handling cost.
    """
    if band.empty:
        return 0.0, 0

    if current_gmv_proxy_col not in band.columns:
        raise ValueError(f"Missing column: {current_gmv_proxy_col}")

    if action_name == "throttle":
        if seller_col in band.columns:
            seller_level_proxy = (
                band.groupby(seller_col)[current_gmv_proxy_col]
                .max()
                .fillna(0.0)
            )
            current_gmv_proxy_brl = (
                (seller_level_proxy / gmv_window_days).sum() * throttle_duration_days
            )
            ops_units = seller_level_proxy.index.nunique()
        else:
            current_gmv_proxy_brl = (band[current_gmv_proxy_col].fillna(0.0) / gmv_window_days).sum()
            ops_units = len(band)
    else:
        current_gmv_proxy_brl = (band[current_gmv_proxy_col].fillna(0.0) / gmv_window_days).sum()
        ops_units = len(band)

    return float(current_gmv_proxy_brl), int(ops_units)


def simulate_ranked_scenario(
    scored_df: pd.DataFrame,
    score_col: str,
    future_event_count_col: str,
    future_event_gmv_col: str,
    current_gmv_proxy_col: str,
    severe_harm_row: Dict,
    baseline_row: Dict,
    assumption_profile_name: str,
    assumption_profile: Dict,
    scenario_def: Dict,
    gmv_window_days: int = 14,
    throttle_duration_days: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate one ROI scenario under one assumption profile.

    Economic logic
    --------------
    - A flagged seller-day captures some future severe-event count and GMV.
    - An intervention prevents a fraction of those harms (`efficacy`).
    - Avoided compensation proxy is proportional to prevented severe-event GMV.
    - Avoided reputation proxy is proportional to avoided incremental low ratings.
    - Review-score points saved are reported as a secondary non-monetised KPI.
    - Restrictive actions (throttle) reduce current GMV throughput and incur
      a contribution-margin loss.
    - Net benefit is defined relative to the explicit no-intervention baseline:
        net benefit = avoided harm proxy - intervention cost

    Important action-unit correction
    --------------------------------
    The panel is at seller-day level, but throttle is treated as a seller-level
    restrictive action for costing purposes:
    - throttle exposure uses unique-seller deduplicated GMV proxy
    - throttle ops cost uses unique flagged sellers, not flagged seller-days

    Parameters
    ----------
    scored_df : pd.DataFrame
        Test/deployment window with risk scores and future-event columns.
    score_col : str
        Risk score column (e.g., score_lr_14d).
    future_event_count_col : str
        Count of future severe events in the chosen horizon.
    future_event_gmv_col : str
        GMV of future severe events in the chosen horizon.
    current_gmv_proxy_col : str
        Current rolling GMV exposure proxy (e.g., delivered_gmv_14d).
    severe_harm_row : dict
        Severe-harm coefficient row derived from H2.
    baseline_row : dict
        No-intervention baseline row for the same assumption profile.
    assumption_profile_name : str
        Name of the assumption profile.
    assumption_profile : dict
        One nested assumption profile from `build_assumption_profiles()`.
    scenario_def : dict
        One scenario definition from the scenario library.
    gmv_window_days : int, default 14
        Number of days represented by the rolling GMV proxy.
    throttle_duration_days : int, default 1
        Assumed number of days the throttle decision remains in effect.
        Passed directly to `_compute_action_exposure_and_ops_units`.
        Use 1 (default), 3, or 7 for sensitivity analysis.

    Returns
    -------
    (scenario_summary_df, tier_detail_df)
        scenario_summary_df : pd.DataFrame
            Single-row scenario summary.
        tier_detail_df : pd.DataFrame
            One row per scenario tier, with detailed metrics.
    """
    required_cols = [
        score_col,
        future_event_count_col,
        future_event_gmv_col,
        current_gmv_proxy_col,
    ]
    missing = [c for c in required_cols if c not in scored_df.columns]
    if missing:
        raise ValueError(f"scored_df is missing required columns: {missing}")

    total_rows = len(scored_df)
    total_unique_sellers = (
        scored_df["seller_id"].nunique() if "seller_id" in scored_df.columns else np.nan
    )
    total_future_events = scored_df[future_event_count_col].sum()
    total_future_gmv = scored_df[future_event_gmv_col].sum()

    tier_rows = []
    flagged_parts = []

    for tier in scenario_def["tiers"]:
        band = _slice_rank_band(
            scored_df=scored_df,
            score_col=score_col,
            start_frac=tier["start_frac"],
            end_frac=tier["end_frac"],
        )
        flagged_parts.append(band)

        action_name = tier["action"]
        action_params = assumption_profile["actions"][action_name]

        efficacy = action_params["efficacy"]
        ops_cost_per_flag = action_params["ops_cost_per_flag"]
        throttle_loss_rate = action_params["throttle_loss_rate"]

        flagged_rows = len(band)
        unique_flagged_sellers = (
            band["seller_id"].nunique() if "seller_id" in band.columns else np.nan
        )

        captured_future_events = band[future_event_count_col].sum()
        captured_future_gmv = band[future_event_gmv_col].sum()

        # ------------------------------------------------------------------
        # Action-aligned exposure proxy and ops units
        # ------------------------------------------------------------------
        current_gmv_proxy, ops_units = _compute_action_exposure_and_ops_units(
            band=band,
            action_name=action_name,
            current_gmv_proxy_col=current_gmv_proxy_col,
            gmv_window_days=gmv_window_days,
            seller_col="seller_id",
            throttle_duration_days=throttle_duration_days,
        )

        prevented_future_events = efficacy * captured_future_events
        prevented_future_gmv = efficacy * captured_future_gmv

        avoided_incremental_low_ratings = (
            prevented_future_events * severe_harm_row["delta_low_rating"]
        )
        avoided_review_points = (
            prevented_future_events * severe_harm_row["delta_review_loss"]
        )

        avoided_compensation_cost = (
            prevented_future_gmv
            * assumption_profile["compensation_rate_on_prevented_gmv"]
        )
        avoided_reputation_cost = (
            avoided_incremental_low_ratings
            * assumption_profile["cost_per_incremental_low_rating_proxy_brl"]
        )

        avoided_harm_proxy = avoided_compensation_cost + avoided_reputation_cost

        # ------------------------------------------------------------------
        # Cost side
        # ------------------------------------------------------------------
        ops_cost = ops_units * ops_cost_per_flag
        throttled_gmv = current_gmv_proxy * throttle_loss_rate
        margin_loss = throttled_gmv * assumption_profile["margin_rate_on_gmv"]

        total_cost = ops_cost + margin_loss
        net_benefit = avoided_harm_proxy - total_cost
        roi = net_benefit / total_cost if total_cost > 0 else np.nan

        tier_rows.append(
            {
                "assumption_profile": assumption_profile_name,
                "scenario_family": scenario_def["scenario_family"],
                "scenario_name": scenario_def["scenario_name"],
                "scenario_description": scenario_def["description"],
                "tier_name": tier["tier_name"],
                "action": action_name,
                "start_frac": tier["start_frac"],
                "end_frac": tier["end_frac"],
                "flagged_rows": flagged_rows,
                "flagged_share": flagged_rows / total_rows if total_rows > 0 else np.nan,
                "unique_flagged_sellers": unique_flagged_sellers,
                "flagged_seller_share": (
                    unique_flagged_sellers / total_unique_sellers
                    if total_unique_sellers and total_unique_sellers > 0 else np.nan
                ),
                "ops_units": ops_units,
                "captured_future_events": captured_future_events,
                "captured_future_gmv": captured_future_gmv,
                "prevented_future_events": prevented_future_events,
                "prevented_future_gmv": prevented_future_gmv,
                "avoided_incremental_low_ratings": avoided_incremental_low_ratings,
                "avoided_review_points": avoided_review_points,
                "avoided_compensation_cost_proxy_brl": avoided_compensation_cost,
                "avoided_reputation_cost_proxy_brl": avoided_reputation_cost,
                "avoided_harm_proxy_brl": avoided_harm_proxy,
                "current_gmv_proxy_brl": current_gmv_proxy,
                "ops_cost_brl": ops_cost,
                "throttled_gmv_brl": throttled_gmv,
                "margin_loss_brl": margin_loss,
                "total_cost_brl": total_cost,
                "net_benefit_brl": net_benefit,
                "roi": roi,
                "capture_rate_events": (
                    captured_future_events / total_future_events if total_future_events > 0 else np.nan
                ),
                "capture_rate_gmv": (
                    captured_future_gmv / total_future_gmv if total_future_gmv > 0 else np.nan
                ),
                "prevented_event_rate": (
                    prevented_future_events / total_future_events if total_future_events > 0 else np.nan
                ),
                "prevented_gmv_rate": (
                    prevented_future_gmv / total_future_gmv if total_future_gmv > 0 else np.nan
                ),
                "current_gmv_proxy_share": (
                    current_gmv_proxy / baseline_row["current_gmv_proxy_brl"]
                    if baseline_row["current_gmv_proxy_brl"] > 0 else np.nan
                ),
            }
        )

    tier_detail_df = pd.DataFrame(tier_rows)

    if flagged_parts:
        flagged_all = pd.concat(flagged_parts, ignore_index=True)
    else:
        flagged_all = pd.DataFrame(columns=scored_df.columns)

    total_avoided_harm = tier_detail_df["avoided_harm_proxy_brl"].sum()
    total_cost = tier_detail_df["total_cost_brl"].sum()
    total_net_benefit = tier_detail_df["net_benefit_brl"].sum()

    scenario_summary = {
        "assumption_profile": assumption_profile_name,
        "scenario_family": scenario_def["scenario_family"],
        "scenario_name": scenario_def["scenario_name"],
        "scenario_description": scenario_def["description"],
        "flagged_rows": tier_detail_df["flagged_rows"].sum(),
        "flagged_share": tier_detail_df["flagged_rows"].sum() / total_rows if total_rows > 0 else np.nan,
        "unique_flagged_sellers": (
            flagged_all["seller_id"].nunique()
            if "seller_id" in flagged_all.columns and len(flagged_all) > 0 else np.nan
        ),
        "flagged_seller_share": (
            flagged_all["seller_id"].nunique() / total_unique_sellers
            if "seller_id" in flagged_all.columns and total_unique_sellers and total_unique_sellers > 0 else np.nan
        ),
        "ops_units": tier_detail_df["ops_units"].sum(),
        "captured_future_events": tier_detail_df["captured_future_events"].sum(),
        "captured_future_gmv": tier_detail_df["captured_future_gmv"].sum(),
        "prevented_future_events": tier_detail_df["prevented_future_events"].sum(),
        "prevented_future_gmv": tier_detail_df["prevented_future_gmv"].sum(),
        "avoided_incremental_low_ratings": tier_detail_df["avoided_incremental_low_ratings"].sum(),
        "avoided_review_points": tier_detail_df["avoided_review_points"].sum(),
        "avoided_compensation_cost_proxy_brl": tier_detail_df["avoided_compensation_cost_proxy_brl"].sum(),
        "avoided_reputation_cost_proxy_brl": tier_detail_df["avoided_reputation_cost_proxy_brl"].sum(),
        "avoided_harm_proxy_brl": total_avoided_harm,
        "current_gmv_proxy_brl": tier_detail_df["current_gmv_proxy_brl"].sum(),
        "ops_cost_brl": tier_detail_df["ops_cost_brl"].sum(),
        "throttled_gmv_brl": tier_detail_df["throttled_gmv_brl"].sum(),
        "margin_loss_brl": tier_detail_df["margin_loss_brl"].sum(),
        "total_cost_brl": total_cost,
        "net_benefit_brl": total_net_benefit,
        "roi": total_net_benefit / total_cost if total_cost > 0 else np.nan,
        "capture_rate_events": (
            tier_detail_df["captured_future_events"].sum() / total_future_events
            if total_future_events > 0 else np.nan
        ),
        "capture_rate_gmv": (
            tier_detail_df["captured_future_gmv"].sum() / total_future_gmv
            if total_future_gmv > 0 else np.nan
        ),
        "prevented_event_rate": (
            tier_detail_df["prevented_future_events"].sum() / total_future_events
            if total_future_events > 0 else np.nan
        ),
        "prevented_gmv_rate": (
            tier_detail_df["prevented_future_gmv"].sum() / total_future_gmv
            if total_future_gmv > 0 else np.nan
        ),
        "current_gmv_proxy_share": (
            tier_detail_df["current_gmv_proxy_brl"].sum() / baseline_row["current_gmv_proxy_brl"]
            if baseline_row["current_gmv_proxy_brl"] > 0 else np.nan
        ),
        "baseline_total_future_events": baseline_row["total_future_events"],
        "baseline_total_future_gmv_brl": baseline_row["total_future_gmv_brl"],
        "baseline_incremental_low_ratings_proxy": baseline_row["incremental_low_ratings_proxy"],
        "baseline_review_points_lost": baseline_row["review_points_lost"],
        "baseline_total_harm_proxy_brl": baseline_row["total_harm_proxy_brl"],
        "residual_harm_proxy_brl": baseline_row["total_harm_proxy_brl"] - total_avoided_harm,
    }

    scenario_summary_df = pd.DataFrame([scenario_summary])
    return scenario_summary_df, tier_detail_df

def run_scenario_grid(
    scored_df: pd.DataFrame,
    score_col: str,
    future_event_count_col: str,
    future_event_gmv_col: str,
    current_gmv_proxy_col: str,
    severe_harm_row: Dict,
    baseline_df: pd.DataFrame,
    assumption_profiles: Dict[str, Dict],
    scenario_library: List[Dict],
    gmv_window_days: int = 14,
    throttle_duration_days: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all scenario x assumption combinations and return combined results.
    """
    summary_frames = []
    tier_frames = []

    for profile_name, profile in assumption_profiles.items():
        baseline_row = baseline_df.loc[
            baseline_df["assumption_profile"] == profile_name
        ].iloc[0].to_dict()

        for scenario_def in scenario_library:
            summary_df, tier_df = simulate_ranked_scenario(
                scored_df=scored_df,
                score_col=score_col,
                future_event_count_col=future_event_count_col,
                future_event_gmv_col=future_event_gmv_col,
                current_gmv_proxy_col=current_gmv_proxy_col,
                severe_harm_row=severe_harm_row,
                baseline_row=baseline_row,
                assumption_profile_name=profile_name,
                assumption_profile=profile,
                scenario_def=scenario_def,
                gmv_window_days=gmv_window_days,
                throttle_duration_days=throttle_duration_days,
            )
            summary_frames.append(summary_df)
            tier_frames.append(tier_df)

    scenario_summary = pd.concat(summary_frames, ignore_index=True)
    scenario_tiers = pd.concat(tier_frames, ignore_index=True)

    return scenario_summary, scenario_tiers