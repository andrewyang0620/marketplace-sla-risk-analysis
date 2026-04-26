# Intervention ROI Simulation – Summary

- **Analysis scope & data**
  - Unit of analysis: individual seller-days, inherited from the H3 deployment panel.
  - Deployment panel: **16,497 seller-days** across **698 unique sellers** (test split, last 30% of dates; Jan 2017 – Aug 2018).
  - Ranking signal: Logistic Regression risk score rebuilt from the H3 pipeline (`class_weight='balanced'`, H = 14 days); test ROC AUC = **0.7529**, Average Precision = **0.3826**.
  - Harm bridge: H2 severe-harm coefficients (`6+_days_late` vs `on_time_or_early`): `delta_low_rating = 0.674` (low-rating rate diff: 84.7% − 17.3%), `delta_review_loss = 2.550` (mean review diff: 4.29 − 1.74).
  - Assumption profiles: **conservative / base / aggressive** (3 levels of efficacy, ops cost per flag, throttle loss rate).
  - Scenarios evaluated: 3 default (A/B/C) × 3 profiles + 4 K-sensitivity (1/3/5/10%) × 3 profiles = **21 simulations**.

- **Economic proxy design**
  - **Compensation cost proxy** = prevented severe-event GMV × `compensation_rate_on_prevented_gmv`.
  - **Reputation cost proxy** = avoided incremental low ratings × `cost_per_incremental_low_rating_proxy_brl`.
  - **Review-score points saved**: reported as a non-monetised supporting KPI only; not included in net benefit calculation.
  - **Net benefit** = avoided harm proxy − total intervention cost (ops cost + margin loss from throttle).
  - **ROI** = net benefit / total intervention cost.

- **Baseline (No-Intervention Benchmark)**
  - Under base assumptions, the deployment window implies the following if no action is taken:

  | Metric | Value |
  |---|---|
  | Future severe events | **3,608** |
  | Future severe-event GMV | **551,466 BRL** |
  | Incremental low-rating proxy | **2,431** low ratings |
  | Review points lost | **9,200 pts** |
  | Compensation cost proxy | 11,029 BRL |
  | Reputation cost proxy | 29,177 BRL |
  | **Total harm proxy** | **40,206 BRL** |
  | Current GMV proxy footprint | 1,974,666 BRL |

- **Section 6: Default scenario grid (3 scenarios × 3 profiles)**
  - Three default scenarios evaluated:
    - **A — throttle top 1%**: hard throttle on the 164 highest-risk seller-days (9 unique sellers after deduplication).
    - **B — tiered top 5%**: tiered intervention (intensive top 1% + light-touch remaining 4%).
    - **C — monitor top 10%**: monitoring-only on the top 10% seller-days.
  - Base-case results:

  | Scenario | Flagged | Capture GMV | Prevented GMV | Net benefit | ROI | GMV share affected | Margin loss |
  |---|---|---|---|---|---|---|---|
  | **A — throttle top 1%** | 1.0% | 9.5% | **7.6%** | **+2,418 BRL** | **5.83** | **0.46%** | 361 BRL |
  | B — tiered top 5% | 5.0% | 27.9% | 8.9% | +105 BRL | 0.03 | 23.2% | 0 |
  | C — monitor top 10% | 10.0% | 37.9% | 3.8% | −846 BRL | −0.34 | 36.7% | 0 |

  - Scenario B captures 3× more future severe-event GMV in scope than A, but its broader ops cost base nearly offsets the larger avoided harm, leaving only marginal net benefit (+105 BRL).
  - Scenario C has the worst economics despite the highest coverage: monitoring-only efficacy (5%) is too low to justify ops cost at a 10% flag rate.
  - Scenario A achieves the highest ROI because throttle is costed at the **unique-seller level** (9 sellers, ops cost = 54 BRL) rather than the seller-day level, keeping intervention cost minimal relative to avoided harm.

- **Section 7: Base-case scenario comparison**
  - Under base assumptions, Scenario A dominates on ROI (5.83) and net benefit (+2,418 BRL) with the lowest guardrail footprint (0.46% GMV share, 361 BRL margin loss).
  - Scenario B captures more absolute GMV in scope (27.9% vs 9.5%) but provides only marginal net benefit (ROI = 0.03), offering limited safety margin against adverse cost shocks.
  - `residual_harm_proxy_brl` after Scenario A intervention: **37,373 BRL** (93% of baseline harm remains — acceptable given the 0.46% guardrail constraint).

- **Section 8: K-sensitivity analysis (tiered design)**
  - Holding the tiered intervention design constant, ROI under base assumptions:

  | K | ROI | Net benefit | GMV share affected | Capture GMV |
  |---|---|---|---|---|
  | 1% | 0.21 | +282 BRL | 7.7% | 9.5% |
  | 3% | 0.08 | +205 BRL | 16.4% | 18.4% |
  | **5%** (H3 default) | 0.03 | +105 BRL | 23.2% | 27.9% |
  | 10% | −0.27 | −1,733 BRL | 36.7% | 37.9% |

  - ROI falls monotonically as K increases; the tiered scenario turns negative at K = 10%.
  - The H3-recommended K = 5% policy remains marginally positive under base assumptions but offers limited headroom against cost or efficacy shocks.
  - ROI peak at K = 1% reflects that the highest-ranked seller-days carry disproportionately more future severe-event concentration than the marginal rows added by widening K.

- **Section 9: Scenario visualisations**
  - Three charts generated:
    1. **Net benefit by scenario and assumption profile** (grouped bar): confirms Scenario A dominance across all profiles; B is near-zero under base and turns negative under conservative; C is negative across all profiles.
    2. **Trade-off frontier** (scatter, all 3 profiles × 3 default scenarios): color = assumption profile, shape = scenario; illustrates that A maintains high prevented GMV rate at minimal GMV share regardless of profile.
    3. **K-sensitivity ROI curve** (line, base profile): clean monotone decline from K=1% to K=10%; confirms ROI sign change at K=10%.

- **Section 10: Recommendation rule**
  - Viability gate: ROI > 0 under **conservative** assumptions.
  - Only **A_throttle_top_1pct** passes; B and C fail under conservative efficacy and cost parameters.
  - Among viable scenarios, highest base-case net benefit selects **A_throttle_top_1pct** as the recommended scenario.

- **H4 Verdict**

  | Evaluation dimension | Finding | Support |
  |---|---|---|
  | Positive ROI under base | Scenario A: ROI = 5.83 | **Strong** |
  | Positive ROI under conservative | Scenario A only (ROI > 0) | **Moderate** |
  | Guardrail within bounds | GMV share 0.46%, margin loss 361 BRL | **Strong** |
  | K robustness | ROI positive for K ≤ 5% (tiered); peak at K = 1% | **Moderate** |
  | CX harm avoided | 167 low ratings, 630 review-score points saved | **Moderate** (proxied) |

  **H4 is supported under the recommended scenario.** A tightly scoped, high-precision throttle on the top 1% of highest-risk seller-days yields positive ROI across both base and conservative assumption profiles, while keeping the GMV guardrail footprint well below 1%.

- **Operational recommendation**

  | Decision | Setting | Rationale |
  |---|---|---|
  | **Recommended scenario** | A — hard throttle top 1% seller-days | Only scenario with positive conservative ROI; highest base-case net benefit |
  | **Ranking signal** | Logistic Regression, H = 14 days | H3 deployment model; AUC 0.753 |
  | **Flagging policy** | Top 1% seller-days by risk score → 9 unique sellers per cycle | 164 seller-days → 9 sellers after deduplication |
  | **Action** | Hard throttle, seller-level (1-day default) | Extend to 3–7 days for persistent risk signals |
  | **Primary KPI** | ROI and net benefit BRL under conservative assumptions | Robustness-first; conservative = viable threshold |
  | **Guardrail KPI** | `current_gmv_proxy_share` < 5%; `margin_loss_brl` < 500 BRL | Confirmed: 0.46% and 361 BRL under recommended scenario |
  | **Model governance** | Inherit from H3: monitor flag-rate drift; retrain periodically | Non-stationarity confirmed in H3 walk-forward CV |

- **Limitations**
  - **Assumption-based efficacy**: scenario efficacy (30–70%) is not causally identified; actual harm prevention may differ from the assumed rate.
  - **Reputation cost is proxied**: `cost_per_incremental_low_rating_proxy_brl` is a monetisation assumption, not a directly observed revenue figure.
  - **Review-score savings not monetised**: avoided review-point loss is reported as a non-monetised KPI only.
  - **Seller-day action unit**: throttle is deduplicated to unique sellers, but other actions remain at seller-day level and may overstate handling burden for sellers flagged on many consecutive days.
  - **GMV proxy**: `delivered_gmv_14d` is a 14-day rolling delivered GMV, not exact realised forward revenue; actual business impact may differ.
  - **No causal identification**: the simulation quantifies correlation-based opportunity, not counterfactual intervention gain. A randomised experiment would be required to establish the true causal effect of throttle on future severe-event rates.
  - **Single throttle duration**: the base simulation assumes a 1-day throttle decision. Extending to 3–7 days is supported via the `throttle_duration_days` parameter but is not explored in the default scenario grid.
  - **Dataset scale**: Olist contains 3,095 total sellers across 2016–2018; top 1% of seller-days deduplicates to 9 unique targets. Absolute BRL figures are illustrative at this scale, not operationally significant. The framework scales linearly: a marketplace with 30,000 sellers would flag ~90 sellers per cycle with proportionally larger avoided harm and the same ROI structure.

- **Artifacts saved for downstream analysis**
  - `h4_baseline_no_intervention.parquet`: No-intervention benchmark across all 3 assumption profiles.
  - `h4_default_scenario_summary.parquet`: One row per scenario × profile (9 rows); full economic metrics.
  - `h4_default_scenario_tiers.parquet`: Tier-level detail for each default scenario simulation.
  - `h4_k_sensitivity_summary.parquet`: One row per K-scenario × profile (12 rows).
  - `h4_k_sensitivity_tiers.parquet`: Tier-level detail for K-sensitivity simulations.
  - `h4_recommended_scenario.parquet`: Viable base-case rows sorted by net benefit descending; `iloc[0]` is the top recommendation.

**Project conclusion**:  
H1 through H4 form a complete evidence chain. H1 confirms risk concentration (top 10% of sellers → 37% of violations). H2 establishes dose–response customer harm with a 3-day severity cliff. H3 provides a validated early-warning model with 14-day lead time and 27.9% GMV recall at K=5%. H4 closes the loop: the H3 ranking signal, applied with a tightly scoped throttle action, yields a defensible positive-ROI intervention that protects customer experience while keeping GMV guardrail exposure below 0.5%.
