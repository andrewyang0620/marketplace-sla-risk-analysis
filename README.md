# Optimizing Marketplace Reliability: SLA Risk Analysis & Intervention ROI

![Business Analytics](https://img.shields.io/badge/Business-Analytics-blue)
![Python](https://img.shields.io/badge/Python-3.13-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikitlearn)
![Altair](https://img.shields.io/badge/Altair-6.0-blueviolet)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **End-to-end analytical engineering project** spanning four linked business questions on marketplace seller reliability.
> 
> **Analytical chain:** raw transactional data → risk concentration (H1) → customer harm quantification (H2) → 14-day early-warning model (H3) → intervention ROI simulation (H4).
> 
> Each stage is backed by a modular `src/` package, cross-validated where applicable, and delivered with a written analytical summary.

---

## Executive Summary

| Hypothesis | Business Question | Finding | Key Metric |
|---|---|---|---|
| **H1 - Risk Concentration** | Is SLA risk evenly distributed? | Highly concentrated in a small seller tail | Top 10% sellers → **37% of all violations** |
| **H2 - Customer Harm** | Do SLA delays harm customers? | Dose-dependent harm; severity cliff at 3 days late | **−1.94 review score**, **+52 pp** low-rating rate |
| **H3 - Early Warning** | Can we predict seller risk 14 days out? | Logistic regression beats random at scale with validated lead time | AUC = **0.753**, GMV Recall@K=5% = **27.9%** |
| **H4 - Intervention ROI** | Which intervention is worth the cost? | Precision throttle on top 1% is the only conservative-viable strategy | ROI = **5.83×**, net benefit **+2,418 BRL**, GMV footprint **0.46%** |

**Operational recommendation:** Throttle the top 1% of seller-days by risk score (9 unique sellers at Olist's scale). Total cost: 415 BRL (54 ops + 361 margin). Avoided harm proxy: 2,833 BRL. Avoids **167 incremental low ratings** and **630 review-score points** per cycle, positive ROI confirmed under both base and conservative assumption profiles. Absolute figures reflect dataset scale; the ROI structure and decision logic transfer to larger deployments.

---

## Analytical Design Decisions

Four non-trivial choices distinguish this from a standard notebook pipeline:

1. **Model selection rationale, not just model selection.** Logistic Regression was chosen over LightGBM after walk-forward cross-validation revealed that LightGBM's AUC degrades to near-chance in at least one fold under temporal distribution shift. LR fold-to-fold AUC range stays < 0.04 across all 5 folds. Choosing the simpler, temporally stable model was a deliberate production decision.

2. **End-to-end economic chain.** H2 harm coefficients (review score drop, low-rating rate lift) feed directly into H4's ROI simulation as monetisation bridges. H1's risk concentration finding motivates the 14-day prediction horizon in H3, which determines intervention timing in H4. The chain is explicit, not post-hoc.

3. **Business-relevant evaluation metric.** Beyond ROC AUC, the primary metric is **GMV Recall@K**: the share of future severe-event GMV captured by flagging the top K% of seller-days. This reflects the operational reality that a high-GMV seller missing SLA matters more than a low-volume seller with the same violation count.

4. **Observational depth, not causal overclaim.** H2 is validated across five analytical levels (global descriptive, within-category stratification, within-seller pre/post, dose–response gradient, threshold discovery). Causal limitations are documented in every summary; no DiD or IV is claimed.

---

## 1. Project Motivation

As marketplaces scale, operational failures are rarely evenly distributed. A small subset of sellers can disproportionately undermine platform reliability by repeatedly missing delivery SLAs, triggering cancellations, and eroding customer trust.

Left unmanaged, these failures compound:
- They degrade customer experience and reduce repeat purchase likelihood
- They increase operational costs through refunds and support burden
- They threaten long-term platform credibility

The challenge is not whether poor-performing sellers exist, but whether their risk can be **identified early and managed economically** without sacrificing growth.

---

## 2. Key Business Questions

1. **H1 - Concentration:** Is SLA risk evenly distributed across sellers, or highly concentrated?  
   *If risk is concentrated, high impact is achievable at low operational cost.*

2. **H2 - Customer Harm:** Do SLA violations associate with degraded customer experience and retention?  
   *A delay is not just a bad review; quantify the dose–response relationship and identify the severity threshold that warrants intervention.*

3. **H3 - Early Warning:** Can high-risk sellers be identified prior to severe SLA breaches?  
   *Early identification enables proactive intervention, reducing severe SLA incidence.*

4. **H4 - Intervention ROI:** Which intervention strategies maximize reliability improvement per unit of GMV at risk?  
   *Balance reliability improvement against GMV guardrail exposure; find the optimal threshold.*

5. **H5 - Attribution** *(scope boundary)*: Is risk driven by seller behaviour or logistics infrastructure?  
   *Identified as a stretch goal; out of scope for this analysis.*

---

## 3. Data Source

- **Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)
- **Scale:** 99,441 orders · 3,095 sellers · 96,096 customers · 2016–2018
- **Tables used:** orders, order items, order reviews, customers, sellers, products, geolocation, category translations
- **SLA definition (custom):** `delay_days = carrier_delivery_date − estimated_delivery_date`
  - SLA violation: `delay_days > 0` (any late delivery, 6.57% of delivered orders)
  - Severe violation: `delay_days > 7` (used as H3 model prediction target, 2.88% of delivered orders)

---

## 4. Analysis Architecture

```
Raw Data  (8 CSV tables, data/raw/)
    │
    ▼
00_data_validation.ipynb   ── Data quality audit, SLA definition, join validation
    │
    ▼
01_seller_risk.ipynb        ── H1: Risk concentration, seller-level violation rates & ranking
    │
    ▼
02_customer_impact.ipynb    ── H2: Dose–response harm model (review scores, low-rating rates)
    │
    ▼
03_early_warning.ipynb      ── H3: Logistic regression, walk-forward CV, GMV recall curves
    │
    ▼
04_intervention_roi.ipynb   ── H4: Scenario grid (3 actions × 3 cost profiles), ROI simulation
    │
    ▼
data/interim/ + data/processed/   ── Parquet artefacts consumed by downstream steps
```

Each notebook ends with a **Markdown executive summary cell**. Written deep-dives live in [`docs/summary-of-analysis/`](docs/summary-of-analysis/).

---

## 5. Key Findings

### H1 - Seller Risk Concentration
- SLA violation rate is **highly skewed**: the top 10% of sellers account for **37%** of all violations.
- Concentration justifies a precision intervention strategy; acting on the full seller base is unnecessary and expensive.
- Seller-level features (repeat violation rate, processing time variance) carry strong predictive signal for future breaches.

### H2 - Customer Harm Quantification
- SLA violations associate with **dose-dependent harm** with a critical severity cliff at ≥ 3 days late:
  - Mean review score drops **−1.94 points** (on-time: 4.29 → severe: 1.74 at 6+ days)
  - Low-rating rate (≤ 3 stars) rises **+52 pp** across the on-time → 6-day-late range (17.3% → 84.7%)
- Validated across **five analytical levels**: global aggregate, within-category stratification (29 categories, 100% consistent), within-seller pre/post event, dose–response gradient, and threshold discovery.
- *All analyses are observational. No DiD or IV is used; the findings support association, not strict causation.*

### H3 - Early-Warning Model
- **Logistic Regression** (14-day horizon, `class_weight='balanced'`) achieves test **ROC AUC = 0.753**, **AP = 0.383**.
- **GMV Recall@K=5%: 27.9%** - flagging 5% of seller-days captures 27.9% of future severe-event GMV in scope.
- LR selected over LightGBM: walk-forward CV showed LightGBM AUC degrades to near-chance in unstable folds due to temporal distribution shift; LR fold-to-fold range < 0.04. Robustness over raw accuracy under non-stationarity.
- Flag-rate monitoring and periodic retraining specified in model governance; positive rate drift (+12 pp from earliest to latest fold) documented as a production constraint.

### H4 - Intervention ROI Simulation
- Three scenarios (throttle top 1% / tiered top 5% / monitor top 10%) × three cost profiles (conservative / base / aggressive).
- **Only Scenario A (throttle top 1%) is viable under conservative assumptions (ROI > 0).**

  | Scenario | ROI (base) | Net Benefit | GMV Footprint | Conservative Viable |
  |---|---|---|---|---|
  | A - Throttle top 1% | **5.83×** | +2,418 BRL | **0.46%** | √ |
  | B - Tiered top 5% | 0.03× | +105 BRL | 23.2% | X |
  | C - Monitor top 10% | −0.34× | −846 BRL | 36.7% | X |

- K-sensitivity confirms ROI falls monotonically; the tiered design turns negative at K = 10%.

> **Scale context:** At Olist's scale (3,095 total sellers), absolute BRL figures are illustrative. The same framework applied to a marketplace 10× larger would flag ~90 sellers per cycle with proportionally larger avoided harm; the pipeline and economic logic scale linearly.

---

## 6. Project Structure

```
├── notebooks/
│   ├── 00_data_validation.ipynb    # Data quality & SLA definition
│   ├── 01_seller_risk.ipynb        # H1: Risk concentration analysis
│   ├── 02_customer_impact.ipynb    # H2: Customer harm quantification
│   ├── 03_early_warning.ipynb      # H3: Predictive early-warning model
│   └── 04_intervention_roi.ipynb   # H4: Intervention ROI simulation
│
├── src/
│   ├── config.py                   # Paths & constants
│   ├── data/
│   │   ├── load_raw.py             # Raw CSV loaders
│   │   └── preprocessing.py        # Join, SLA computation, feature prep
│   ├── features/
│   │   ├── sla_metrics.py          # SLA violation metrics
│   │   ├── seller_metrics.py       # Seller-level aggregation & scoring
│   │   ├── customer_impact.py      # Customer harm quantification
│   │   ├── early_warning.py        # Feature engineering for H3 model
│   │   └── intervention_roi.py     # ROI simulation: scenario grid, baseline, exposure
│   └── utils/
│       ├── eda.py                  # EDA helpers
│       ├── metrics.py              # Evaluation metrics
│       └── validation.py           # Data quality checks
│
├── data/
│   ├── raw/                        # Original Olist CSVs (not committed; see Data Source)
│   ├── interim/                    # Processed parquet artefacts (model inputs/outputs)
│   └── processed/                  # Final output tables (orders_sellers, risk ranking)
│
├── docs/
│   └── summary-of-analysis/        # Written deep-dives for H0–H4 (Markdown)
│
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 7. How to Reproduce

### Prerequisites
Python 3.10+ with packages listed in `requirements.txt`.

```bash
# 1. Clone the repo
git clone <repo-url>
cd Marketplace-Sla-Risk-Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add raw data
# Download from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Extract all CSVs into data/raw/

# 4. Run notebooks in order
jupyter lab
# Execute: 00 → 01 → 02 → 03 → 04
```

Each notebook reads from `data/raw/` (or `data/interim/` for downstream steps) and writes parquet artefacts consumed by subsequent notebooks.

---

## 8. Technical Stack

| Layer | Tools |
|---|---|
| Data processing | `pandas 2.2`, `numpy 2.1` |
| Machine learning | `scikit-learn 1.6` (LogisticRegression, StratifiedKFold, calibration) |
| Visualisation | `altair 6.0` (interactive charts rendered in JupyterLab) |
| Serialisation | `pyarrow 22` / Parquet |
| Statistical analysis | `scipy 1.15` |
| Notebook environment | `JupyterLab 4.5` |
| Code organisation | Modular `src/` package: feature, data, and utility layers |


