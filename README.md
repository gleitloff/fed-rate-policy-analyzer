# FED Policy Decision Analyzer

## Overview

This project implements a two-step logistic regression framework to model and predict Federal Reserve policy decisions using macroeconomic and financial market data.

The model separates the decision to act (rate change vs hold) from the directional choice (hike vs cut). The objective is to approximate the Fed's reaction function and to evaluate whether macro-financial conditions can predict policy decisions out of sample.

The model is deliberately empirical: variables and structure are motivated by observed regularities in monetary policy decisions. This project does not have any extensive theoretical foundations, nor aims to produce theoretical insights about FED decision making.

---

## Methodological Framework

The modeling strategy follows a sequential decision structure.

**Step 1 — Action decision:** a binary logistic regression estimates P(pivot), the probability that the Fed changes the policy rate rather than holding.

**Step 2 — Directional decision:** conditional on a predicted pivot, a second logistic regression estimates whether the move is a hike or a cut.

Joint outcome probabilities are computed as:

- P(hike) = P(pivot) × P(hike | pivot)
- P(cut)  = P(pivot) × P(cut  | pivot)
- P(hold) = 1 − P(pivot)

---

## Data

All series are sourced from FRED and cover January 1990 to the present.

| Variable | FRED code |
|----------|-----------|
| Federal Funds Rate | FEDFUNDS |
| Core CPI | CPILFESL |
| Unemployment Rate | UNRATE |
| Industrial Production | INDPRO |
| Average Hourly Earnings | AHETPI |
| 2Y Treasury Yield | DGS2 |
| 10Y Treasury Yield | DGS10 |
| VIX | VIXCLS |

Daily series are forward-filled then averaged to monthly frequency. Monthly series are used directly.

---

## Feature Engineering

### Financial indicators (lag 1 month)
- **Yield curve slope** — 10Y minus 2Y Treasury spread
- **Log VIX** — market volatility proxy

### Macroeconomic indicators (lag 2 months)
- **Core inflation YoY** — year-on-year change in core CPI
- **Industrial production growth YoY**
- **Wage growth acceleration** — first difference of YoY wage growth
- **Unemployment gap** — deviation from 24-month rolling mean

The FED takes monetary policy decisions based on the data they have.
The lag structure aims to reflect and approximate real time data availability: financial variables are observable within the month, macro releases arrive with a publication delay. This prevents look-ahead bias and aligns the model with information realistically available to policymakers at the time of decision.

The feature set includes the lagged policy move as an autoregressive component. This captures the empirical persistence of Fed cycles but introduces an endogeneity risk (discussed in the limitations section).

---

## Target Construction

Monthly changes in the Federal Funds Rate are classified into three outcomes:

| Class | Label | Condition |
|-------|-------|-----------|
| 0 | Hold | \|Δrate\| < 12.5 bps |
| 1 | Hike | Δrate ≥ +12.5 bps |
| 2 | Cut  | Δrate ≤ −12.5 bps |

The 12.5 bps threshold filters rounding noise in FRED monthly averages without discarding genuine small moves.

---

## Estimation Strategy

The model uses a rolling window approach: at each step, both logit models are re-estimated on all available past data and generate a one-month-ahead prediction. No future data enters the training set at any point.

The initial window is set to 72 months (6 years). This is an arbitrary but defensible choice — it ensures the first training set covers at least one full rate cycle. The value can be adjusted manually.

> Notes about the two step rolling logit specification <br>
> i) In step 1, we fix `class_weight = balanced` to account for many hold months. However, for step 2 (condition on pivot), we do not balance weights.<br>
> ii) The `LogisticRegression` uses the basic specification from the sci-kit learn package, and we apply an *lbfgs* solver.

---

## Data Diagnostics

**Stationarity (ADF test, 5% threshold)**

All features pass except `core_infl_yoy_lag2` (p = 0.055). This variable is borderline non-stationary, which may affect coefficient stability.

**Multicollinearity (VIF)**

`log_vix` (VIF = 13.3) and `core_infl_yoy` (VIF = 9.1) exceed the conventional threshold of 10. This does not affect out-of-sample predictive performance but may limit the individual interpretability of their rolling coefficients.

---

## Results (WINDOW = 72 months, 340 out-of-sample observations, 1996–2026 - as of May 2026)

| Metric | Value |
|--------|-------|
| Overall accuracy | 66.5% |
| ROC AUC — pivot detection | 0.77 |
| Recall — Hold | 65% |
| Recall — Hike | 66% |
| Recall — Cut | 74% |
| Precision when signalling Hike | 37.5% |
| Precision when signalling Cut | 45.6% |

The model detects policy pivots with reasonable discrimination capacity (AUC 0.77). Directional precision is moderate: when the model signals a hike or cut, it is correct roughly 4 times in 10. This reflects the inherent difficulty of predicting not just whether the Fed moves, but in which direction.

---

## Limitations

**Autoregressive component** :  Including the lagged policy move may improve fit by capturing cycle persistence, but introduces endogeneity. Performance with and without this variable has not yet been formally compared (planned for a future version).

**Non-stationary inflation variable** : `core_infl_yoy` is borderline non-stationary.

**Classification threshold** : The current decision rule picks the argmax of the predicted class. A probabilistic rule comparing P(hike), P(cut), and P(hold) directly would be more principled (planned for a future version).

**No feature scaling** : Logistic regression coefficients are not scale-invariant. The rolling coefficient chart is indicative of sign and direction, not of relative magnitudes across variables. Comparison among feature sensitivity is not immediate nor comprehensive.

**Window optimisation** : The `optimise_window` function evaluates each candidate window on the full dataset, which may introduce a certain look-ahead bias in the window selection itself. The default of 72 months (ie. 6 years) is set independently of this optimisation.

**Structural breaks** :  The model does not test for parameter instability over time. The Fed's reaction function plausibly shifted across the 1990s, the ZLB period (2009–2015), and the post-2022 hiking cycle.

---

## Roadmap

- Probabilistic decision rule (argmax over joint outcome probabilities)
- Formal comparison with and without the lagged policy variable
- Feature scaling for coefficient interpretability
- First-difference specification for core inflation
- Structural break testing (Chow test or recursive coefficient plot)

---

## Reproducibility

The full workflow runs from a single script using publicly available FRED data. No external files required.

---

## Disclaimer

This project is independent personal research, conducted for learning and exploration purposes. Not intended for professional or investment use.

---

## Development notice

This project was developed with AI assistance, used as an exploration and development accelerator. All methodological choices, variable selection and model structure were defined by the author. AI suggestions were critically reviewed, and selectively implemented. 
