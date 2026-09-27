# Heavy Equipment Selling Price Prediction

Solution for the *Heavy Equipment Selling Price Prediction Challenge* (MLP Project 2026 T2) — predicting the resale price of heavy industrial machinery from a ~138K-row dataset of operational, transactional, and technical records, evaluated on RMSLE.

## Overview

The dataset captures finalized accounting/operational events for heavy equipment, with system metadata, geographic indicators, and deep component specifications. The goal is to explore the data and build models that forecast `TargetValue` (selling price), scored on RMSLE.

## Approach

**EDA**
- `TargetValue` is heavily right-skewed → trained on `log1p(TargetValue)` (MSE-on-log ≈ optimizing RMSLE directly)
- `ManufactureYear` had invalid placeholder values (1000/1001) → nulled out before computing equipment age
- `OperationalHoursMeter` relates to price multiplicatively, not linearly → used its log
- `TransactionDate` showed real price drift over time → decomposed into year/month/quarter/day-of-week

**Feature engineering**
- Cyclical encoding for `SaleMonth` (sin/cos)
- Missingness-flag columns for sparse spec fields (presence itself is a signal)
- Asset resale-count feature from `AssetID`
- K-fold target encoding for high-cardinality categoricals (`AssetID`, `ProductConfigID`) to avoid leakage

**Modeling**
- Baseline: `DummyRegressor` (mean/median)
- Ridge regression, tuned via `GridSearchCV`
- LightGBM, XGBoost, CatBoost, each tuned via `RandomizedSearchCV`
- Final prediction: NNLS-weighted blend of the three gradient boosting models' out-of-fold predictions

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, seaborn, matplotlib

## Repo contents

- `notebook.ipynb` — full EDA, feature engineering, and modeling pipeline
