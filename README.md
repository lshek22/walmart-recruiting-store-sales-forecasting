# Walmart Recruiting — Store Sales Forecasting

University ML final project based on the
[Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)
Kaggle competition.

## Problem description

The goal is to forecast **weekly sales** for every **department in each of Walmart’s 45 stores**, using historical sales and store/feature data.

There are thousands of independent `(Store, Dept)` series. Predictions are scored with **Weighted Mean Absolute Error (WMAE)**:

```
WMAE = Σ(w_i · |y_i − ŷ_i|) / Σ(w_i)
w_i = 5 if the week is a holiday week, else 1
```

Holiday weeks (Super Bowl, Labor Day, Thanksgiving, Christmas) therefore dominate the leaderboard metric.

### Challenges

- **Seasonal swings.** Sales move strongly with the retail calendar — especially Thanksgiving, Christmas, and the weeks around them.
- **Promotions / MarkDowns.** Temporary discounts affect departments unevenly; which series respond, and by how much, is hard to know a priori.
- **Holiday-weighted evaluation.** Mistakes on the four major holiday weeks cost **5×** as much as mistakes on ordinary weeks.
- **Limited holiday history.** Each holiday appears only once per year, so seasonal patterns have few positive examples.
- **Heterogeneous series.** Some Store×Dept pairs are long and smooth; others are short, sparse, or nearly constant.

### Goal

Train forecasting models that minimize WMAE on weekly sales, with validation schemes that stress holiday and seasonal regimes — not only the last few quiet weeks of the training window.

---

## Repository structure

```
walmart-recruiting-store-sales-forecasting/
├── README.md
├── preprocessing.ipynb          # shared EDA / cleaning exploration
├── features.ipynb               # shared feature exploration
│
├── # Tree / classical models
├── experiment-xgboost-v4.ipynb
├── experiment_light_v4.ipynb
├── experiment-RandomForest.ipynb
├── experiment-arima-v2.ipynb
├── experiment-SARIMA.ipynb
├── experiment-Prophet.ipynb
├── experiment-NeuralProphet.ipynb
│
├── # Neural forecasting models
├── experiment-DLinear.ipynb
├── experiment-NBEATS.ipynb
├── experiment_tft_v3.ipynb
└── experiment_patchtst_v1.ipynb
```

Older notebook revisions (`experiment-xgboost.ipynb`, `experiment-lightgbm-v2/v3`, `experiment-arima.ipynb`, `experiment-tft.ipynb`, …) are kept for history; the sections below describe the **current model versions**.

Competition data (`train.csv`, `test.csv`, `features.csv`, `stores.csv`, `sampleSubmission.csv`) is loaded from Kaggle / Colab secrets and is not committed here.

---

## MLflow / WandB links

| Service | Link / project |
|---|---|
| **GitHub** | https://github.com/lshek22/walmart-recruiting-store-sales-forecasting |
| **DagsHub + MLflow** | https://dagshub.com/lshek22/walmart-recruiting-store-sales-forecasting |
| **WandB project** | `ml-final-projekt-walmart-sales-forecasting` (used by TFT / PatchTST login; most notebooks log only via MLflow) |

Almost every experiment logs parameters, WMAE metrics, and artifacts through **MLflow on DagsHub** (`repo_owner=lshek22`). Typical experiment names:

| Model | MLflow experiment(s) |
|---|---|
| XGBoost | `XGBoost_v4_Cleaning`, `_Feature_Engineering`, `_Feature_Selection`, `_HPO`, `_Training` |
| LightGBM | `LightGBM_v4_*` (same stage pattern) |
| Random Forest | `RandomForest_Cleaning`, `_Feature_Selection`, `_Training` |
| ARIMA | `ARIMA_Training_v2` |
| SARIMA | `SARIMA_Training` |
| Prophet | `Prophet_Training` |
| NeuralProphet | `NeuralProphet_Training` |
| DLinear | `DLinear_Training` |
| N-BEATS | `NBEATS_Training` |
| TFT | `TFT_Training` |
| PatchTST | `PatchTST_Training` / `patchtst_v1` |

**Note on WandB:** DLinear and N-BEATS install `wandb` but do **not** log training curves there. TFT and PatchTST call `wandb.login` and set a project name; primary metrics still go to MLflow.

---

## Train / Val / Test split

**Test set** is always the official Kaggle `test.csv` horizon (weekly Fridays after the end of `train.csv`). Models never train on Kaggle test labels.

Validation differs by family:

### A. Holiday-aware 3-fold rolling origin
Used by **DLinear, N-BEATS, Prophet, SARIMA, NeuralProphet**.

| Fold | Validation window |
|---|---|
| **holiday** | 2011-11-01 → 2012-01-31 |
| **spring** | 2012-02-01 → 2012-04-30 |
| **late** | 2012-08-01 → 2012-10-31 |

Training uses all history **before** each fold start. Model selection minimizes **mean WMAE across the three folds**, so Christmas / Thanksgiving weeks are always represented in validation.

### B. Single walk-forward holdouts (tree / neuralforecast notebooks)

| Model | Validation window |
|---|---|
| **XGBoost v4** | 2012-01-27 → 2012-10-26 |
| **LightGBM v4** | 2011-10-01 → 2012-01-15 |
| **TFT v3 / PatchTST v1** | dates ≥ 2011-10-01 |
| **ARIMA v2** | last **8 weeks** of train (≈ 2012-09-07 → 2012-10-26) |

### C. TimeSeriesSplit (Random Forest)

**Random Forest** uses `TimeSeriesSplit(n_splits=4)` over unique weeks (not the Holiday/Spring/Late calendar folds).

---

## Feature engineering

Feature strategy depends on the model class:

### Shared preprocessing (most notebooks)
- Clip `Weekly_Sales` at **≥ 0**
- Fill MarkDown NaNs with **0**
- Merge `stores.csv` + `features.csv` on Store / Date
- Holiday weight **5** for WMAE and (where supported) sample weights

### Rich tabular FE (XGBoost, LightGBM, Random Forest)
- Store metadata: `Type`, `Size`
- Weather / economy: Temperature, Fuel_Price, CPI, Unemployment
- Calendar: year / month / week-of-year, sin/cos encodings, weeks elapsed
- Holiday flags and distances (signed distance; days-to / days-post per holiday)
- Lags (`1, 2, 4, 8, 52`, …), rolling means/std, YoY ratios
- Expanding target encodings with `shift(1)` (leakage-safe)
- MarkDown aggregates and interactions (holiday × MarkDown, Unemployment × MarkDown)
- Tree importance pruning before final fit

### Classical / additive time-series FE
- **ARIMA:** optional exogenous `IsHoliday` only
- **SARIMA / Prophet / NeuralProphet:** custom Walmart holiday table — `super_bowl`, `labor_day`, `thanksgiving`, `christmas`, **`pre_christmas`**
- Gap handling: per-series weekly grids (interior fill / interpolate); Prophet floors `y ≥ 1` for multiplicative seasonality

### Deep forecasting FE
- **DLinear / N-BEATS / PatchTST:** primarily **univariate** Store×Dept sales after `log1p` + per-series Min–Max (DLinear/N-BEATS) or NeuralForecast scaling
- **TFT:** full covariate wiring — **future** (calendar, holidays, weather, MarkDowns), **historical** (`PrevYearSales`), **static** (Store, Dept, Type, Size)

---

## Models

### XGBoost Training

**Notebook:** `experiment-xgboost-v4.ipynb`

Global `XGBRegressor` on the rich tabular feature set.

- **Objective:** `reg:absoluteerror` with holiday sample weights `{5, 1}`
- **HPO:** Optuna (~15 trials) over `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`
- **Inference:** recursive multi-step forecast on the Kaggle test horizon; blend with `lag_52`; Christmas week mass-shift post-process
- **Tracking:** stage-wise MLflow experiments under `XGBoost_v4_*`
- **Reported val WMAE (notebook):** blended ≈ **1688** (`α = 0.75`)

---

### LightGBM Training

**Notebook:** `experiment_light_v4.ipynb`

Same FE / recursive-forecast recipe as XGBoost, with `LGBMRegressor`.

- **Objective:** `regression_l1` + holiday sample weights
- **HPO:** Optuna (~15 trials) over `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`
- **Post-process:** `lag_52` blend + Christmas shift (same family as XGBoost)
- **Tracking:** `LightGBM_v4_*` on DagsHub
- **Reported val WMAE (notebook):** blended ≈ **2387** (`α = 0.30`)

---

### Random Forest Training

**Notebook:** `experiment-RandomForest.ipynb`

`RandomForestRegressor` with leakage-safe v2 FE (IQR clip from train only, signed holiday distances, expanding encodings).

- **Fit:** MSE + holiday `sample_weight`; select by CV WMAE
- **Search:** small grid (`max_depth`, `min_samples_leaf`, `max_features`; `n_estimators=300`)
- **Validation:** 4-fold `TimeSeriesSplit` on weeks
- **Tracking:** `RandomForest_*` MLflow experiments
- **Output:** `submission_randomforest.csv`

---

### ARIMA Training

**Notebook:** `experiment-arima-v2.ipynb`

Local **per Store×Dept** ARIMA / ARIMAX models (`statsmodels`).

- **Diagnostics:** ADF, ACF/PACF, seasonal decomposition on train only
- **Order screen:** manual grid `(0,1,0)…(2,1,2)` ± `IsHoliday` exog on a 50-series subset (≥80 weeks)
- **Validation:** last 8 train weeks
- **Fallbacks:** recent mean / global median for short or failed fits
- **Tracking:** `ARIMA_Training_v2`
- **Reported subset WMAE:** champion `ARIMA(2,1,2)` ≈ **2408** (ARIMAX+holiday was worse on that subset)

---

### SARIMA Training

**Notebook:** `experiment-SARIMA.ipynb`

Seasonal ARIMA with period **52**, upgraded to the shared **3 holiday-aware folds**.

- **Candidates:** `(0,1,1)×(0,1,1,52)`, `(1,1,1)×(0,1,1,52)`, `(1,1,1)×(0,1,0,52)`, plus SARIMAX with `IsHoliday` or Walmart event dummies
- **Selection:** mean fold WMAE on a series subset (`MIN_TRAIN_WEEKS=80`)
- **Submission:** per-series refit with checkpoint/resume; dept-median fallback; forecast cap at 5× historical max
- **Tracking:** `SARIMA_Training`
- **Internal champion (DagsHub):** `(0,1,1)×(0,1,1,52)`, mean fold val WMAE ≈ **2129**

---

### Prophet Training

**Notebook:** `experiment-Prophet.ipynb`

Local Prophet models with a custom Walmart holiday table and Optuna priors.

- **Preprocess:** per-series interior gap-fill (no global leading zeros); `y` floor = 1
- **HPO:** Optuna (~20 trials) over changepoint / seasonality / holiday priors, `n_changepoints`, yearly Fourier order, additive vs multiplicative
- **Validation:** Holiday / Spring / Late → mean WMAE (subset of series during search)
- **Tracking:** `Prophet_Training` (train & val WMAE per fold)
- **Kaggle (reported):** public ≈ **2734**, private ≈ **2829** (Prophet v4 submission)

---

### NeuralProphet Training

**Notebook:** `experiment-NeuralProphet.ipynb`

**Global** NeuralProphet (one model, many IDs) with AR-Net.

- **Design:** `n_lags=52`, yearly seasonality, optional Walmart event future regressors; `trend_global_local='global'`
- **Screen:** small config set (`base_ar`, `events`, `events_deep_ar`) on the holiday fold, then re-score all three folds
- **Stability / speed:** drop constant / short series; truncate history before predict; chunked prediction; series cap for Colab runs
- **Tracking:** `NeuralProphet_Training` (MLflow only; WandB removed)

---

### DLinear Training

**Notebook:** `experiment-DLinear.ipynb`

Univariate DLinear (trend + seasonal linear heads) trained globally on windowed Store×Dept series.

- **Preprocess:** weekly grid gap-fill(0) → `log1p` → per-series Min–Max (fit on train only)
- **HPO:** Optuna (~30 trials): `seq_len`, `pred_len`, kernel size, `const_init`, batch size, learning rate
- **Train:** L1 loss, Adam, 30 epochs, grad clip 1.0
- **Validation:** mean WMAE over Holiday / Spring / Late
- **Tracking:** `DLinear_Training` (MLflow; WandB not used for gradients)
- **Kaggle (reported):** public ≈ **2977**, private ≈ **3163** (DLinear v2 submission)

---

### N-BEATS Training

**Notebook:** `experiment-NBEATS.ipynb`

Univariate N-BEATS (MLP backcast/forecast stacks) with the same preprocess and folds as DLinear.

- **HPO:** Optuna (~25 trials): stacks, blocks, layers, width, dropout, lookback, horizon, batch, lr
- **Train:** L1, AdamW, 25 epochs; **max 10 windows / series** during search
- **Tracking:** `NBEATS_Training`
- **Kaggle (reported):** public ≈ **2779**, private ≈ **2879** (N-BEATS v2 submission)

---

### TFT Training

**Notebook:** `experiment_tft_v3.ipynb`

Temporal Fusion Transformer via **NeuralForecast**, with proper future / historical / static covariates.

- **Split:** train `< 2011-10-01`, validate on later weeks; final fit on full train; `h=60` for test
- **HPO:** sequential 1-D sweeps (not Optuna): `input_size`, `hidden_size`, `n_head`, `batch_size`, `lr`, `dropout` — with and without covariates
- **Loss:** MAE in training; WMAE for selection
- **Tracking:** `TFT_Training` + WandB login (`ml-final-projekt-walmart-sales-forecasting`)
- **Note:** covariates help in some configs but can also hurt; target-only baselines remain important references

---

### PatchTST Training

**Notebook:** `experiment_patchtst_v1.ipynb`

Patch Time Series Transformer (NeuralForecast), treated as a **univariate** global multi-series model.

- **Same val date** as TFT (`≥ 2011-10-01`)
- **HPO:** 1-D sweeps over `input_size`, `patch_len`, `batch_size`, `lr`, `dropout`; longer `max_steps` runs for final candidates
- **Tracking:** `PatchTST_Training` / `patchtst_v1` + WandB login
- **Reported val WMAE (notebook):** best printed runs down to ≈ **2234**

---

## Metric summary

| Model | Typical selection metric | Notes |
|---|---|---|
| XGBoost v4 | Val WMAE ≈ 1688 (blended) | Strongest tabular baseline in-notebook |
| LightGBM v4 | Val WMAE ≈ 2387 (blended) | Same recipe, different val window |
| PatchTST | Val WMAE ≈ 2234 | Univariate transformer |
| SARIMA | Mean fold WMAE ≈ 2129 | Seasonal local models |
| N-BEATS | Kaggle public ≈ 2779 | Shared 3-fold recipe |
| Prophet | Kaggle public ≈ 2734 | Holiday table + Optuna |
| DLinear | Kaggle public ≈ 2977 | Shared 3-fold recipe |
| TFT | Val sweeps ~3140–4500 | Sensitive to covariate setup |
| ARIMA | Subset WMAE ≈ 2408 | Non-seasonal local baseline |
| Random Forest / NeuralProphet | Runtime CV | See DagsHub runs |

Exact leaderboard numbers depend on the submission file version; always prefer the latest DagsHub / Kaggle entry for a given notebook.

---

## How to run

1. Attach Kaggle competition data (or set `KAGGLE_API_TOKEN` on Colab).
2. Optional secrets: `DAGSHUB_USER_TOKEN`, `WANDB_API_KEY` (TFT / PatchTST).
3. Open the target `experiment-*.ipynb`.
4. Prefer **GPU** for DLinear, N-BEATS, TFT, PatchTST, NeuralProphet; **CPU** is fine for Prophet, SARIMA, ARIMA, trees.
5. Use each notebook’s `FAST_RUN` (where present) for a smoke test before a full Optuna / sweep run.
6. Upload the generated `submission_*.csv` to Kaggle.

---

## Authors

Collaborative ML course project — notebooks cover complementary model families (gradient boosting, classical time series, Prophet-style decompositions, and deep forecasters), all tracked toward the same WMAE objective.
