# walmart-recruiting-store-sales-forecasting

## ARIMA — Walmart Store Sales Forecasting

This module covers the **ARIMA / ARIMAX** approach to the [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) competition. It's one of several modeling approaches in this project — see the top-level project README for how it compares to the others.

## Notebook

`experiment-arima-with-acf-and-submission.ipynb`

## Problem framing

The competition asks for weekly sales forecasts for every `(Store, Dept)` combination, evaluated with **Weighted MAE (WMAE)**, where holiday weeks get 5x the weight of normal weeks:

```
WMAE = Σ(weight_i * |actual_i - predicted_i|) / Σ(weight_i)
weight_i = 5 if IsHoliday else 1
```

There are **3,331 unique `(Store, Dept)` series**, each a separate weekly time series. ARIMA is fit independently per series rather than as one global model.

## Pipeline

1. **Load & merge** — `train.csv`, `test.csv`, `stores.csv`, `features.csv` merged on `Store`/`Date`; markdown columns filled with 0; negative sales clipped to 0.
2. **Stationarity check** — Augmented Dickey-Fuller test on a sample series (Store 1, Dept 1) at the raw level, 1st difference, and 2nd difference, plus a 50-series sample to estimate how many series are stationary at the raw level (in general, most are not — retail sales trend and have strong yearly seasonality).
3. **Autocorrelation diagnostics** — ACF and PACF plots (both a manual bar-chart version with 95% confidence bands, and the standard `statsmodels.plot_acf` / `plot_pacf`) on the differenced series, used to reason about AR (p) and MA (q) order. A seasonal decomposition (`period=52`) is also plotted to show trend/seasonal components.
4. **Train/validation split** — last 8 weeks of `train.csv` held out as validation; everything before that is training.
5. **Single-series baseline** — `ARIMA(1,1,1)` fit on Store 1 / Dept 1, forecast against the holdout, scored with WMAE.
6. **Order search** — `(0,1,0)`, `(1,1,0)`, `(0,1,1)`, `(1,1,1)`, `(2,1,2)` compared by AIC/BIC/WMAE on that same series.
7. **ARIMAX** — `IsHoliday` added as an exogenous regressor, compared against the equivalent plain ARIMA.
8. **Scale-up to 50 series** — a random sample of 50 `(Store, Dept)` pairs (≥80 weeks of history) used to compare `ARIMA(1,1,1)`, `ARIMA(2,1,2)`, `ARIMA(1,1,0)`, and the best order + `IsHoliday` exog, by aggregate WMAE. The best-scoring config becomes the "champion" configuration.
9. **Full submission** — champion order/exog config refit on the **full** history of every `(Store, Dept)` pair present in `test.csv`, forecast forward, clipped at 0, written to `submission.csv` in Kaggle's `Id,Weekly_Sales` format.

## Handling edge cases

- Series with fewer than 20 weeks of history are not fit; they fall back to the mean of the last 4 known weeks.
- `(Store, Dept)` pairs with no training history at all (cold start) fall back to the global median of `Weekly_Sales`.
- Any series where `ARIMA.fit()` raises an exception (non-convergence, singular matrix, etc.) also falls back to the recent-average / global-median rule above.

## Experiment tracking

Runs are logged to MLflow (via DagsHub) under experiment `ARIMA_Training_v2`, including:
- Order comparison metrics (AIC, BIC, WMAE) per configuration
- The ARIMA vs ARIMAX comparison
- The 50-series subset WMAE for each candidate order
- The final champion model, registered as `ARIMA_Walmart_Forecaster`
- The generated `submission.csv` as a run artifact

## Known limitations

- **Per-series independence**: each `(Store, Dept)` gets its own model with no information sharing across series — no pooling of related departments/stores, unlike a global gradient-boosted or deep learning model.
- **No explicit yearly seasonality term**: ARIMA orders used here are non-seasonal (`d`-order differencing only); a `SARIMAX` seasonal order (e.g. `(1,1,1)x(1,1,1,52)`) is imported but not used, and would likely better capture the strong 52-week retail cycle (holidays, back-to-school, etc.).
- **Order search is per single series**, then generalized to all series — the champion order may not be optimal for every individual `(Store, Dept)` pair.
- **Runtime**: fitting ~3,300 individual ARIMA models for the full submission takes a non-trivial amount of time (tens of minutes on CPU).

## Output

- `submission.csv` — ready to upload to Kaggle.
- Diagnostic plots saved as PNGs (`arima_single_series_forecast.png`, `arima_vs_arimax.png`, `arima_order_comparison.png`, `arima_all_runs.png`, `acf_pacf_official.png`) and logged as MLflow artifacts.
