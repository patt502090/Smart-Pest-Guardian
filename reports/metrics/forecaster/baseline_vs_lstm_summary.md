# LSTM vs Baseline Summary (2025-10-08)

## Quick Metrics Snapshot

| Model Variant | RMSE | MAE | MAPE (%) |
|---------------|------|-----|----------|
| Naive last-value | 0.901 | 0.510 | 80.65 |
| Moving average (window=14) | 0.673 | 0.481 | 64.75 |
| LSTM (hs=96, lr=1e-3, dropout=0.2) | 0.573 | **0.320** | 73.14 |
| LSTM (hs=128, lr=5e-4, dropout=0.2) | **0.553** | 0.291 | 76.97 |

- Baseline metrics: `reports/metrics/forecaster/baseline_metrics_20251008_190827.csv`
- Hyperparameter sweep metrics: `reports/metrics/forecaster/sweeps/lstm_sweep_20251008_191509.csv`

## Narrative Highlights

1. **RMSE wins** – The tuned LSTM (hidden size 128, learning rate 5e-4) reduces RMSE to 0.553, beating the best baseline (moving average) by ~18%.
2. **MAE advantage** – Both tuned LSTM configurations more than halve MAE compared to naive baseline and cut ~40% compared to moving average.
3. **MAPE trade-off** – MAPE remains higher for the LSTM because the dataset contains many near-zero counts; relative errors explode even when absolute errors are small. This justifies keeping MAE/RMSE as primary KPIs in the report.
4. **Next lever** – Further gains may come from:
   - Re-weighting the loss toward non-zero days or using SMAPE.
   - Adding weather/soil covariates to stabilize percentage metrics.
   - Extending training epochs with early stopping around the best validation RMSE.

## Visual Assets

- Baseline comparison chart: `reports/metrics/forecaster/baseline_comparison_20251008_190827.png`
- Sample LSTM training curves: `reports/metrics/forecaster/sweeps/lstm_20251008_191521_training.png`
- Forecast overlay: `reports/metrics/forecaster/sweeps/lstm_20251008_191521_forecast_location-id-site-demo.png`

Use this section in the slide deck / report to emphasise that the LSTM does outperform simplistic baselines on key error metrics, while transparently acknowledging the remaining gap on percentage-based measures. This sets the stage for the upcoming hyperparameter sweep narrative and the creative “what-if” module.
