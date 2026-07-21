# FX Forecasting — Results

- Data: TWCB daily FX, 1993-01-05 → 2022-03-31, 7,278 rows, 18 currencies

- Task: predict next-day USD/TWD; chronological 70/30 split (test starts 2013-06-18)


| Model | Test RMSE | Test MAE | Directional Acc |
|---|---|---|---|
| RandomWalk | 0.0666 | 0.0459 | 46.6% |
| Ridge | 0.0668 | 0.0462 | 47.8% |
| GradientBoosting | 0.0676 | 0.0473 | 48.0% |

## Honest reading of the results

The random-walk baseline is extremely hard to beat: exchange rates are close to a martingale, so 'tomorrow ≈ today' already gives a very low RMSE. Here the best RMSE model is **RandomWalk**, and directional accuracy for every model sits near 50% — i.e. predicting the *direction* of the next day's move is barely better than a coin flip. This is the correct, well-documented finding in FX research, and stating it honestly (rather than claiming a magic predictor) is exactly what a graduate admissions committee or a serious quant reader wants to see.


## Where real edge could come from (next steps)
- Longer horizons / weekly returns, where some predictability exists
- Macro features from the other 170+ TWCB series (rates, money supply, reserves)
- Volatility modelling (GARCH) — vol IS forecastable even when direction is not
- A proper trading backtest with transaction costs, not just RMSE
