# FX — Deep-dive Results

## A. GARCH(1,1) volatility

- Persistence α+β = **0.885** (close to 1 → strong volatility clustering / long memory)
- 1-day-ahead vol forecast vs realized |return| correlation = **0.607**
- 解讀：方向不可預測，但**波動度可預測**——這正是選擇權定價與風險管理的基礎。

## B. Monthly macro model

- Random-walk RMSE **0.0093** vs RandomForest RMSE **0.0085**
- Directional accuracy **67.6%** (71/105), binomial test vs 50% p = **0.000**
- Most important feature: **ret** (importance 0.36)
- 解讀：月頻加入總經基本面後，方向準確率**在統計上顯著優於擲硬幣**(p<0.05)，顯示基本面確有邊際資訊。
- **前視偏誤檢查**：月頻總經數據隔月才公布，故將基本面特徵延遲一個月(MACRO_LAG=1)後重跑；優勢仍存在(天真版與修正版差異極小)，代表結果不是單純偷看未來造成的。
- **仍需保留的疑慮**(誠實)：67% 的月方向準確率明顯高於 FX 文獻常見的 55–58%，且此處僅用單一 70/30 切分。要確認這個訊號為真，下一步應做**擴張窗 walk-forward**、多幣別/多期間穩健性測試，並排查外匯存底等變數的殘留同步性。在那之前，把它當『值得深究的訊號』而非『已驗證的獲利策略』。

## C. Trading backtest (out-of-sample, 0.1% cost)

- Strategy total return **39.4%** (Sharpe 1.30)
- Buy&hold USD **-5.9%** (Sharpe -0.20)
- 解讀：回測含交易成本並用樣本外資料，避免前視偏誤；Sharpe 才是公平比較基準。

## 這次深化把專案從『baseline』推進到『研究等級』的關鍵

1. 從『日頻近似隨機漫步』延伸到**月頻總經預測**，問對了問題。
2. 用 **GARCH** 展示『波動度可預測』這個 FX 中最實用的可預測性。
3. 用 **binomial / Sharpe / 樣本外回測** 做統計嚴謹的評估，而非只看 RMSE。
