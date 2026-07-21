# 01 · 央行匯率分析與預測

用台灣央行的每日匯率（1993 年到現在，九種貨幣）做分析，並試著預測隔天的台幣匯率。這個題目的重點其實不是「預測多準」，而是有沒有拿一個像樣的基準來比。

## 跑法

```bash
TWCB_JSON=/path/to/download_TWCB.json python src/build.py
```

會產生 `data/fx_daily.csv`、`figures/` 裡的圖、還有 `results/metrics.md`。

## 做法

特徵用了幾階的報酬 lag、移動平均、滾動波動度、星期幾。模型是 Ridge 和 Gradient Boosting，預測的是隔天的對數報酬，再換算回價位。切分用時間序的 70/30，不讓模型看到未來。指標看 RMSE、MAE，還有方向猜對的比例。

深化的部分（GARCH、月頻總經模型、回測）在 `src/deep_analysis.py` 和 `src/walkforward_validation.py`。

## 結果

日頻匯率基本上就是隨機漫步，「隔天等於今天」這個最笨的基準 RMSE 反而最低，各模型猜方向的比例都在 50% 上下。這其實是匯率研究裡很常見的結論，把它照實寫出來，比硬做一個看起來會賺的預測器有意義。

不過換到月頻、加進央行的總經指標之後就不一樣了：方向準確率到 65% 左右，而且用擴張窗的 walk-forward 驗證過還站得住。這條結果比較值得往下追，細節在深化的那兩支程式和 `results/` 裡。
