# 01 · 央行匯率分析與預測 (旗艦)

用台灣央行統計資料庫的每日匯率(1993–，9 種貨幣)做探索分析與**次日匯率預測**，
重點是**與隨機漫步基準誠實對照**。

## 執行
```bash
TWCB_JSON=/path/to/download_TWCB.json python src/build.py
```
產出：`data/fx_daily.csv`、`figures/*.png`、`results/metrics.md`

## 方法
- 特徵：多階報酬 lag、移動平均、滾動波動度、星期幾
- 模型：Ridge、Gradient Boosting，預測「明日對數報酬」再還原為價位
- 驗證：時間序 70/30 切分(不洩漏未來)
- 指標：RMSE、MAE、**方向準確率**

## 結論(誠實)
匯率近似 martingale，**隨機漫步基準的 RMSE 最低**，各模型方向準確率都在 ~50%。
如實呈現「難以預測」正是這個專案的價值——它示範了正確的評估方法與科學誠信。
延伸：見根目錄 ROADMAP.md(加總經特徵、GARCH 波動度、含成本回測)。
