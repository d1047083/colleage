# 林信廷 · 資料科學與 AI 作品集 (Data Science & AI Portfolio)

把大學時期的課程作業，系統性地升級為五個可執行、可展示的 side project。
主軸涵蓋 **金融資料科學、電腦視覺、語音訊號處理、機器學習與模擬**。

> 定位：同時作為 (1) 出國留學作品集、(2) 金融投資的實驗場、(3) 技術深造練功房。
> 每個子專案都遵循相同原則——**誠實呈現結果、有基準對照、程式可重現**。

## 專案總覽

| # | 專案 | 一句話 | 技術棧 | 狀態 |
|---|------|--------|--------|------|
| 01 | **央行匯率分析與預測** (旗艦) | GARCH波動度 + 月頻總經預測(方向67.6%, p<0.001) + 含成本回測(Sharpe 1.30) + 前視偏誤檢查 | pandas, scikit-learn, arch(GARCH) | ✅ 深化, 真成果 |
| 02 | **影像去模糊** | PSNR/SSIM量化基準(傳統法) + U-Net+感知損失訓練程式 | scikit-image, Keras | ✅ 量化基準 / U-Net程式 |
| 03 | **水果好壞影像分類** | 香蕉 good/bad 用 MobileNetV2 遷移學習做正規二元分類 | TensorFlow/Keras | 🟡 可訓練骨架 |
| 04 | **語音訊號分析** | 真實wav的MFCC/基頻 + ICA盲訊號分離(兩段人聲分離, 相關1.00) | NumPy, SciPy, librosa | ✅ 深化, 真成果 |
| 05 | **自動駕駛場景模擬** | Python從零解析OpenDRIVE(.xodr), 重建道路網(138路/10路口/4.63km) | OpenDRIVE, XML | ✅ 深化, 真成果 |
| 06 | **YOLOv8 多鹿種偵測** | 辨識5鹿種(白尾/紅/梅花/黇/馴鹿), iNaturalist真實照片+自動標框, 已打包可本地跑 | YOLOv8, ultralytics | ✅ 實跑 mAP50=0.34 |

## 快速開始

```bash
pip install -r requirements.txt

# 旗艦：匯率分析(需指向央行 json)
TWCB_JSON=/path/to/download_TWCB.json python 01-fintech-fx-forecasting/src/build.py

# 語音分析(可直接跑合成測試訊號)
python 04-speech-dsp-python/src/speech_features.py [your.wav]

# 傳統去模糊(可直接跑合成測試圖)
python 02-cv-image-deblur/src/classical_deblur.py [your_image.png]
```

## 設計原則
1. **有基準才有意義**——匯率預測一定跟「隨機漫步」比；分類一定看混淆矩陣。
2. **不誇大**——FX 次日方向準確率就在 50% 附近，如實寫出來反而是專業的展現。
3. **可重現**——固定亂數種子、相對路徑、requirements 鎖定。

詳細的每個專案升級規劃見 [`ROADMAP.md`](ROADMAP.md)。
