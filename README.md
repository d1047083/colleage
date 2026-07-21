# 資料科學與 AI 作品集

這裡放的是我大學幾門課的作業。後來我把它們重做、補齊，整理成六個可以實際跑的小專案，內容大致落在金融資料、電腦視覺、語音處理，還有一點模擬。

會做這件事有幾個原因：之後申請國外學校時手上要有能拿出來的東西；我自己想試著用資料分析做點金融的觀察；再來就是把以前學一半的部分補起來。

## 專案

| # | 專案 | 做了什麼 | 主要用到 | 狀態 |
|---|------|----------|----------|------|
| 01 | 央行匯率分析與預測 | GARCH 波動度、月頻搭總經指標預測台幣、含成本的回測，並檢查前視偏誤 | pandas, scikit-learn, arch | 有結果 |
| 02 | 影像去模糊 | 傳統方法的 PSNR/SSIM 基準，加上一份 U-Net 訓練程式 | scikit-image, Keras | 傳統法有結果 |
| 03 | 水果好壞分類 | 香蕉 good/bad 的 MobileNetV2 遷移學習 | TensorFlow/Keras | 可訓練骨架 |
| 04 | 語音訊號分析 | 用真實錄音做 MFCC、基頻、ICA 盲訊號分離、關鍵詞辨識 | NumPy, SciPy, librosa | 有結果 |
| 05 | 自動駕駛場景 | 用 Python 從頭解析 OpenDRIVE，把 RoadRunner 的道路網重畫出來 | OpenDRIVE, XML | 有結果 |
| 06 | 多鹿種偵測 | YOLOv8 分辨五種鹿，資料抓自 iNaturalist，已打包能在本地跑 | YOLOv8, ultralytics | mAP50 約 0.34 |

## 怎麼跑

```bash
pip install -r requirements.txt

# 匯率分析(需指到央行匯出的 json)
TWCB_JSON=/path/to/download_TWCB.json python 01-fintech-fx-forecasting/src/build.py

# 語音分析(不給檔案就用內建的合成訊號)
python 04-speech-dsp-python/src/speech_features.py [your.wav]

# 傳統去模糊(不給圖就用內建的合成模糊圖)
python 02-cv-image-deblur/src/classical_deblur.py [your_image.png]
```

## 我寫這些時守的幾件事

有結果就一定跟一個基準比。匯率預測會跟「隔天不變」的隨機漫步比，分類會看混淆矩陣，不然數字再漂亮也看不出好壞。

數字難看的時候照實寫。匯率其實很難預測，日頻幾乎就是隨機漫步，我就直接寫出來，沒有硬湊一個好看的結果。

盡量能重跑。亂數種子固定、路徑用相對的、套件版本記在 requirements 裡。

每個專案接下來想怎麼往下做，寫在 [ROADMAP.md](ROADMAP.md)。
