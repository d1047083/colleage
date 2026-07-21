# 04 · 語音訊號分析

把以前 MATLAB 寫的 ZCR、短時能量、頻譜移植成 Python，順便做了簡單的語音/靜音偵測。

```bash
python src/speech_features.py [your.wav]   # 不給檔案就用內建的合成訊號
```

會畫出波形、短時能量、ZCR 三張圖。

另外用真實錄音做了幾件事，程式在 `src/speech_deep.py` 和 `src/keyword_spotting.py`：MFCC 特徵、基頻追蹤、ICA 盲訊號分離（把兩段混在一起的人聲拆開），還有用 MFCC 加 DTW 做的關鍵詞辨識。
