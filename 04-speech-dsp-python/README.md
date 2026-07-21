# 04 · 語音訊號分析 (Speech DSP, Python)

把 MATLAB 的 ZCR / 短時能量 / 頻譜移植成 Python，並做基礎的語音/靜音偵測(VAD)。

```bash
python src/speech_features.py [your.wav]   # 不給檔案會用合成測試訊號
```
產出三聯圖(波形 / 短時能量 / ZCR)。延伸：MFCC + 關鍵詞辨識、PCA-ICA 盲訊號分離。
