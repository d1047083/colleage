# Speech — Deep-dive Results

- Audio: real wavs `star.wav`, `happy.wav` @ 16000 Hz

## A. MFCC / Mel-spectrogram

- Extracted 13-dim MFCC, shape (13, 626) (frames × coeff) — the standard front-end for ASR/keyword spotting.

## B. Fundamental frequency (pyin)

- Voiced frames: 98.4% · median f0 ≈ **263 Hz**

## C. ICA blind source separation (cocktail party)

- Mixed two real voices into 2 mics, then FastICA unmixed them.
- Recovery correlation with originals: **1.00**, **1.00** (1.0 = perfect).
- 分離後的音檔存在 results/ica_recovered_*.wav，可直接聽出兩人聲音被拆開。

延伸：MFCC 接一個小分類器即可做關鍵詞辨識；ICA 可延伸到多麥克風陣列與去噪。
