# Keyword spotting — MFCC + DTW

- 詞庫(真實 wav): jarvus, hello3, bug, bath, star, happy, tiger
- 每詞取能量最高的 1.5s 核心片段為模板；測試查詢 = 3 SNR × 3 變速 = 9 種/詞
- **Top-1 辨識準確率 = 95.2%** (共 63 次查詢)

## 解讀
MFCC 捕捉音色包絡、DTW 容忍語速伸縮，因此在加噪與變速下仍能辨識——這是 one-shot(每詞僅一模板)的作法，適合資料稀少情境。
延伸：資料變多時可改用 CNN/RNN 端到端關鍵詞偵測(如 Google Speech Commands)。
