# 鹿偵測 — 實跑成果 (Open Images, 真實資料)

用 `src/run_openimages_demo.py` 在**無 GPU 的環境**實際跑出(非模擬)：

- 資料：Open Images V7 「Deer」類別，**120 張訓練 / 40 張驗證**(自動下載，含 bounding box)
- 模型：YOLOv8n，自 HuggingFace 取得預訓練權重後**微調**
- 設定：CPU、10 epochs、imgsz 320(受限於無 GPU 的時間；有 GPU 應調高)

| 指標 | 數值 |
|---|---|
| mAP@0.5 | **0.772** |
| mAP@0.5:0.95 | 0.465 |
| Recall | 0.955 |

成果圖(figures/)：deer_detections.png(偵測結果)、deer_train_curves.png(訓練曲線)、deer_confusion.png(混淆矩陣)。

## 誠實解讀
- **定位(框位置)已相當準**，mAP50=0.77 對如此少的訓練量與 epoch 是不錯的結果。
- **信心分數偏低**(框 conf 多在 0.01~0.05)：CPU 僅 10 epoch 未充分收斂的正常現象；
  有 GPU 跑 30~100 epoch、imgsz 640 後會校準得更高、可用較高門檻乾淨輸出。
- 這裡是**單類別(Deer)**；細分鹿種需 species 標註資料集。
