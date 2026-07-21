# 單類別鹿偵測結果（Open Images）

用 src/run_openimages_demo.py 在沒有 GPU 的機器上實際跑出來的，不是模擬。資料是 Open Images V7 的 Deer 類別，120 張訓練、40 張驗證（自動下載，含框）。模型是 YOLOv8n，權重從 HuggingFace 抓下來後微調。設定是 CPU、10 epoch、imgsz 320，這是被沒 GPU 的時間限制住的，有 GPU 應該調高。

| 指標 | 數值 |
|---|---|
| mAP@0.5 | 0.772 |
| mAP@0.5:0.95 | 0.465 |
| Recall | 0.955 |

圖在 figures/：deer_detections.png（偵測結果）、deer_train_curves.png（訓練曲線）、deer_confusion.png（混淆矩陣）。

框的位置其實抓得蠻準，mAP50 0.77 對這麼少的訓練量和 epoch 算不錯。信心分數偏低（框的 conf 多在 0.01~0.05），是 CPU 只跑 10 epoch 沒收斂完的正常現象，有 GPU 跑個 30 到 100 epoch、imgsz 640 之後會校準得高一點，就能用比較高的門檻輸出乾淨的結果。這裡是單一類別的鹿，要細分品種得用有物種標註的資料集。
