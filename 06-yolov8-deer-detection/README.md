# 06 · YOLOv8 鹿種偵測 (Deer Species Detection)

用 YOLOv8 做鹿的**偵測**(框出每隻鹿並辨識種類)。接續自駕/電腦視覺主題，練習物件偵測全流程。

## ⚠️ 執行環境
YOLOv8 訓練需要 GPU 且需下載預訓練權重/資料集。**本專案的訓練請在你本機(有 GPU)或 Google Colab 執行**
(建置此專案的雲端沙箱無 GPU 且封鎖權重下載，僅用合成資料驗證過訓練流程可運作)。

## ✅ 已實跑驗證的成果 (真實資料)
`src/run_openimages_demo.py` 用 **Open Images「Deer」類別**(免 API key)下載真實鹿圖並微調 YOLOv8n，
已在無 GPU 環境實際跑出：**mAP@0.5 = 0.772、Recall = 0.955**(120 訓練/40 驗證、CPU、10 epoch)。
成果圖見 `figures/`(deer_detections / deer_train_curves / deer_confusion)、數據見 `results/openimages_metrics.md`。
```bash
pip install fiftyone ultralytics huggingface_hub
python src/run_openimages_demo.py --epochs 30 --imgsz 640 --device 0   # 有GPU時
```

## 最快路徑：Colab (免費 GPU)
打開 `notebooks/train_deer_yolov8_colab.ipynb` → 執行階段設為 GPU → 由上而下執行即可。
資料集提供兩條路：**A. Open Images(免 API key，已驗證)** 或 **B. Roboflow(需 API key)**。

## 本機路徑
```bash
pip install ultralytics roboflow
# 1) 下載資料(填入你的 Roboflow API key)
python src/download_dataset.py
# 2) 訓練
python src/train.py --data ../dataset/data.yaml --model yolov8n.pt --epochs 100 --device 0
# 3) 推論
python src/predict.py --weights runs/detect/deer_yolov8n/weights/best.pt --source some_imgs/
```

## 關於「鹿種」資料
公開的 deer 偵測資料集多半是**單類別(deer)**。要做真正的**鹿種(species)**辨識，兩條路：
1. 找 species 標註的資料集(Roboflow Universe 搜尋，注意 classes 是否含多種鹿)。
2. 自行標註：用 Roboflow Annotate / LabelImg / CVAT 對你的鹿照片畫框標種類，匯出 YOLO 格式。
   `dataset/data.yaml` 已給多鹿種(梅花鹿/紅鹿/水鹿/山羌)的 names 範本，依你的資料調整。

## 檔案
- `src/train.py` 訓練 · `src/predict.py` 推論 · `src/download_dataset.py` Roboflow 下載
- `dataset/data.yaml` 資料設定範本 · `notebooks/*.ipynb` Colab 一鍵訓練

## 評估指標
物件偵測看 **mAP50 / mAP50-95**、Precision/Recall 與混淆矩陣(YOLOv8 訓練後自動產生於 runs/)。
