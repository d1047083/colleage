# 06 · YOLOv8 多鹿種偵測 (Deer **Species** Detection)

用 YOLOv8 偵測並**辨識鹿的品種**（不只判斷「是不是鹿」）。目前 5 個鹿種：
白尾鹿 (white_tailed_deer)、紅鹿 (red_deer)、梅花鹿 (sika_deer)、黇鹿 (fallow_deer)、馴鹿 (reindeer)。

本專案是**打包好、可在本地端直接跑**的：已附真實照片資料集、訓練好的權重、與完整程式。

## 內容
```
06-yolov8-deer-detection/
├─ species_data/deer_species_yolo/   # 225 張真實鹿照(來自 iNaturalist) + YOLO 標註 + data.yaml
├─ weights/
│   ├─ deer_species_best.pt          # 訓練好的多鹿種模型(可直接推論)
│   └─ generic_deer_detector.pt      # 通用鹿偵測器(用於自動標框、擴充資料)
├─ src/
│   ├─ predict.py                    # 用權重對新圖偵測
│   ├─ train.py                      # 訓練
│   └─ build_species_dataset.py      # 從 iNaturalist 重建/擴充資料集(自動標框)
├─ notebooks/train_deer_yolov8_colab.ipynb
└─ results/species_metrics.md        # 實跑成果與誠實說明
```

## 在本地端跑（三步）
```bash
# 0) 安裝（建議有 NVIDIA GPU）
pip install ultralytics fiftyone huggingface_hub

# 1) 直接用附上的權重，對你的圖片偵測鹿種
python src/predict.py --weights weights/deer_species_best.pt --source 你的圖片資料夾/ --conf 0.25 --device 0

# 2) （可選）用附上的資料集自己重新訓練（在專案根目錄執行）
python src/train.py --data species_data/deer_species_yolo/data.yaml \
                    --model weights/deer_species_best.pt --epochs 100 --imgsz 640 --device 0

# 3) （可選）擴充資料：每種抓更多張、自動標框，再重訓
python src/build_species_dataset.py --per 200
```

## 目前成果（誠實）
在無 GPU 環境用 225 張(180/45)、CPU 中途訓練得到 **mAP@0.5 ≈ 0.34**，能區分五個鹿種
（見 `figures/deer_species_detections.png`，每種各一例；馴鹿最好認）。這是**弱監督**結果：
bounding box 由自動標框產生、本身有雜訊，且資料量小。**要顯著提升**：
1. 每種抓 200+ 張（`build_species_dataset.py --per 200`）
2. 人工用 LabelImg/Roboflow 校正框與品種
3. 用 GPU 跑 100~150 epoch、imgsz 640

鹿種外觀相近（尤其雌鹿/幼鹿），比「是不是鹿」難很多，這是正常的難度。整條**物種級偵測流程**已打通，
資料與算力補上去就能得到堪用模型。

## Colab（免費 GPU）
`notebooks/train_deer_yolov8_colab.ipynb` 也可用；把資料改成本專案 species 版即可在 GPU 上快速訓練。
