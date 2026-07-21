# 06 · 多鹿種偵測

用 YOLOv8 偵測鹿，並分出品種，不只是判斷「是不是鹿」。目前分五種：白尾鹿、紅鹿、梅花鹿、黇鹿、馴鹿。

這個資料夾是打包好、可以直接在本地跑的：照片、標註、訓練好的權重、程式都在裡面。

## 內容

```
species_data/deer_species_yolo/   225 張鹿照片(來自 iNaturalist)、YOLO 標註、data.yaml
weights/
  deer_species_best.pt            訓練好的多鹿種模型，可直接拿來偵測
  generic_deer_detector.pt        通用鹿偵測器，用來自動標框、擴充資料
src/
  predict.py                      用權重對新圖偵測
  train.py                        訓練
  build_species_dataset.py        從 iNaturalist 重建或擴充資料集
notebooks/train_deer_yolov8_colab.ipynb
results/species_metrics.md
```

## 在本地跑

```bash
pip install ultralytics fiftyone huggingface_hub

# 用附的權重，對你的圖偵測鹿種
python src/predict.py --weights weights/deer_species_best.pt --source 你的圖片資料夾/ --conf 0.25 --device 0

# 想自己重訓(在這個資料夾底下執行)
python src/train.py --data species_data/deer_species_yolo/data.yaml \
                    --model weights/deer_species_best.pt --epochs 100 --imgsz 640 --device 0

# 想擴充資料：每種多抓一點再重訓
python src/build_species_dataset.py --per 200
```

## 現在的狀況

225 張圖（每種 45 張），在沒有 GPU 的機器上訓練，mAP@0.5 大概 0.34，五種鹿都分得出來（見 `figures/deer_species_detections.png`，馴鹿最好認）。

準確度不高有幾個原因，講清楚比較好：框是用通用偵測器自動標的，本身有雜訊；每種才 45 張，量很少；而且是 CPU 中途停下來的。鹿的長相彼此又很像，尤其母鹿和小鹿，分品種本來就比「是不是鹿」難不少。

要往上拉，順序大概是：每種抓到 200 張以上（`build_species_dataset.py --per 200`），花點時間用 LabelImg 或 Roboflow 把框和品種手動校正，再用 GPU 跑個 100 到 150 epoch。整條流程是通的，資料和算力補上去就會好很多。
