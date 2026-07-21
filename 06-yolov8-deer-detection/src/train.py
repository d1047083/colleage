"""
YOLOv8 鹿偵測的訓練腳本，在有 GPU 的機器或 Colab 上跑。
資料是 YOLO 偵測格式，見 dataset/data.yaml。第一次會自動下載預訓練權重，需要連網。
"""
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../dataset/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m/l/x.pt 或 .yaml(從零)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' GPU / 'cpu'")
    ap.add_argument("--name", default="deer_yolov8n")
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        name=a.name, patience=20, seed=42,
        # 資料增強(對小資料集尤其重要)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=5.0, translate=0.1,
        scale=0.5, fliplr=0.5, mosaic=1.0, mixup=0.1,
    )
    metrics = model.val()                       # 在驗證集評估
    print("mAP50-95:", metrics.box.map, "| mAP50:", metrics.box.map50)
    print("最佳權重: runs/detect/%s/weights/best.pt" % a.name)

if __name__ == "__main__":
    main()
