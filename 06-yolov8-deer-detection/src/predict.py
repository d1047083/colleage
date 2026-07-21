"""
YOLOv8 鹿種偵測 — 推論/視覺化
用訓練好的權重對新影像做偵測，輸出畫框結果。
用法:
  python predict.py --weights runs/detect/deer_yolov8n/weights/best.pt --source path/to/imgs --conf 0.25
"""
import argparse, os
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="影像檔、資料夾、或影片")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    model = YOLO(a.weights)
    results = model.predict(source=a.source, conf=a.conf, device=a.device,
                            save=True, line_width=2)
    for r in results:
        names = r.names
        cnt = {}
        for c in r.boxes.cls.tolist():
            cnt[names[int(c)]] = cnt.get(names[int(c)], 0) + 1
        print(os.path.basename(r.path), "->", cnt or "no detection")
    print("畫框結果已存於 runs/detect/predict*/")

if __name__ == "__main__":
    main()
