"""
可重現的鹿偵測示範，不需要 Roboflow API key。
用 Open Images 的 Deer 類別（含框）下載真實鹿圖，轉成 YOLO 格式，微調 YOLOv8n。
本專案的單類別成果就是這支跑出來的。有 GPU 時把 epochs 和 imgsz 調高。
"""
import argparse, os, glob
import fiftyone as fo, fiftyone.zoo as foz
from fiftyone import ViewField as F

def get_weights():
    """優先用本地/官方權重；若下載被擋，改從 HuggingFace 鏡像取得。"""
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id="Ultralytics/YOLOv8", filename="yolov8n.pt")
    except Exception:
        return "yolov8n.pt"   # 交給 ultralytics 自行下載

def build_dataset(n_train, n_val, out):
    tr=foz.load_zoo_dataset("open-images-v7",split="train",label_types=["detections"],
          classes=["Deer"],max_samples=n_train,dataset_name="deer_tr",overwrite=True)
    va=foz.load_zoo_dataset("open-images-v7",split="validation",label_types=["detections"],
          classes=["Deer"],max_samples=n_val,dataset_name="deer_va",overwrite=True)
    for ds,split in [(tr,"train"),(va,"val")]:
        ds.filter_labels("ground_truth",F("label")=="Deer").export(
            export_dir=out,dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",split=split,classes=["Deer"])
    return os.path.join(out,"dataset.yaml")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=int,default=120); ap.add_argument("--val",type=int,default=40)
    ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--device",default="cpu")
    a=ap.parse_args()
    from ultralytics import YOLO
    data_yaml=build_dataset(a.train,a.val,os.path.abspath("deer_yolo"))
    model=YOLO(get_weights())
    model.train(data=data_yaml,epochs=a.epochs,imgsz=a.imgsz,batch=16,device=a.device,
                name="deer_openimages",seed=42,plots=True)
    m=model.val()
    print(f"mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  Recall={m.box.mr:.3f}")
    # 視覺化(低 conf 以顯示偵測；短訓練時信心偏低)
    val_imgs=sorted(glob.glob("deer_yolo/images/val/*.jpg"))[:6]
    model.predict(val_imgs,conf=0.05,save=True,name="deer_pred",max_det=3)
    print("預測結果存於 runs/detect/deer_pred/")

if __name__=="__main__":
    main()
