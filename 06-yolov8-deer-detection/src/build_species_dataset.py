"""
建立「多鹿種」偵測資料集 (可重現、可擴充)
=================================================
流程：從 iNaturalist 下載各鹿種真實照片 → 用通用鹿偵測器自動框出鹿身 →
      標上品種類別 → 輸出 YOLO 偵測資料集。

這就是本專案 species_data/ 的產生方式。你可以改 SPECIES、加大 PER_SPECIES
來擴充資料(每種抓 200+ 張、人工校正框後，模型會好很多)。

用法:
  pip install ultralytics fiftyone huggingface_hub    # fiftyone 非必須(此腳本用 iNat API)
  python build_species_dataset.py --per 60 --out species_data
需要 weights/generic_deer_detector.pt (本專案已附)；沒有時可用 yolov8n.pt 但框較不準。
"""
import argparse, os, json, ssl, glob, shutil, random, urllib.request, urllib.parse

# 鹿種 -> 學名 (iNaturalist 以學名精確比對)
SPECIES = {
    "white_tailed_deer": "Odocoileus virginianus",
    "red_deer":          "Cervus elaphus",
    "sika_deer":         "Cervus nippon",
    "fallow_deer":       "Dama dama",
    "reindeer":          "Rangifer tarandus",
}
UA = {"User-Agent": "deer-species-demo/1.0"}
_ctx = ssl.create_default_context(); _ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE

def _get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25, context=_ctx))

def taxon_id(sci):
    r = _get("https://api.inaturalist.org/v1/taxa?rank=species&q=" + urllib.parse.quote(sci))
    for t in r["results"]:
        if t["name"].lower() == sci.lower():
            return t["id"]
    return r["results"][0]["id"]

def download_species(name, sci, per, raw):
    tid = taxon_id(sci); os.makedirs(f"{raw}/{name}", exist_ok=True)
    got, page = 0, 1
    while got < per and page <= 8:
        obs = _get(f"https://api.inaturalist.org/v1/observations?taxon_id={tid}"
                   f"&photos=true&per_page=30&page={page}&quality_grade=research&order_by=votes")
        for o in obs["results"]:
            if got >= per: break
            try:
                ph = o["photos"][0]["url"].replace("square", "medium")
                urllib.request.urlretrieve(ph, f"{raw}/{name}/{got:03d}.jpg"); got += 1
            except Exception: pass
        page += 1
    return got

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=45, help="每種下載張數")
    ap.add_argument("--out", default="species_data")
    ap.add_argument("--detector", default="weights/generic_deer_detector.pt")
    ap.add_argument("--val_frac", type=float, default=0.2)
    a = ap.parse_args(); random.seed(42)
    from ultralytics import YOLO
    raw = f"{a.out}/raw"; yolo = f"{a.out}/deer_species_yolo"
    for s in ["images/train","images/val","labels/train","labels/val"]:
        os.makedirs(f"{yolo}/{s}", exist_ok=True)
    det = YOLO(a.detector if os.path.exists(a.detector) else "yolov8n.pt")
    classes = list(SPECIES)
    for ci, name in enumerate(classes):
        n = download_species(name, SPECIES[name], a.per, raw)
        imgs = sorted(glob.glob(f"{raw}/{name}/*.jpg")); random.shuffle(imgs)
        nval = max(1, int(len(imgs) * a.val_frac)); boxed = 0
        for i, ip in enumerate(imgs):
            split = "val" if i < nval else "train"
            r = det.predict(ip, conf=0.02, imgsz=320, max_det=1, agnostic_nms=True, verbose=False)[0]
            if len(r.boxes) >= 1:
                cx, cy, w, h = r.boxes.xywhn[0].tolist(); boxed += 1
            else:
                cx, cy, w, h = 0.5, 0.5, 0.72, 0.72     # 無偵測 -> 中央框
            stem = f"{name}_{i:03d}"
            shutil.copy(ip, f"{yolo}/images/{split}/{stem}.jpg")
            open(f"{yolo}/labels/{split}/{stem}.txt", "w").write(f"{ci} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}\n")
        print(f"{name:20s} downloaded={n} auto-boxed={boxed}")
    y = f"path: {a.out}/deer_species_yolo\ntrain: images/train\nval: images/val\nnames:\n"
    y += "\n".join(f"  {i}: {c}" for i, c in enumerate(classes)) + "\n"
    open(f"{yolo}/data.yaml", "w").write(y)
    print("YOLO dataset ->", yolo)

if __name__ == "__main__":
    main()
