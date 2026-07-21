# 03 · 水果好壞影像分類 (Banana Freshness)

good/bad banana 影像的正規二元分類，採 **MobileNetV2 遷移學習**。

```bash
python src/train.py --data ../data --epochs 15   # data/ 下需有 good/ 與 bad/
```
含資料增強、早停、混淆矩陣與分類報告。延伸：Grad-CAM 可解釋性、Flask/Gradio 網頁 Demo。
