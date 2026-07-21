# 03 · 水果好壞分類

拿 good/bad 的香蕉照片做二元分類，用 MobileNetV2 遷移學習。

```bash
python src/train.py --data ../data --epochs 15   # data/ 底下要有 good/ 和 bad/ 兩個資料夾
```

程式裡有資料增強、早停、混淆矩陣和分類報告。之後可以加 Grad-CAM 看模型憑哪裡判斷，或包成 Flask/Gradio 的網頁 demo。
