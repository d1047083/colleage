"""
香蕉「好 / 壞」影像分類 — 遷移學習升級版 (MobileNetV2)
原作業有 good banana / bad banana 影像，本檔用預訓練模型做正規的二元分類，
含資料增強、驗證集、早停、混淆矩陣與 Grad-CAM 可解釋性(選配)。

資料結構 (ImageFolder 慣例):
    data/
      good/ *.jpg
      bad/  *.jpg
用法: python train.py --data ../data --epochs 15
"""
import os, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG = 224

def make_ds(data_dir, subset, val_split=0.2, seed=42, bs=32):
    return tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset=subset, seed=seed,
        image_size=(IMG,IMG), batch_size=bs, label_mode="binary")

def build():
    aug = tf.keras.Sequential([layers.RandomFlip("horizontal"),
                               layers.RandomRotation(0.1),
                               layers.RandomZoom(0.1)])
    base = MobileNetV2(input_shape=(IMG,IMG,3), include_top=False, weights="imagenet")
    base.trainable = False
    inp = layers.Input((IMG,IMG,3))
    x = aug(inp); x = preprocess_input(x); x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data"); ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    if not os.path.isdir(a.data):
        print(f"找不到資料夾 {a.data}；請把 good/ 與 bad/ 影像放進去。"); return
    train = make_ds(a.data,"training"); val = make_ds(a.data,"validation")
    model = build()
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)]
    model.fit(train, validation_data=val, epochs=a.epochs, callbacks=cb)
    # 混淆矩陣
    y_true, y_pred = [], []
    for xb, yb in val:
        p = (model.predict(xb, verbose=0) > 0.5).astype(int).ravel()
        y_pred += p.tolist(); y_true += yb.numpy().astype(int).ravel().tolist()
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["bad","good"]))
    model.save("banana_classifier.keras")

if __name__ == "__main__":
    main()
