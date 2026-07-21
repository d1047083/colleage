"""
U-Net 影像去模糊模型 (Keras) — 深度學習升級版骨架
在成對的 (blur, sharp) 影像上訓練，學習「模糊 -> 清晰」的還原映射。
資料集: GOPRO_Large / 你的 final/blur + final/sharp

註: 需要 GPU 與成對且對齊的資料。本檔為可訓練的完整骨架。
"""
import os, glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

IMG = 256

def build_unet():
    def enc(x, f):
        x = layers.Conv2D(f,3,padding="same",activation="relu")(x)
        x = layers.Conv2D(f,3,padding="same",activation="relu")(x)
        return x, layers.MaxPool2D()(x)
    def dec(x, skip, f):
        x = layers.Conv2DTranspose(f,2,strides=2,padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(f,3,padding="same",activation="relu")(x)
        return layers.Conv2D(f,3,padding="same",activation="relu")(x)
    inp = layers.Input((IMG,IMG,3))
    c1,p1 = enc(inp,32); c2,p2 = enc(p1,64); c3,p3 = enc(p2,128)
    b = layers.Conv2D(256,3,padding="same",activation="relu")(p3)
    u3 = dec(b,c3,128); u2 = dec(u3,c2,64); u1 = dec(u2,c1,32)
    out = layers.Conv2D(3,1,activation="sigmoid")(u1)          # 直接輸出清晰影像
    return Model(inp, out)

def load_pairs(blur_dir, sharp_dir):
    def load(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img,(IMG,IMG))/255.0; return img
    bs = sorted(glob.glob(os.path.join(blur_dir,"*")))
    ss = sorted(glob.glob(os.path.join(sharp_dir,"*")))
    ds = tf.data.Dataset.from_tensor_slices((bs, ss))
    ds = ds.map(lambda b,s:(load(b),load(s)), tf.data.AUTOTUNE)
    return ds.shuffle(200).batch(8).prefetch(tf.data.AUTOTUNE)

def psnr(y,yp): return tf.image.psnr(y,yp,max_val=1.0)

if __name__=="__main__":
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","final")
    blur, sharp = os.path.join(base,"blur"), os.path.join(base,"sharp")
    if not (os.path.isdir(blur) and os.path.isdir(sharp)):
        print("找不到 final/blur 與 final/sharp，請把去模糊資料集放到專案的 final/ 下。"); raise SystemExit
    ds = load_pairs(blur, sharp)
    model = build_unet()
    model.compile(optimizers.Adam(1e-4), loss="mae", metrics=[psnr])
    model.summary()
    model.fit(ds, epochs=30)
    model.save("deblur_unet.keras")
    print("訓練完成，模型存為 deblur_unet.keras")
