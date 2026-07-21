"""
Classical image deblurring / sharpening — 可直接執行的基準方法
(原作業把影像做成「模糊 vs 清晰」二元分類，其實框架有誤；真正該做的是「還原」。)

本檔提供不需訓練即可跑的傳統去模糊基準，作為深度學習模型的對照組:
  - Unsharp masking (非銳化遮罩)
  - Richardson–Lucy 反卷積 (假設高斯模糊核)

用法:
    python classical_deblur.py <input_image>   # 不給則自動合成模糊測試圖
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import color, restoration, filters, img_as_float, data
from scipy.signal import convolve2d

def gaussian_kernel(size=7, sigma=2.0):
    ax = np.arange(-size//2+1, size//2+1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2+yy**2)/(2*sigma**2)); return k/k.sum()

def unsharp(img, amount=1.5, sigma=1.5):
    blur = filters.gaussian(img, sigma=sigma, channel_axis=-1 if img.ndim==3 else None)
    return np.clip(img + amount*(img-blur), 0, 1)

def rl_deconv(gray, psf, iters=30):
    return restoration.richardson_lucy(gray, psf, num_iter=iters, clip=True)

def main(path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(here, "..", "figures"); os.makedirs(figdir, exist_ok=True)
    if path and os.path.exists(path):
        from skimage.io import imread
        img = img_as_float(imread(path));
        if img.ndim==3 and img.shape[2]==4: img=img[...,:3]
        blurred = img
        title_in = "Input"
    else:
        print("no image given -> synthesizing a blurred test image")
        img = img_as_float(data.astronaut())
        psf = gaussian_kernel(9, 3.0)
        blurred = np.stack([convolve2d(img[...,c], psf, "same", "symm") for c in range(3)], -1)
        title_in = "Blurred (synthetic)"
    gray = color.rgb2gray(blurred)
    psf = gaussian_kernel(9, 3.0)
    restored = rl_deconv(gray, psf, 30)
    sharp = unsharp(blurred, 1.5, 1.5)

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(blurred); ax[0].set_title(title_in)
    ax[1].imshow(sharp);   ax[1].set_title("Unsharp masking")
    ax[2].imshow(restored, cmap="gray"); ax[2].set_title("Richardson–Lucy (gray)")
    for a in ax: a.axis("off")
    out=os.path.join(figdir,"deblur_demo.png"); fig.tight_layout(); fig.savefig(out,dpi=120)
    print("saved ->", out)

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else None)
