"""
去模糊量化評估 (Deblurring benchmark)  —  電腦視覺深化
==========================================================
不需訓練即可產出真實數字：對清晰影像施加「已知的高斯模糊+雜訊」造成退化，
再用多種傳統方法還原，並以 PSNR / SSIM 客觀衡量還原品質。
這建立了深度學習模型(U-Net/DeblurGAN)要超越的『傳統基準線』。
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage import color, restoration, filters, img_as_float, data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"figures"); RES=os.path.join(ROOT,"results")
for d in (FIG,RES): os.makedirs(d,exist_ok=True)

def gk(size=9,sigma=3.0):
    ax=np.arange(-size//2+1,size//2+1); xx,yy=np.meshgrid(ax,ax)
    k=np.exp(-(xx**2+yy**2)/(2*sigma**2)); return k/k.sum()

def degrade(img,psf,noise=0.01,seed=0):
    rng=np.random.default_rng(seed)
    b=convolve2d(img,psf,"same","symm")
    return np.clip(b+rng.normal(0,noise,img.shape),0,1)

def restore(deg,psf):
    out={}
    out["Wiener"]=np.clip(restoration.wiener(deg,psf,balance=0.02),0,1)
    out["Richardson-Lucy"]=np.clip(restoration.richardson_lucy(deg,psf,num_iter=30,clip=True),0,1)
    out["Unsharp"]=np.clip(filters.unsharp_mask(deg,radius=2,amount=1.5),0,1)
    return out

def main():
    img=color.rgb2gray(img_as_float(data.astronaut()))
    psf=gk(9,3.0)
    deg=degrade(img,psf,noise=0.01)
    res=restore(deg,psf)
    rows=[("Degraded (blur+noise)",deg)]+list(res.items())
    metrics={name:(psnr(img,im,data_range=1.0),ssim(img,im,data_range=1.0)) for name,im in rows}

    fig,ax=plt.subplots(1,len(rows)+1,figsize=(3*(len(rows)+1),3.4))
    ax[0].imshow(img,cmap="gray"); ax[0].set_title("Original (target)"); ax[0].axis("off")
    for a,(name,im) in zip(ax[1:],rows):
        p,s=metrics[name]
        a.imshow(im,cmap="gray"); a.axis("off")
        a.set_title(f"{name}\nPSNR {p:.1f} · SSIM {s:.2f}",fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/deep_deblur_benchmark.png",dpi=120); plt.close(fig)

    L=["# Deblurring — quantitative benchmark\n",
       "施加已知高斯模糊(σ=3)+高斯雜訊(σ=0.01)後還原，對照原圖計算 PSNR/SSIM：\n",
       "| Method | PSNR (dB) | SSIM |","|---|---|---|"]
    for name,(p,s) in metrics.items():
        L.append(f"| {name} | {p:.2f} | {s:.3f} |")
    best=max((k for k in metrics if k!="Degraded (blur+noise)"),key=lambda k:metrics[k][0])
    L+=["",f"- 最佳傳統法：**{best}** (PSNR {metrics[best][0]:.2f} dB)。",
        "- 這就是深度學習模型必須超越的基準線；U-Net 訓練程式見 src/unet_deblur.py。",
        "- 評估用 PSNR(數值保真)與 SSIM(結構相似)雙指標，比只看肉眼更客觀。\n"]
    open(f"{RES}/deblur_metrics.md","w",encoding="utf-8").write("\n".join(L))
    print("\n".join(L))

if __name__=="__main__": main()
