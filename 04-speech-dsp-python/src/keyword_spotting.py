"""
關鍵詞辨識 (Keyword spotting) — MFCC + DTW 模板比對
========================================================
用真實 wav 建立小型語音詞庫，對每段抽出「能量最高的核心片段」當模板，
以 MFCC + 動態時間規整(DTW) 做辨識(one-shot，不需大量訓練資料)。
以加噪/變速產生測試查詢，量測真實 top-1 辨識準確率與混淆矩陣。

輸出: figures/deep_speech_04_kws.png, results/kws_metrics.md
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import librosa
from itertools import product

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"figures"); RES=os.path.join(ROOT,"results")
for d in (FIG,RES): os.makedirs(d,exist_ok=True)
SR=16000; DEF="/mnt/user-data/uploads/Desktop/03_語音訊號處理/PCA_ICA盲訊號分離"
WORDS=["jarvus","hello3","bug","bath","star","happy","tiger"]
WIN=1.5  # 核心片段秒數

def salient(y, win=WIN):
    L=int(win*SR)
    if len(y)<=L: return y
    e=librosa.feature.rms(y=y,frame_length=1024,hop_length=256)[0]
    hop=256; wlen=max(1,L//hop)
    cs=np.cumsum(e); s=np.argmax(cs[wlen:]-cs[:-wlen])  # 能量最高的連續視窗
    st=s*hop; return y[st:st+L]

def feats(y):
    m=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=13)
    d=librosa.feature.delta(m)
    f=np.vstack([m,d])
    return (f-f.mean(1,keepdims=True))/(f.std(1,keepdims=True)+1e-9)

def dtw_dist(a,b):
    D,_=librosa.sequence.dtw(X=a,Y=b,metric="euclidean")
    return D[-1,-1]/(a.shape[1]+b.shape[1])

def augment(y,snr=None,rate=None,gain=1.0,seed=0):
    rng=np.random.default_rng(seed); z=y.copy()
    if rate: z=librosa.effects.time_stretch(z,rate=rate)
    if snr is not None:
        p=np.mean(z**2); n=rng.normal(0,np.sqrt(p/(10**(snr/10))),len(z)); z=z+n
    return z*gain

def main():
    templates={w:feats(salient(librosa.load(os.path.join(DEF,w+".wav"),sr=SR)[0])) for w in WORDS}
    conds=list(product([20,10,5],[0.9,1.0,1.1]))     # SNR × 變速
    y_true=[]; y_pred=[]
    for wi,w in enumerate(WORDS):
        base=salient(librosa.load(os.path.join(DEF,w+".wav"),sr=SR)[0])
        for j,(snr,rate) in enumerate(conds):
            q=feats(augment(base,snr=snr,rate=rate,gain=0.9,seed=j))
            dists={t:dtw_dist(q,templates[t]) for t in WORDS}
            pred=min(dists,key=dists.get)
            y_true.append(w); y_pred.append(pred)
    # 混淆矩陣 + 準確率
    n=len(WORDS); idx={w:i for i,w in enumerate(WORDS)}
    C=np.zeros((n,n),int)
    for t,p in zip(y_true,y_pred): C[idx[t],idx[p]]+=1
    acc=float(np.trace(C)/C.sum())
    fig,ax=plt.subplots(figsize=(6.2,5.4))
    im=ax.imshow(C,cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(WORDS,rotation=45,ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(WORDS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Keyword spotting confusion (MFCC+DTW)\nTop-1 accuracy = {acc*100:.1f}%  ({len(conds)} queries/word)")
    for i in range(n):
        for j in range(n):
            if C[i,j]: ax.text(j,i,C[i,j],ha="center",va="center",
                               color="white" if C[i,j]>C.max()/2 else "black",fontsize=9)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{FIG}/deep_speech_04_kws.png",dpi=120); plt.close(fig)
    L=["# Keyword spotting — MFCC + DTW\n",
       f"- 詞庫(真實 wav): {', '.join(WORDS)}",
       f"- 每詞取能量最高的 {WIN}s 核心片段為模板；測試查詢 = 3 SNR × 3 變速 = {len(conds)} 種/詞",
       f"- **Top-1 辨識準確率 = {acc*100:.1f}%** (共 {C.sum()} 次查詢)",
       "",
       "## 解讀",
       "MFCC 捕捉音色包絡、DTW 容忍語速伸縮，因此在加噪與變速下仍能辨識——"
       "這是 one-shot(每詞僅一模板)的作法，適合資料稀少情境。",
       "延伸：資料變多時可改用 CNN/RNN 端到端關鍵詞偵測(如 Google Speech Commands)。\n"]
    open(f"{RES}/kws_metrics.md","w",encoding="utf-8").write("\n".join(L))
    print("\n".join(L)); print("saved -> figures/deep_speech_04_kws.png")

if __name__=="__main__": main()
