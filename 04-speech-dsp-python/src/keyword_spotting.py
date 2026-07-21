"""
關鍵詞辨識，用 MFCC 加 DTW 做模板比對。
每個詞取能量最高的一段當模板，靠加噪和變速產生測試查詢，算 top-1 準確率。
每個詞只要一個範例，適合資料很少的情況。
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
    L=["# 關鍵詞辨識（MFCC + DTW）\n",
       f"詞庫用的是真實錄音：{', '.join(WORDS)}。",
       f"每個詞取能量最高的 {WIN} 秒當模板，測試查詢是 3 種 SNR × 3 種變速，每詞 {len(conds)} 種，一共 {C.sum()} 次。",
       f"Top-1 準確率 {acc*100:.1f}%。",
       "",
       "MFCC 抓的是音色的包絡，DTW 能容忍語速伸縮，所以加了雜訊和變速還是認得出來。"
       "這是每個詞只有一個模板的做法，適合資料很少的情況。資料多了之後可以換成 CNN/RNN 的端到端關鍵詞偵測。\n"]
    open(f"{RES}/kws_metrics.md","w",encoding="utf-8").write("\n".join(L))
    print("\n".join(L)); print("saved -> figures/deep_speech_04_kws.png")

if __name__=="__main__": main()
