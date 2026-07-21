"""
用真實錄音做的幾件事：
A. MFCC 和梅爾頻譜。
B. 基頻追蹤（pyin）。
C. ICA 盲訊號分離，把兩段混在一起的人聲拆開。
輸出圖和分離後的音檔。
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy.io import wavfile
from sklearn.decomposition import FastICA

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"figures"); RES=os.path.join(ROOT,"results")
for d in (FIG,RES): os.makedirs(d,exist_ok=True)
SR=16000
DEF="/mnt/user-data/uploads/Desktop/03_語音訊號處理/PCA_ICA盲訊號分離"

def load(path):
    y,_=librosa.load(path,sr=SR,mono=True); return y

def mfcc_block(y,name):
    mf=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=13)
    mel=librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=SR,n_mels=64),ref=np.max)
    fig,ax=plt.subplots(2,1,figsize=(10,6))
    im0=librosa.display.specshow(mel,sr=SR,x_axis="time",y_axis="mel",ax=ax[0],cmap="magma")
    ax[0].set_title(f"Mel-spectrogram — {name}"); fig.colorbar(im0,ax=ax[0],format="%+0.0f dB")
    im1=librosa.display.specshow(mf,sr=SR,x_axis="time",ax=ax[1],cmap="viridis")
    ax[1].set_title("MFCC (13)"); ax[1].set_ylabel("coeff"); fig.colorbar(im1,ax=ax[1])
    fig.tight_layout(); fig.savefig(f"{FIG}/deep_speech_01_mfcc.png"); plt.close(fig)
    return mf.shape

def f0_block(y,name):
    f0,vflag,vprob=librosa.pyin(y,fmin=70,fmax=400,sr=SR)
    t=librosa.times_like(f0,sr=SR)
    voiced=np.isfinite(f0)
    fig,ax=plt.subplots(figsize=(10,3.4))
    ax.plot(t,f0,".",ms=2,color="#dc2626")
    ax.set_title(f"Fundamental frequency f0 (pyin) — {name}")
    ax.set_ylabel("Hz"); ax.set_xlabel("Time (s)")
    fig.tight_layout(); fig.savefig(f"{FIG}/deep_speech_02_f0.png"); plt.close(fig)
    med=float(np.nanmedian(f0)) if voiced.any() else float("nan")
    return {"voiced_frac":float(np.mean(voiced)),"median_f0":med}

def ica_block(s1,s2):
    n=min(len(s1),len(s2)); S=np.c_[s1[:n],s2[:n]]
    S=(S-S.mean(0))/S.std(0)                       # 標準化
    A=np.array([[0.8,0.35],[0.4,0.9]])              # 混音矩陣(兩支麥克風)
    X=S@A.T                                         # 觀測到的混合訊號
    ica=FastICA(n_components=2,random_state=0,max_iter=1000,whiten="unit-variance")
    Shat=ica.fit_transform(X)
    # 對齊還原訊號與原始源(以相關係數配對)
    def norm(a): return (a-a.mean())/(a.std()+1e-9)
    c=np.abs(np.corrcoef(np.c_[norm(Shat),norm(S)].T)[:2,2:])  # 2x2 相關
    order=[int(np.argmax(c[0])),int(np.argmax(c[1]))]
    corrs=[float(c[i,order[i]]) for i in range(2)]

    labels=["Source A (star)","Source B (happy)"]
    fig,ax=plt.subplots(3,2,figsize=(11,7))
    for j in range(2):
        ax[0,j].plot(S[:,j],lw=.4,color="#334155"); ax[0,j].set_title(f"Original {labels[j]}")
        ax[1,j].plot(X[:,j],lw=.4,color="#2563eb"); ax[1,j].set_title(f"Mixed mic {j+1} (both voices)")
        ax[2,j].plot(Shat[:,j],lw=.4,color="#dc2626"); ax[2,j].set_title(f"ICA recovered #{j+1}")
    for a in ax.ravel(): a.set_xticks([])
    fig.tight_layout(); fig.savefig(f"{FIG}/deep_speech_03_ica.png"); plt.close(fig)
    # 輸出分離後音檔
    for j in range(2):
        w=Shat[:,j]; w=(w/np.max(np.abs(w))*0.9*32767).astype(np.int16)
        wavfile.write(f"{RES}/ica_recovered_{j+1}.wav",SR,w)
    # 混合音檔(給人聽對照)
    xm=(X[:,0]/np.max(np.abs(X[:,0]))*0.9*32767).astype(np.int16)
    wavfile.write(f"{RES}/mixed_mic1.wav",SR,xm)
    return corrs

if __name__=="__main__":
    a=sys.argv[1] if len(sys.argv)>1 else os.path.join(DEF,"star.wav")
    b=sys.argv[2] if len(sys.argv)>2 else os.path.join(DEF,"happy.wav")
    s1,s2=load(a),load(b)
    print("A. MFCC ..."); sh=mfcc_block(s2,os.path.basename(b))
    print("B. f0 ...");  f0=f0_block(s2,os.path.basename(b))
    print("C. ICA ...");  corrs=ica_block(s1,s2)
    lines=["# 語音分析結果\n",
      f"錄音用的是真實的 {os.path.basename(a)} 和 {os.path.basename(b)}，{SR} Hz。\n",
      f"MFCC 和梅爾頻譜：抽出 13 維 MFCC，形狀 {sh}，這是語音辨識和關鍵詞偵測常用的前端特徵。\n",
      f"基頻（pyin）：有聲的幀佔 {f0['voiced_frac']*100:.1f}%，中位基頻大約 {f0['median_f0']:.0f} Hz。\n",
      f"ICA 盲訊號分離：把兩段人聲混進兩支麥克風，再用 FastICA 拆開，還原和原始訊號的相關是 {corrs[0]:.2f} 和 {corrs[1]:.2f}。"
      "拆開後的音檔在 results/ica_recovered_*.wav，可以直接聽出兩個人的聲音被分開了。\n",
      "再往下，MFCC 接一個小分類器就能做關鍵詞辨識，ICA 也可以延伸到多麥克風陣列和去噪。\n"]
    open(f"{RES}/speech_metrics.md","w",encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
