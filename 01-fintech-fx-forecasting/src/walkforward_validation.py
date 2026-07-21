"""
月頻模型的 walk-forward 驗證。
單一切分容易靠運氣，這裡改成逐月往前、每次只用過去的資料重訓，是時間序該有的做法。
輸出 figures/deep_04_walkforward.png、results/walkforward_metrics.md。
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import importlib.util

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"figures"); RES=os.path.join(ROOT,"results")
RAW=os.environ.get("TWCB_JSON","/mnt/user-data/uploads/Desktop/資料夾/數學/download_TWCB.json")
# 借用 deep_analysis 的載入工具
spec=importlib.util.spec_from_file_location("da",os.path.join(HERE,"deep_analysis.py"))
da=importlib.util.module_from_spec(spec); spec.loader.exec_module(da)

def build_frame(raw):
    K="中央銀行統計資料庫_金融統計月報"
    fx=da.series(raw,"中央銀行統計資料庫_我國與主要貿易對手通貨對美元之匯率_月","新台幣NTD/USD")
    f={"discount_rate":da.series(raw,f"{K}_利率統計_中央銀行利率_月","重貼現率"),
       "interbank_on":da.series(raw,f"{K}_利率統計_金融業拆款利率_月","隔夜-加權平均_原始值"),
       "M1B_yoy":da.series(raw,f"{K}_重要金融指標_季節調整後重要金融指標_月","貨幣總計數-M1B-日平均_年增率"),
       "M2_yoy":da.series(raw,f"{K}_重要金融指標_季節調整後重要金融指標_月","貨幣總計數-M2-日平均_年增率"),
       "fx_reserves":da.series(raw,f"{K}_重要金融指標_其他_月","外匯存底(百萬美元)"),
       "cpi_yoy":da.series(raw,f"{K}_重要金融指標_其他_月","消費者物價指數年增率"),
       "stock_idx":da.series(raw,f"{K}_重要金融指標_其他_月","股票市場股價指數(民國５５年=１００)(１９６６=１００）")}
    df=pd.DataFrame({"fx":fx})
    for k,v in f.items(): df[k]=v
    df=df.resample("MS").last()
    df["ret"]=np.log(df["fx"]).diff()
    df["reserves_chg"]=df["fx_reserves"].pct_change()
    df["stock_ret"]=np.log(df["stock_idx"]).diff()
    df["rate_mom"]=df["discount_rate"].diff()
    df["target"]=df["ret"].shift(-1)
    macro=["discount_rate","interbank_on","M1B_yoy","M2_yoy","cpi_yoy","reserves_chg","stock_ret","rate_mom"]
    for c in macro: df[c]=df[c].shift(1)          # 公布落後修正
    X=macro+["ret"]
    return df[X+["target","fx"]].dropna(), X

def walk_forward(d, X, init=60):
    idx=d.index; preds=[]; acts=[]; times=[]
    for t in range(init, len(d)-1):
        tr=d.iloc[:t]
        rf=RandomForestRegressor(n_estimators=120,max_depth=4,random_state=42,min_samples_leaf=3,n_jobs=-1)
        rf.fit(tr[X].values, tr["target"].values)
        p=rf.predict(d[X].values[t:t+1])[0]
        preds.append(p); acts.append(d["target"].values[t]); times.append(idx[t])
    preds=np.array(preds); acts=np.array(acts); times=pd.DatetimeIndex(times)
    dir_acc=float(np.mean(np.sign(preds)==np.sign(acts)))
    k=int(np.sum(np.sign(preds)==np.sign(acts))); n=len(acts)
    pval=stats.binomtest(k,n,0.5,alternative="greater").pvalue
    rw_rmse=float(np.sqrt(np.mean(acts**2))); rf_rmse=float(np.sqrt(np.mean((acts-preds)**2)))
    # walk-forward 回測
    cost=0.001; pos=np.sign(preds)
    turn=np.abs(np.diff(np.concatenate([[0],pos])))
    strat=pos*acts - turn*cost
    sr=lambda r: float(np.mean(r)/(np.std(r)+1e-9)*np.sqrt(12))
    eq_s=np.cumprod(1+strat); eq_b=np.cumprod(1+acts)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(times,np.cumsum(np.sign(preds)==np.sign(acts))/np.arange(1,n+1)*100,color="#2563eb")
    ax[0].axhline(50,color="k",ls="--",lw=1); ax[0].set_title("Running directional accuracy (%)"); ax[0].set_ylim(30,80)
    ax[1].plot(times,eq_s,color="#2563eb",label=f"WF strategy (Sharpe {sr(strat):.2f})")
    ax[1].plot(times,eq_b,color="#64748b",label=f"Buy&hold (Sharpe {sr(acts):.2f})")
    ax[1].set_title("Walk-forward out-of-sample equity"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/deep_04_walkforward.png",dpi=120); plt.close(fig)
    return {"n":n,"dir_acc":dir_acc,"binom_p":float(pval),"rw_rmse":rw_rmse,"rf_rmse":rf_rmse,
            "strat_sharpe":sr(strat),"bh_sharpe":sr(acts),"strat_total":float(eq_s[-1]-1),
            "bh_total":float(eq_b[-1]-1)}

if __name__=="__main__":
    raw=json.load(open(RAW,encoding="utf-8"))
    d,X=build_frame(raw)
    r=walk_forward(d,X,init=60)
    verdict=("還是明顯比丟硬幣好，訊號的可信度反而更高。" if r["binom_p"]<0.05 and r["dir_acc"]>0.55
             else "優勢明顯縮水，表示原本的 67% 有一部分是單一切分的運氣，這也是為什麼要做 walk-forward。")
    L=["# 月頻模型的 walk-forward 驗證\n",
       f"逐月往前、每次只用過去的資料重訓，測試 {r['n']} 個月，初始訓練窗 60 個月。",
       f"方向準確率 {r['dir_acc']*100:.1f}%，跟 50% 的 binomial 檢定 p 值 {r['binom_p']:.3f}。",
       f"RandomForest RMSE {r['rf_rmse']:.4f}，隨機漫步 {r['rw_rmse']:.4f}。",
       f"回測策略 Sharpe {r['strat_sharpe']:.2f}（總報酬 {r['strat_total']*100:.1f}%），"
       f"買進持有 Sharpe {r['bh_sharpe']:.2f}（{r['bh_total']*100:.1f}%）。",
       "",
       "## 結論",
       f"先前單一 70/30 切分是 67.6%，換成 walk-forward 之後，{verdict}"
       "這是時間序模型該有的驗證方式，與其宣稱準確率多高，不如用最嚴格的方法檢驗一遍。\n"]
    open(f"{RES}/walkforward_metrics.md","w",encoding="utf-8").write("\n".join(L))
    print("\n".join(L)); print("saved -> figures/deep_04_walkforward.png")
