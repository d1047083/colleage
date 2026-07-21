"""
匯率分析主流程：從央行匯出的 json 解析出每日匯率，做 EDA 和隔天匯率的預測。
輸出 data/fx_daily.csv、figures/、results/metrics.md。
圖表都用英文標籤避免字型缺字；結果照實寫，日頻匯率其實接近隨機漫步。
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIG  = os.path.join(ROOT, "figures")
RES  = os.path.join(ROOT, "results")
for d in (DATA, FIG, RES): os.makedirs(d, exist_ok=True)

# 允許用環境變數指定原始 json 路徑
RAW = os.environ.get("TWCB_JSON",
      "/mnt/user-data/uploads/Desktop/資料夾/數學/download_TWCB.json")

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True,
                     "grid.alpha": .3, "font.size": 11})
ACCENT = "#2563eb"; ACCENT2 = "#dc2626"; MUTE = "#64748b"

# ---------- 1. 萃取每日匯率 ----------
def load_fx_daily():
    raw = json.load(open(RAW, encoding="utf-8"))
    key = [k for k in raw if "匯率" in k and k.endswith("_日")][0]
    nested = json.loads(raw[key])
    df = pd.DataFrame(nested)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    # 欄位改英文代碼(抓非 USD 的三字幣別代碼)，數值轉 float
    import re
    def code(c):
        toks = [t for t in re.findall(r"[A-Z]{3}", c) if t != "USD"]
        return toks[0] if toks else c
    df = df.rename(columns={c: code(c) for c in df.columns})
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    df.to_csv(os.path.join(DATA, "fx_daily.csv"))
    return df

# ---------- 2. EDA 圖 ----------
def eda(df):
    ntd = df["NTD"].dropna()
    # 2.1 長期趨勢
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ntd.index, ntd.values, color=ACCENT, lw=1)
    ax.set_title("USD/TWD Exchange Rate, 1993–present (TWD per 1 USD)")
    ax.set_ylabel("TWD per USD"); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(f"{FIG}/01_twd_trend.png"); plt.close(fig)

    # 2.2 多幣別正規化比較
    cur = [c for c in ["NTD","JPY","KRW","CNY","SGD"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(10,4))
    for c in cur:
        s = df[c].dropna()
        s = s / s.iloc[0] * 100
        ax.plot(s.index, s.values, lw=1, label=c)
    ax.set_title("Currencies vs USD, indexed to 100 at series start")
    ax.set_ylabel("Index (start = 100)"); ax.legend(ncol=len(cur), fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/02_multi_currency.png"); plt.close(fig)

    # 2.3 報酬率與滾動波動度
    ret = np.log(ntd).diff().dropna()
    vol = ret.rolling(30).std() * np.sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(vol.index, vol.values, color=ACCENT2, lw=1)
    ax.set_title("USD/TWD 30-day Rolling Annualized Volatility (%)")
    ax.set_ylabel("Annualized vol (%)")
    fig.tight_layout(); fig.savefig(f"{FIG}/03_volatility.png"); plt.close(fig)
    return ntd, ret

# ---------- 3. 特徵工程 ----------
def make_features(ntd):
    df = pd.DataFrame({"y": ntd})
    df["ret"] = np.log(df["y"]).diff()
    for l in (1,2,3,5,10):
        df[f"ret_lag{l}"] = df["ret"].shift(l)
    df["ma5"]  = df["ret"].rolling(5).mean().shift(1)
    df["ma10"] = df["ret"].rolling(10).mean().shift(1)
    df["vol10"]= df["ret"].rolling(10).std().shift(1)
    df["dow"]  = df.index.dayofweek
    df["target_ret"] = df["ret"].shift(-1)          # 預測明日報酬
    return df.dropna()

# ---------- 4. 走動式(walk-forward)驗證 ----------
def evaluate(df):
    feats = [c for c in df.columns if c.startswith(("ret_lag","ma","vol")) or c=="dow"]
    X = df[feats].values; y = df["target_ret"].values
    y_level = df["y"].values                         # 今日價位(用於還原)
    n = len(df); split = int(n*0.7)                  # 前70%訓練，後30%測試(時間序)
    idx_test = df.index[split:]

    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            random_state=42),
    }
    results = {}
    # 隨機漫步基準：預測明日=今日(報酬=0)
    rw_pred_level = y_level[split:]                   # ŷ_{t+1}=y_t
    actual_level  = df["y"].shift(-1).values[split:]
    mask = ~np.isnan(actual_level)
    def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
    def mae(a,b):  return float(np.mean(np.abs(a-b)))
    results["RandomWalk"] = {
        "rmse": rmse(rw_pred_level[mask], actual_level[mask]),
        "mae":  mae(rw_pred_level[mask], actual_level[mask]),
        "dir_acc": float(np.mean((0>0)==(df["target_ret"].values[split:][mask]>0)))
    }
    for name, m in models.items():
        m.fit(X[:split], y[:split])
        pred_ret = m.predict(X[split:])
        pred_level = y_level[split:] * np.exp(pred_ret)   # 由報酬還原價位
        dir_acc = float(np.mean((pred_ret[mask]>0)==(df["target_ret"].values[split:][mask]>0)))
        results[name] = {
            "rmse": rmse(pred_level[mask], actual_level[mask]),
            "mae":  mae(pred_level[mask], actual_level[mask]),
            "dir_acc": dir_acc,
        }
    # 圖：測試期預測 vs 實際(GBR)
    gbr = models["GradientBoosting"]
    pred_ret = gbr.predict(X[split:])
    pred_level = y_level[split:]*np.exp(pred_ret)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(idx_test, actual_level, color=MUTE, lw=1.2, label="Actual")
    ax.plot(idx_test, pred_level, color=ACCENT, lw=.9, label="GBR 1-day-ahead")
    ax.set_title("Test-set 1-day-ahead Forecast vs Actual (USD/TWD)")
    ax.set_ylabel("TWD per USD"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/04_forecast_vs_actual.png"); plt.close(fig)

    # 圖：RMSE 與方向準確率比較
    names = list(results.keys())
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].bar(names, [results[n]["rmse"] for n in names], color=ACCENT)
    axes[0].set_title("Test RMSE (lower = better)"); axes[0].tick_params(axis='x', rotation=20)
    axes[1].bar(names, [results[n]["dir_acc"]*100 for n in names], color=ACCENT2)
    axes[1].axhline(50, color="k", ls="--", lw=1)
    axes[1].set_title("Directional Accuracy (%)  — 50% = coin flip")
    axes[1].tick_params(axis='x', rotation=20); axes[1].set_ylim(40,60)
    fig.tight_layout(); fig.savefig(f"{FIG}/05_model_comparison.png"); plt.close(fig)
    return results, idx_test[0]

def write_report(df_fx, results, test_start):
    lines = []
    lines.append("# 匯率預測結果\n")
    lines.append(f"資料是央行每日匯率，{df_fx.index.min().date()} 到 {df_fx.index.max().date()}，"
                 f"{len(df_fx):,} 筆，{df_fx.shape[1]} 種貨幣。任務是預測隔天的美元對台幣，"
                 f"用時間序的 70/30 切分（測試從 {pd.Timestamp(test_start).date()} 開始）。\n")
    lines.append("| 模型 | 測試 RMSE | 測試 MAE | 方向準確率 |\n|---|---|---|---|")
    for n,r in results.items():
        lines.append(f"| {n} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['dir_acc']*100:.1f}% |")
    best_rmse = min(results, key=lambda k: results[k]["rmse"])
    lines.append("\n## 怎麼看這個結果\n")
    lines.append(
        f"隨機漫步這個基準很難贏。匯率接近 martingale，「隔天約等於今天」就已經給出很低的 RMSE。"
        f"這裡 RMSE 最低的是 {best_rmse}，各模型猜方向的比例都在 50% 附近，猜隔天漲跌跟丟硬幣差不多。"
        f"這在匯率研究裡是很常見的結論，照實寫出來比硬做一個看起來會賺的預測器好。\n")
    lines.append("\n## 下一步能往哪找優勢\n"
        "拉長預測區間，例如週報酬。用央行另外 170 多條總經序列當特徵。"
        "改做波動度預測，方向難猜但波動可以預測。做一個把交易成本算進去的回測。\n")
    open(os.path.join(RES,"metrics.md"),"w",encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    print("Loading TWCB data ...")
    fx = load_fx_daily()
    print("FX shape:", fx.shape, "| currencies:", list(fx.columns))
    ntd, ret = eda(fx)
    feat = make_features(ntd)
    res, test_start = evaluate(feat)
    write_report(fx, res, test_start)
    print("\nDone. Figures in figures/, report in results/metrics.md")
