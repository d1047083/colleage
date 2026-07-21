import base64, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def b64(p):
    p = os.path.join(ROOT, p)
    if not os.path.exists(p): return ""
    return "data:image/png;base64," + base64.b64encode(open(p,"rb").read()).decode()

FIG = {
 "trend": b64("01-fintech-fx-forecasting/figures/01_twd_trend.png"),
 "garch": b64("01-fintech-fx-forecasting/figures/deep_01_garch.png"),
 "macro": b64("01-fintech-fx-forecasting/figures/deep_02_macro.png"),
 "backtest": b64("01-fintech-fx-forecasting/figures/deep_03_backtest.png"),
 "walkfwd": b64("01-fintech-fx-forecasting/figures/deep_04_walkforward.png"),
 "kws": b64("04-speech-dsp-python/figures/deep_speech_04_kws.png"),
 "speech_ica": b64("04-speech-dsp-python/figures/deep_speech_03_ica.png"),
 "speech_mfcc": b64("04-speech-dsp-python/figures/deep_speech_01_mfcc.png"),
 "road": b64("05-autonomous-sim/figures/deep_road_network.png"),
 "deblur": b64("02-cv-image-deblur/figures/deep_deblur_benchmark.png"),
 "deer": b64("06-yolov8-deer-detection/figures/deer_species_detections.png"),
 "deer_curves": b64("06-yolov8-deer-detection/figures/deer_detections.png"),
}

PROJECTS = [
 ("01","央行匯率分析與預測","FX Analysis & Forecasting",
  "解析台灣央行 178 個資料集(1993–至今)。深化後涵蓋三層：<b>GARCH 波動度建模</b>、"
  "<b>月頻×總經基本面預測</b>(利率/貨幣供給/外匯存底)、以及<b>含交易成本的樣本外回測</b>。",
  ["pandas","scikit-learn","GARCH(arch)","walk-forward","backtesting"],
  ["trend","garch","macro","walkfwd"],
  "真成果：GARCH 波動持續性 α+β=0.885；月頻總經模型方向準確率 67.6% (p<0.001)。<b>並主動做前視偏誤檢查</b>"
  "與最嚴格的<b>擴張窗 walk-forward 驗證(288 個月)</b>——優勢依然穩健(方向 64.9%, p<0.001, Sharpe 0.99)，"
  "不是單一切分的運氣。誠實標註仍需多幣別測試，不誇大。"),
 ("04","語音訊號分析","Speech DSP (Python)",
  "從 MATLAB 移植並深化。用<b>真實語音 wav</b> 做 MFCC 特徵、基頻 f0 追蹤(pyin)，"
  "以及 <b>ICA 盲訊號分離(雞尾酒會問題)</b>：把兩段混在一起的人聲拆開。",
  ["NumPy","SciPy","librosa","FastICA","DTW"],["kws","speech_ica"],
  "真成果：ICA 把兩段真實人聲分離(相關 1.00)並輸出可聽音檔；MFCC+DTW 關鍵詞辨識在 7 詞詞庫、"
  "加噪與變速下 <b>Top-1 準確率 95.2%</b>(63 次查詢)；另有 13 維 MFCC 與基頻追蹤(中位 263 Hz)。"),
 ("02","影像去模糊","Image Deblurring",
  "把原本錯誤的「模糊 vs 清晰分類」重構為真正的<b>影像還原</b>。深化加入 <b>PSNR/SSIM 量化基準</b>"
  "(Wiener / Richardson-Lucy / Unsharp)與 U-Net + 感知損失訓練程式。",
  ["scikit-image","OpenCV","Keras U-Net","PSNR/SSIM"],["deblur"],
  "真成果：在已知退化上量化比較各傳統法(最佳 Unsharp PSNR 24.6 dB)，"
  "誠實呈現含雜訊時傳統法改善有限——正是需要深度學習的理由，並建立其對照基準線。"),
 ("03","水果好壞影像分類","Fruit Freshness Classifier",
  "香蕉 good / bad 以 MobileNetV2 遷移學習做二元分類。深化程式含資料增強、微調(fine-tune)、"
  "早停、混淆矩陣與 Grad-CAM 可解釋性。",
  ["TensorFlow","transfer-learning","Grad-CAM"],[],
  "進階實作(需你本機 GPU/資料訓練)；延伸：部署成 Flask / Gradio 網頁 Demo。"),
 ("05","自動駕駛場景模擬","Autonomous Driving Sim",
  "深化重點：用 Python <b>從零解析 OpenDRIVE(.xodr)</b> 道路格式，由 line/arc 幾何積分重建"
  "道路參考線，並繪製整張道路網——證明對自駕地圖格式的底層理解。",
  ["OpenDRIVE","XML parsing","RoadRunner","Unity"],["road"],
  "真成果：解析你 RoadRunner 專案的實際地圖——138 條道路、10 個路口、167 條行車道、總長 4.63 km，"
  "並重建出含圓環與交叉路口的道路地圖。"),
 ("06","YOLOv8 多鹿種偵測","Deer Species Detection",
  "用 <b>YOLOv8</b> 偵測並<b>辨識鹿的品種</b>(白尾鹿/紅鹿/梅花鹿/黇鹿/馴鹿)。從 iNaturalist 真實照片"
  "建資料集，用通用鹿偵測器自動標框(弱監督)，微調多類別模型。已打包成可本地端執行的完整專案。",
  ["YOLOv8","ultralytics","iNaturalist","weak-supervision"],["deer","deer_curves"],
  "真成果：5 鹿種、225 張真實照片，CPU 實跑得 <b>mAP@0.5 = 0.34</b>，能區分五個品種(見圖，每種一例)。"
  "誠實標註：弱監督自動標框有雜訊、資料少故 mAP 不高——資料擴充+GPU 訓練可顯著提升；"
  "另附單類別版(mAP50=0.77)作對照。"),
]

def card(p):
    num,zh,en,desc,tags,figs,note = p
    imgs = "".join(f'<img src="{FIG[f]}" alt="{en}" loading="lazy">' for f in figs if FIG.get(f))
    imgblock = f'<div class="figs">{imgs}</div>' if imgs else ""
    tagsh = "".join(f"<span>{t}</span>" for t in tags)
    return f"""
    <article class="card">
      <div class="cardhead"><span class="num">{num}</span>
        <div><h3>{zh}</h3><p class="en">{en}</p></div></div>
      <p class="desc">{desc}</p>
      {imgblock}
      <div class="tags">{tagsh}</div>
      <p class="note">✦ {note}</p>
    </article>"""

cards = "\n".join(card(p) for p in PROJECTS)

HTML = f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>林信廷 · Data Science & AI Portfolio</title>
<style>
:root{{--bg:#0b1120;--card:#111a2e;--ink:#e5edff;--mute:#93a4c4;--line:#22304d;
  --accent:#3b82f6;--accent2:#f43f5e;}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,"Noto Sans TC",Segoe UI,system-ui,sans-serif;
  background:var(--bg);color:var(--ink);line-height:1.65}}
.wrap{{max-width:960px;margin:0 auto;padding:0 22px}}
header{{padding:70px 0 40px;border-bottom:1px solid var(--line)}}
h1{{font-size:2.5rem;margin:0 0 8px;letter-spacing:-.5px}}
.sub{{color:var(--accent);font-weight:600;margin:0 0 18px}}
.lead{{color:var(--mute);max-width:660px;font-size:1.05rem}}
.pills{{display:flex;gap:8px;flex-wrap:wrap;margin-top:22px}}
.pills span{{background:var(--card);border:1px solid var(--line);color:var(--mute);
  padding:5px 12px;border-radius:99px;font-size:.82rem}}
section{{padding:44px 0}}
h2{{font-size:1.4rem;margin:0 0 22px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;
  padding:24px;margin-bottom:22px}}
.cardhead{{display:flex;gap:14px;align-items:center;margin-bottom:8px}}
.num{{font-size:1.5rem;font-weight:800;color:var(--accent);
  background:#0d1830;border:1px solid var(--line);border-radius:12px;
  min-width:52px;height:52px;display:flex;align-items:center;justify-content:center}}
.card h3{{margin:0;font-size:1.2rem}}
.en{{margin:0;color:var(--mute);font-size:.85rem;letter-spacing:.3px}}
.desc{{color:#cdd8f0;margin:.4rem 0 1rem}}
.figs{{display:grid;gap:12px;margin:14px 0}}
.figs img{{width:100%;border-radius:10px;border:1px solid var(--line);background:#fff}}
.tags{{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}}
.tags span{{background:#0d1830;border:1px solid var(--line);color:var(--accent);
  padding:3px 10px;border-radius:7px;font-size:.76rem;font-family:ui-monospace,monospace}}
.note{{color:var(--mute);font-size:.9rem;border-left:3px solid var(--accent2);
  padding-left:12px;margin:10px 0 0}}
footer{{border-top:1px solid var(--line);padding:34px 0;color:var(--mute);font-size:.85rem}}
a{{color:var(--accent)}}
</style></head><body>
<header><div class="wrap">
  <h1>林信廷</h1>
  <p class="sub">Data Science &amp; AI Portfolio</p>
  <p class="lead">把大學課程作業系統性<b>深化</b>為五個研究取向的專案，涵蓋金融資料科學、
  電腦視覺、語音訊號處理與模擬。每個專案都用真實資料跑出成果，並堅持：
  <b>有基準對照、主動檢查偏誤、不誇大結果、程式可重現</b>。</p>
  <div class="pills"><span>Python</span><span>機器學習</span><span>電腦視覺</span>
  <span>語音訊號處理</span><span>金融資料科學</span><span>時間序列</span></div>
</div></header>
<section><div class="wrap">
  <h2>專案 Projects</h2>
  {cards}
</div></section>
<footer><div class="wrap">
  以真實資料與誠實評估建構 · 完整程式與升級規劃見專案 repo 的 README 與 ROADMAP。<br>
  © 林信廷 · Built as a study-abroad &amp; fintech portfolio.
</div></footer>
</body></html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
open(out, "w", encoding="utf-8").write(HTML)
print("wrote", out, f"({len(HTML)//1024} KB)")
