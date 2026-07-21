"""
解析 RoadRunner 匯出的 OpenDRIVE(.xodr)，把道路網畫出來。
從每條路的直線和圓弧幾何，用起點座標、朝向、曲率積分還原座標，
再統計道路數、車道數、路口數和總長。
用法：python opendrive_parser.py <file.xodr>。
"""
import sys, os, math
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def geom_points(g, ds=1.0):
    s0=float(g.get("s")); x0=float(g.get("x")); y0=float(g.get("y"))
    hdg=float(g.get("hdg")); L=float(g.get("length"))
    n=max(2,int(L/ds)); ss=np.linspace(0,L,n)
    child=g[0]
    if child.tag=="line":
        x=x0+ss*math.cos(hdg); y=y0+ss*math.sin(hdg)
    elif child.tag=="arc":
        k=float(child.get("curvature"))
        if abs(k)<1e-12:
            x=x0+ss*math.cos(hdg); y=y0+ss*math.sin(hdg)
        else:
            th=hdg+k*ss
            x=x0+(np.sin(th)-math.sin(hdg))/k
            y=y0-(np.cos(th)-math.cos(hdg))/k
    else:  # 其他型別(spiral等)這份檔沒有，退化為直線
        x=x0+ss*math.cos(hdg); y=y0+ss*math.sin(hdg)
    return x,y

def parse(path):
    root=ET.parse(path).getroot()
    roads=root.findall("road"); juncs=root.findall("junction")
    total_len=0.0; lane_count=0; segs=[]; junction_road_ids=set()
    for rd in roads:
        total_len+=float(rd.get("length",0))
        is_junc = rd.get("junction","-1")!="-1"
        pts=[geom_points(g) for g in rd.findall("./planView/geometry")]
        segs.append((pts,is_junc))
        ls=rd.find(".//laneSection")
        if ls is not None:
            for side in ("left","right"):
                s=ls.find(side)
                if s is not None:
                    lane_count+=sum(1 for ln in s.findall("lane") if ln.get("type")=="driving")
    return roads,juncs,total_len,lane_count,segs

def plot(segs, out, title):
    fig,ax=plt.subplots(figsize=(9,9))
    for pts,is_junc in segs:
        for x,y in pts:
            ax.plot(x,y,color=("#dc2626" if is_junc else "#2563eb"),
                    lw=(2.2 if is_junc else 1.0),alpha=.9)
    ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=.3)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0],[0],color="#2563eb"),Line2D([0],[0],color="#dc2626",lw=2.2)],
              ["road","junction road"],loc="upper right")
    fig.tight_layout(); fig.savefig(out,dpi=120); plt.close(fig)

if __name__=="__main__":
    HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
    FIG=os.path.join(ROOT,"figures"); RES=os.path.join(ROOT,"results")
    for d in (FIG,RES): os.makedirs(d,exist_ok=True)
    path=sys.argv[1] if len(sys.argv)>1 else \
        "/mnt/user-data/uploads/Desktop/05_自動駕駛模擬_RoadRunner_Unity/RoadRunner專案_road/test.xodr"
    roads,juncs,total_len,lanes,segs=parse(path)
    out=os.path.join(FIG,"deep_road_network.png")
    plot(segs,out,f"OpenDRIVE road network — {len(roads)} roads, {len(juncs)} junctions")
    md=[f"# OpenDRIVE 解析\n",
        f"檔案 {os.path.basename(path)}。解析出 {len(roads)} 條路、{len(juncs)} 個路口、{lanes} 條行車道，"
        f"總長 {total_len:,.0f} 公尺（{total_len/1000:.2f} 公里）。"
        "道路的幾何是從直線和圓弧兩種基本型，用起點座標、朝向、曲率積分還原出來的。\n",
        "道路網的圖在 figures/deep_road_network.png（紅色是路口的路段）。"
        "有了參考線和車道資訊，之後就可以放車輛加相機，接專案 02 的視覺模型做車道偵測。\n"]
    open(os.path.join(RES,"road_metrics.md"),"w",encoding="utf-8").write("\n".join(md))
    print("\n".join(md)); print("saved ->",out)
