"""
從 Roboflow 下載鹿的偵測資料集。
先到 roboflow.com 免費註冊拿 API key，再填下面的 workspace/project/version。
公開的 deer 資料集多半是單類別，要分品種得找有物種標註的，或自己標。
"""
import os
from roboflow import Roboflow

API_KEY   = os.environ.get("ROBOFLOW_API_KEY", "在此貼上你的_API_KEY")
WORKSPACE = "university-itgbn"     # 範例: 可替換成你選的資料集
PROJECT   = "deer-ipsw2"
VERSION   = 2

rf = Roboflow(api_key=API_KEY)
ds = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov8", location="../dataset")
print("資料集已下載到 ../dataset ；其 data.yaml 內含類別名稱。")
