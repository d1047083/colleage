"""
下載鹿類偵測資料集 (Roboflow)
=================================
1. 免費註冊 https://roboflow.com 取得 API key(Settings → API)。
2. 在 https://universe.roboflow.com 搜尋 "deer" 選一個偵測資料集。
   注意：多數公開 deer 資料集是「單類別(deer)」；若要真正的「鹿種(species)」，
   需找 species 標註的資料集，或自行標註(見 README)。
3. 填入下方 workspace/project/version 與你的 API key 後執行。
用法: python download_dataset.py
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
