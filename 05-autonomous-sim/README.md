# 05 · 自動駕駛場景

原本是用 MATLAB RoadRunner 建的道路場景（輸出成 OpenDRIVE 的 .xodr、.fbx、GeoJSON），再匯進 Unity 做視覺化，另外有三份簡報。

這次補的部分是用 Python 從頭解析 .xodr：讀出每條路的幾何（直線和圓弧），積分還原成座標，再把整張道路網畫出來。程式在 `src/opendrive_parser.py`。實際跑那份 RoadRunner 檔的結果是 138 條路、10 個路口、總長 4.63 公里。

模擬環境本身沒辦法在這裡重跑，所以這個專案偏技術文件。之後可以在 Unity 裡放一台車加相機，接專案 02 的視覺模型做車道或號誌辨識，或換 CARLA 這類開源模擬器用程式生成場景。
