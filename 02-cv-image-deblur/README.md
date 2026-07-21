# 02 · 影像去模糊

原本的作業是把圖分成「模糊」和「清晰」兩類。這個定義有點怪，程式也有 bug（已經修掉）。這個題目真正該做的是還原：把一張模糊的圖變清楚。

這裡先放兩支能直接跑的東西：

- `src/classical_deblur.py`：傳統方法（Unsharp、Richardson-Lucy），不用訓練就能跑。
- `src/deblur_benchmark.py`：對已知的模糊加雜訊，用傳統法還原，再算 PSNR/SSIM 看效果。
- `src/unet_deblur.py`：U-Net 的訓練程式，這支要有 GPU 和成對的 blur/sharp 資料才跑得動。

評估用 PSNR 和 SSIM，並拿傳統法當基準。實測時傳統法在有雜訊的情況下改善其實有限，這也是為什麼後面值得換成深度學習模型。
