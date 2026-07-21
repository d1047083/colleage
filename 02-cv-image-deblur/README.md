# 02 · 影像去模糊 (Image Deblurring)

原作業把影像做「模糊/清晰」二元分類——這在概念上是錯的(且原程式有 bug，已修)。
真正該解的是**影像還原**：由模糊影像重建清晰影像。

- `src/classical_deblur.py`：傳統基準(Unsharp、Richardson-Lucy)，**可直接執行**。
- `src/unet_deblur.py`：U-Net 深度學習訓練骨架(需 GPU 與成對 blur/sharp 資料)。

評估建議用 PSNR / SSIM，並以傳統法為基準。
