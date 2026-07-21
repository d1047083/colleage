# Deblurring — quantitative benchmark

施加已知高斯模糊(σ=3)+高斯雜訊(σ=0.01)後還原，對照原圖計算 PSNR/SSIM：

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Degraded (blur+noise) | 23.84 | 0.739 |
| Wiener | 23.36 | 0.721 |
| Richardson-Lucy | 19.72 | 0.764 |
| Unsharp | 24.56 | 0.654 |

- 最佳傳統法：**Unsharp** (PSNR 24.56 dB)。
- 這就是深度學習模型必須超越的基準線；U-Net 訓練程式見 src/unet_deblur.py。
- 評估用 PSNR(數值保真)與 SSIM(結構相似)雙指標，比只看肉眼更客觀。
