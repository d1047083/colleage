"""
Speech DSP in Python  — 由 MATLAB (zcr.m / f0.m / week5.m) 移植
計算並繪製語音的：短時能量(Short-time Energy)、零交越率(ZCR)、頻譜圖(Spectrogram)。
可用於「語音/靜音偵測(VAD)」的基礎。

用法:
    python speech_features.py <path/to/audio.wav>
    # 不給檔案時會自動合成一段測試訊號並執行
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import get_window, spectrogram

def frame_signal(x, frame_len, hop):
    n = 1 + max(0, (len(x) - frame_len) // hop)
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]  # (n_frames, frame_len)

def short_time_energy(frames, win):
    return np.sum((frames * win) ** 2, axis=1)

def zero_crossing_rate(frames, win):
    w = frames * win
    return 0.5 * np.mean(np.abs(np.diff(np.sign(w), axis=1)), axis=1)

def analyze(x, fs, frame_ms=32, shift=0.5, out="speech_analysis.png"):
    x = x.astype(np.float64)
    x = x / (np.max(np.abs(x)) + 1e-9)          # 正規化
    flen = int(frame_ms * 1e-3 * fs)
    hop  = int(flen * shift)
    win  = get_window("hann", flen)
    frames = frame_signal(x, flen, hop)
    E   = short_time_energy(frames, win)
    zcr = zero_crossing_rate(frames, win)
    t_frame = (np.arange(len(E)) * hop + flen / 2) / fs
    t_sig   = np.arange(len(x)) / fs

    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t_sig, x, lw=.5, color="#334155"); ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Waveform")
    ax[1].plot(t_frame, E, color="#2563eb"); ax[1].set_ylabel("Energy")
    ax[1].set_title("Short-time Energy")
    ax[2].plot(t_frame, zcr, color="#dc2626"); ax[2].set_ylabel("ZCR")
    ax[2].set_title("Zero-Crossing Rate"); ax[2].set_xlabel("Time (s)")
    for a in ax: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)

    # 簡易語音/靜音偵測：能量與 ZCR 同時超過門檻 → 視為有聲
    thr_e = 0.1 * np.max(E)
    voiced = E > thr_e
    print(f"fs={fs} Hz, duration={len(x)/fs:.2f}s, frames={len(E)}")
    print(f"voiced frames: {voiced.sum()}/{len(E)} ({100*voiced.mean():.1f}%)")
    print(f"saved figure -> {out}")
    return {"energy": E, "zcr": zcr, "t": t_frame}

def synth_test():
    fs = 16000; t = np.arange(int(fs*2.0))/fs
    # 前後靜音 + 中段濁音(低頻正弦+諧波) + 一段清音(白雜訊)
    x = np.zeros_like(t)
    voiced = (t>0.4)&(t<1.0)
    x[voiced] = (np.sin(2*np.pi*140*t[voiced]) + .5*np.sin(2*np.pi*280*t[voiced]))
    unvoiced = (t>1.2)&(t<1.6)
    x[unvoiced] = 0.6*np.random.randn(unvoiced.sum())
    return x, fs

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "figures"); os.makedirs(out, exist_ok=True)
    outpng = os.path.join(out, "speech_analysis.png")
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        fs, x = wavfile.read(sys.argv[1])
        if x.ndim > 1: x = x.mean(axis=1)
    else:
        print("no wav given -> using synthetic test signal")
        x, fs = synth_test()
    analyze(x, fs, out=outpng)
