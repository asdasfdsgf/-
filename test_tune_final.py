# -*- coding: utf-8 -*-
"""评估：清晰语音段电平跨度 / 停顿底噪 / 舒适噪声频谱。"""
import wave

import numpy as np

import mic_denoiser as md

SR = 48000


def load(p):
    with wave.open(p, "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16) \
            .astype(np.float32) / 32768.0


def env(x, win=480):
    m = len(x) // win
    f = x[:m * win].astype(np.float64).reshape(m, win)
    return (20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-12)).astype(np.float32)


def run(raw, model="dfn3"):
    p = dict(md.DEFAULT_PARAMS)
    p["model"] = model
    den = md.Denoiser(SR, 480, p)
    den.warmup()
    out = np.zeros_like(raw)
    for i in range(len(raw) // 480):
        out[i * 480:(i + 1) * 480] = den.process(raw[i * 480:(i + 1) * 480])
    return out


def main():
    raw = load(r"dist\录音\原始_20261018_093149.wav")
    old = load(r"dist\录音\降噪_20261018_093149.wav")
    new = run(raw)

    n = min(len(raw), len(old), len(new))
    er, eo, en = env(raw)[:n // 480], env(old)[:n // 480], env(new)[:n // 480]
    z = np.abs(raw[:len(er) * 480]).reshape(len(er), 480).max(axis=1) == 0
    bad = np.zeros(len(er), bool)
    for i in np.where(z)[0]:
        bad[max(0, i - 8):min(len(er), i + 9)] = True

    loud = (er > np.percentile(er[~bad], 70)) & ~bad        # 清晰发声段
    quiet = (er < np.percentile(er[~bad], 25)) & ~bad       # 近底噪段
    for tag, e in (("旧", eo), ("新", en)):
        s = e[loud]
        q = e[quiet]
        print(f"[{tag}] 清晰语音 10-90% 跨度 {np.percentile(s,90)-np.percentile(s,10):5.1f} dB"
              f" | 标准差 {s.std():5.1f} dB | 停顿底噪中位 {np.median(q):6.1f} dB")
    print(f"     输入语音中位 {np.median(er[loud]):.1f} dBFS | 输入停顿中位 {np.median(er[quiet]):.1f} dBFS")

    # 停顿段频谱形状（1s 内的静止段）
    seg_start = None
    idx = np.where(quiet)[0]
    if len(idx):
        seg_start = int(idx[0] * 480)
        seg = new[seg_start:seg_start + 9600]
        sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        low = sp[(f > 100) & (f < 1000)].sum()
        mid = sp[(f > 1000) & (f < 3000)].sum()
        high = sp[(f > 3000) & (f < 12000)].sum()
        print(f"     停顿段能量分布: 低频(100-1k) {10*np.log10(low+1e-20):.1f} | "
              f"中频(1-3k) {10*np.log10(mid+1e-20):.1f} | 高频(3-12k) {10*np.log10(high+1e-20):.1f}")
    rr = np.sqrt((new.astype(np.float64) ** 2).mean())
    print(f"     新输出 RMS {20*np.log10(rr):.1f} dBFS | 峰值 {np.abs(new).max():.3f}")

    with wave.open("_new_den.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(new, -1, 1) * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    main()
