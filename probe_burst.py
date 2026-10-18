# -*- coding: utf-8 -*-
"""探针：区分人声/齿音/纸张摩擦的可分特征（平坦度 + 基频周期强度）。"""
import wave

import numpy as np
from scipy.signal import butter, sosfilt

SR = 48000


def load_tts():
    with wave.open("_tts.wav", "rb") as wf:
        sr = wf.getframerate()
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    x = x.astype(np.float32) / 32768.0
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def feats(frame):
    w = np.hanning(len(frame))
    sp = np.abs(np.fft.rfft(frame * w)).astype(np.float64)
    f = np.fft.rfftfreq(len(frame), 1 / SR)
    band = (f >= 1000) & (f <= 8000)
    m = sp[band] + 1e-12
    flat = np.exp(np.mean(np.log(m))) / np.mean(m)

    # 基频周期强度：80-400Hz 带通后做归一化自相关
    sos = butter(2, [80 / (SR / 2), 400 / (SR / 2)], btype="band", output="sos")
    y = sosfilt(sos, frame)
    r = np.correlate(y, y, "full")[len(y) - 1:]
    if r[0] <= 1e-12:
        pitch = 0.0
    else:
        lo, hi = int(SR / 400), min(int(SR / 80), len(r) - 1)
        pitch = float(np.max(r[lo:hi]) / r[0])
    return flat, pitch


def stats(tag, sig, win=480):
    fl, pi = [], []
    for i in range(len(sig) // win):
        a, b = feats(sig[i * win:(i + 1) * win])
        fl.append(a); pi.append(b)
    fl, pi = np.array(fl), np.array(pi)
    print(f"[{tag:10s}] 平坦度 中位 {np.median(fl):.3f} (10-90% {np.percentile(fl,10):.3f}-{np.percentile(fl,90):.3f})"
          f" | 周期强度 中位 {np.median(pi):.3f} (10-90% {np.percentile(pi,10):.3f}-{np.percentile(pi,90):.3f})")
    return fl, pi


def main():
    rng = np.random.default_rng(5)
    tts = load_tts()
    tts = tts * (0.05 / (np.sqrt(np.mean(tts ** 2)) + 1e-12))

    # 只看有声部分（跳过静音）
    env = np.sqrt(np.convolve(tts ** 2, np.ones(480) / 480, "same"))
    voiced = tts[env > np.percentile(env, 60)]
    stats("人声", voiced)

    # 齿音段：4-10kHz 能量占比高的帧
    sp_all = np.abs(np.fft.rfft(tts[:len(tts) // 480 * 480].reshape(-1, 480) * np.hanning(480), axis=1))
    f = np.fft.rfftfreq(480, 1 / SR)
    hi = sp_all[:, (f > 4000)].sum(axis=1)
    tot = sp_all.sum(axis=1) + 1e-12
    ratio = hi / tot
    idx = np.where((ratio > np.percentile(ratio, 90)) & (env[:len(ratio)*480:480] > np.percentile(env, 40)))[0]
    if len(idx):
        stats("齿音s/sh", np.concatenate([tts[i * 480:(i + 1) * 480] for i in idx[:40]]))

    sos = butter(2, [800 / (SR / 2), 6000 / (SR / 2)], btype="band", output="sos")
    for name, peak in (("纸张(0.04)", 0.04), ("纸张(0.01)", 0.01)):
        n = int(SR * 0.15) * 6
        r = sosfilt(sos, rng.standard_normal(n)) * np.tile(np.hanning(int(SR * 0.15)), 6)
        r = (r / (np.abs(r).max() + 1e-9) * peak).astype(np.float32)
        stats(name, r)

    k = int(SR * 0.012) * 6
    t = np.arange(k) / SR
    sos_k = butter(2, [2000 / (SR / 2), 9000 / (SR / 2)], btype="band", output="sos")
    click = sosfilt(sos_k, rng.standard_normal(k)) * np.exp(-t / 0.0025)
    click = (click / (np.abs(click).max() + 1e-9) * 0.05).astype(np.float32)
    stats("键盘", click)


if __name__ == "__main__":
    main()
