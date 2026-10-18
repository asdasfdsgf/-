# -*- coding: utf-8 -*-
"""突发检测参数扫描：在"不误伤语音"和"压住纸张"之间找平衡。"""
import wave

import numpy as np
from scipy.signal import butter, sosfilt

import mic_denoiser as md

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


def rms_db(x):
    x = np.asarray(x, np.float64)
    return 20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-12)


def build():
    rng = np.random.default_rng(5)
    voice = load_tts()[:int(SR * 5)]
    voice = voice * (0.05 / (np.sqrt(np.mean(voice ** 2)) + 1e-12))
    n = len(voice) + SR // 2
    v = np.zeros(n, np.float32)
    v[SR // 4:SR // 4 + len(voice)] = voice
    mix = (v + (0.004 * rng.standard_normal(n)).astype(np.float32)).astype(np.float32)
    sos = butter(2, [800 / (SR / 2), 6000 / (SR / 2)], btype="band", output="sos")
    pos = []
    # 纸张声放在人声前后的停顿里（这是唯一真正听得见的场景）
    for p in (0.06, 5.35):
        nr = int(SR * 0.15)
        r = sosfilt(sos, rng.standard_normal(nr)) * np.hanning(nr)
        r = (r / (np.abs(r).max() + 1e-9) * 0.04).astype(np.float32)
        mix[int(p * SR):int(p * SR) + nr] += r
        pos.append(p)
    return mix, v, pos


def env(x, win=480):
    m = len(x) // win
    f = x[:m * win].astype(np.float64).reshape(m, win)
    return (20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-12)).astype(np.float32)


def run(mix, cfg):
    p = dict(md.DEFAULT_PARAMS)
    p["model"] = "dfn3"
    den = md.Denoiser(SR, 480, p)
    den.warmup()
    if cfg is None:
        den._burst.process = lambda *a, **k: False
    else:
        for k, v in cfg.items():
            setattr(den._burst, k, v)
    out = np.zeros_like(mix)
    for i in range(len(mix) // 480):
        out[i * 480:(i + 1) * 480] = den.process(mix[i * 480:(i + 1) * 480])
    return out


def main():
    mix, voice, pos = build()
    base = run(mix, None)
    eb = env(base)
    ev = env(voice)
    n = min(len(eb), len(ev))
    # 只统计"明确有声"的帧：原始人声大 且 输出电平不低
    speech = (ev[:n] > np.percentile(ev[:n], 70)) & (eb[:n] > -45.0)

    cfgs = {
        "need4 t.70": dict(thresh=0.70, need=4, hold_frames=8, level_margin=-999),
        "need6 t.70": dict(thresh=0.70, need=6, hold_frames=8, level_margin=-999),
        "need8 t.70": dict(thresh=0.70, need=8, hold_frames=8, level_margin=-999),
        "need8 t.66": dict(thresh=0.66, need=8, hold_frames=10, level_margin=-999),
    }
    for name, cfg in cfgs.items():
        out = run(mix, cfg)
        eo = env(out)
        d = (eo[:n] - eb[:n])[speech]
        sup = []
        for t in pos:
            a = rms_db(base[int(t * SR):int((t + 0.15) * SR)])
            b = rms_db(out[int(t * SR):int((t + 0.15) * SR)])
            sup.append(a - b)
        print(f"[{name:16s}] 语音被压>6dB 帧 {np.mean(d < -6)*100:4.1f}%"
              f" | 语音电平差中位 {np.median(d):+5.1f}dB"
              f" | 纸张压制 {sup[0]:4.1f}/{sup[1]:4.1f} dB")


if __name__ == "__main__":
    main()
