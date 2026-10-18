# -*- coding: utf-8 -*-
"""突发检测 A/B：语音段是否被误伤 / 纸张与键盘是否被压掉。"""
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
    mix = v + (0.004 * rng.standard_normal(n)).astype(np.float32)

    sos_r = butter(2, [800 / (SR / 2), 6000 / (SR / 2)], btype="band", output="sos")
    rustles = []
    for p in (1.6, 3.8):
        nr = int(SR * 0.15)
        r = sosfilt(sos_r, rng.standard_normal(nr)) * np.hanning(nr)
        r = (r / (np.abs(r).max() + 1e-9) * 0.04).astype(np.float32)
        mix[int(p * SR):int(p * SR) + nr] += r
        rustles.append(p)
    return mix.astype(np.float32), v, rustles


def run(mix, burst_on=True):
    p = dict(md.DEFAULT_PARAMS)
    p["model"] = "dfn3"
    den = md.Denoiser(SR, 480, p)
    den.warmup()
    if not burst_on:
        den._burst.process = lambda *a, **k: False
    out = np.zeros_like(mix)
    for i in range(len(mix) // 480):
        out[i * 480:(i + 1) * 480] = den.process(mix[i * 480:(i + 1) * 480])
    return out


def env(x, win=480):
    m = len(x) // win
    f = x[:m * win].astype(np.float64).reshape(m, win)
    return (20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-12)).astype(np.float32)


def main():
    mix, voice, rustles = build()
    a = run(mix, burst_on=False)
    b = run(mix, burst_on=True)

    ev = env(voice)
    speech = ev > np.percentile(ev, 70)
    ea, eb = env(a), env(b)
    n = min(len(ea), len(eb), len(ev))
    d = (eb[:n] - ea[:n])[speech[:n]]
    print(f"语音段：开关后的电平差 中位 {np.median(d):+.1f} dB | 10%分位 {np.percentile(d,10):+.1f} dB"
          f" | 被压 >6dB 的语音帧 {np.mean(d < -6)*100:.1f}%")

    for t in rustles:
        i = int(t / 0.01)
        lv_a = rms_db(a[int(t * SR):int((t + 0.15) * SR)])
        lv_b = rms_db(b[int(t * SR):int((t + 0.15) * SR)])
        print(f"纸张 @{t}s:  {lv_a:6.1f} → {lv_b:6.1f} dBFS（压掉 {lv_a-lv_b:5.1f} dB）")

    with wave.open("_burst_on.wav", "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(b, -1, 1) * 32767).astype(np.int16).tobytes())
    print("试听: _burst_on.wav")


if __name__ == "__main__":
    main()
