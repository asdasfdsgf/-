# -*- coding: utf-8 -*-
"""真实语音对比：TTS 中文 + 噪声 → 完整处理链（RNNoise vs DeepFilterNet3）。"""
import wave

import numpy as np
from scipy.signal import resample_poly

import mic_denoiser as md

SR = 48000


def load_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def save_wav(path, x):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def rms_db(x):
    return 20 * np.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12)


def band_db(x, lo, hi):
    if len(x) < 4096:
        return -120.0
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    m = (f >= lo) & (f < hi)
    return 10 * np.log10(float(sp[m].sum()) + 1e-12)


def run_chain(x, model):
    p = dict(md.DEFAULT_PARAMS)
    p["model"] = model
    den = md.Denoiser(SR, 480, p)
    den.warmup()
    out = np.zeros_like(x)
    for i in range(len(x) // 480):
        out[i * 480:(i + 1) * 480] = den.process(x[i * 480:(i + 1) * 480])
    return out


def main():
    voice = load_wav("_tts.wav")
    rng = np.random.default_rng(3)
    # 语音前留 0.6s 纯噪声、中间插一段 0.5s 纯噪声
    silence = np.zeros(int(0.6 * SR), np.float32)
    gap = np.zeros(int(0.5 * SR), np.float32)
    v = np.concatenate([silence, voice, gap, voice])
    noise = (0.010 * rng.standard_normal(len(v))
             + 0.004 * np.sin(2 * np.pi * 50 * np.arange(len(v)) / SR)).astype(np.float32)
    noisy = (v * 0.9 + noise).astype(np.float32)

    save_wav("_noisy.wav", noisy)
    print(f"输入: 语音 {rms_db(v):.1f} dBFS | 底噪 {rms_db(noise):.1f} dBFS")

    for model, tag in (("rnnoise", "rn"), ("dfn3", "dfn")):
        out = run_chain(noisy, model)
        save_wav(f"_out_{tag}.wav", out)
        # 底噪段：开头 0.6s（扣除 DFN3 的 30ms 延迟无影响）
        nb = slice(int(0.15 * SR), int(0.55 * SR))
        # 语音段：取第一段语音
        sp0 = int(0.6 * SR) + int(0.6 * SR)
        sg = slice(sp0, sp0 + int(2.0 * SR))
        print(f"\n[{tag}] 底噪段 {rms_db(out[nb]):6.1f} dBFS (抑制 {rms_db(noisy[nb]) - rms_db(out[nb]):5.1f} dB)"
              f" | 语音段 {rms_db(out[sg]):6.1f} dBFS")
        print(f"      4-6k 相对 500-2k: {band_db(out[sg], 4000, 6000) - band_db(out[sg], 500, 2000):+6.1f} dB"
              f" | 6-9k: {band_db(out[sg], 6000, 9000) - band_db(out[sg], 500, 2000):+6.1f} dB")


if __name__ == "__main__":
    main()
