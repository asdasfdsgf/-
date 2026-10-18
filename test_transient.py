# -*- coding: utf-8 -*-
"""场景测试：语音 + 键盘 + 纸张 + 底噪 → 输出里剩多少非人声。"""
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


def click(rng, peak=0.05):
    n = int(SR * 0.012)
    t = np.arange(n) / SR
    sos = butter(2, [2000 / (SR / 2), 9000 / (SR / 2)], btype="band", output="sos")
    x = sosfilt(sos, rng.standard_normal(n)) * np.exp(-t / 0.0025)
    return (x / (np.abs(x).max() + 1e-9) * peak).astype(np.float32)


def rustle(rng, peak=0.04):
    n = int(SR * 0.15)
    sos = butter(2, [800 / (SR / 2), 6000 / (SR / 2)], btype="band", output="sos")
    x = sosfilt(sos, rng.standard_normal(n)) * np.hanning(n)
    return (x / (np.abs(x).max() + 1e-9) * peak).astype(np.float32)


def main():
    rng = np.random.default_rng(5)
    voice = load_tts()[:int(SR * 5)]
    voice = voice * (0.05 / (np.sqrt(np.mean(voice ** 2)) + 1e-12))   # -26dBFS
    n = len(voice) + SR // 2
    v = np.zeros(n, np.float32)
    v[SR // 4:SR // 4 + len(voice)] = voice

    mix = v + 0.004 * rng.standard_normal(n).astype(np.float32)
    cpos = [0.35, 1.0, 2.2, 3.0, 4.1, 4.9]
    for p in cpos:
        c = click(rng)
        mix[int(p * SR):int(p * SR) + len(c)] += c
    for p in (1.6, 3.8):
        r = rustle(rng)
        mix[int(p * SR):int(p * SR) + len(r)] += r

    mix = mix.astype(np.float32)
    orig_prof = dict(md.DFN3_PROFILE)
    for vad in (-8.0, -10.0, -12.0):
        md.DFN3_PROFILE = dict(orig_prof, vad_lsnr=vad)
        p = dict(md.DEFAULT_PARAMS)
        p["model"] = "dfn3"
        den = md.Denoiser(SR, 480, p)
        den.warmup()
        out = np.zeros_like(mix)
        for i in range(len(mix) // 480):
            out[i * 480:(i + 1) * 480] = den.process(mix[i * 480:(i + 1) * 480])

        def at2(x, t, ms=40):
            i = int(t * SR)
            return rms_db(x[i:i + int(ms * SR / 1000)])

        sp = at2(out, 2.6)
        print(f"[vad={vad:.0f}] 语音 {sp:6.1f} | 键盘 {np.mean([at2(out, t) for t in cpos]):6.1f}"
              f"（{np.mean([at2(out, t) for t in cpos]) - sp:+5.1f}）"
              f" | 纸张 {at2(out, 1.6):6.1f}（{at2(out, 1.6) - sp:+5.1f}）"
              f" | 整段 {rms_db(out):6.1f}")
    md.DFN3_PROFILE = orig_prof

    md.DFN3_PROFILE = dict(orig_prof, vad_lsnr=-8.0)
    p = dict(md.DEFAULT_PARAMS)
    p["model"] = "dfn3"
    den = md.Denoiser(SR, 480, p)
    den.warmup()
    out = np.zeros_like(mix)
    for i in range(len(mix) // 480):
        out[i * 480:(i + 1) * 480] = den.process(mix[i * 480:(i + 1) * 480])
    md.DFN3_PROFILE = orig_prof

    def at(x, t, ms=40):
        i = int(t * SR)
        return rms_db(x[i:i + int(ms * SR / 1000)])

    speech_t = 2.6
    print(f"输入: 语音 {at(mix, speech_t):.1f} | 键盘 {np.mean([at(mix, t) for t in cpos]):.1f}"
          f" | 整段 RMS {rms_db(mix):.1f}")
    print(f"输出: 语音 {at(out, speech_t):.1f} | 键盘 {np.mean([at(out, t) for t in cpos]):.1f}"
          f" | 整段 RMS {rms_db(out):.1f}")
    print(f"键盘相对语音: 输入 {np.mean([at(mix, t) for t in cpos]) - at(mix, speech_t):+.1f} dB"
          f" → 输出 {np.mean([at(out, t) for t in cpos]) - at(out, speech_t):+.1f} dB")
    print(f"纸张相对语音: 输入 {at(mix, 1.6) - at(mix, speech_t):+.1f} dB"
          f" → 输出 {at(out, 1.6) - at(out, speech_t):+.1f} dB")

    with wave.open("_scene_out.wav", "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())
    with wave.open("_scene_in.wav", "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    print("试听: _scene_in.wav → _scene_out.wav")


if __name__ == "__main__":
    main()
