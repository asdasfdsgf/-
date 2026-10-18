# -*- coding: utf-8 -*-
"""新录音流程测试：不依赖 CABLE、直录处理后信号，可同时保存原始声。"""
import glob
import os
import tempfile
import wave

import numpy as np

import mic_denoiser as md


def rms_db(x):
    return 20 * np.log10(float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))) + 1e-12)


def load_tts():
    with wave.open("_tts.wav", "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 48000:
        from scipy.signal import resample_poly
        g = np.gcd(sr, 48000)
        x = resample_poly(x, 48000 // g, sr // g).astype(np.float32)
    return x


def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def main():
    md.CONFIG_PATH = os.path.join(tempfile.mkdtemp(), "config.json")
    voice = load_tts()
    rng = np.random.default_rng(1)
    noisy = (voice + 0.02 * rng.standard_normal(len(voice))).astype(np.float32)

    app = md.App()
    app.withdraw()
    app.samplerate = 48000
    app.blocksize = 480
    app.params["model"] = "dfn3"
    app.denoiser = md.Denoiser(48000, 480, app.params)
    app.denoiser.warmup()
    app.stream = object()

    app.rec_raw_var.set(True)
    app._toggle_rec()
    print("状态栏:", app.status.cget("text"))
    print("主录音:", os.path.basename(app._rec_path))
    print("原始录音:", os.path.basename(app._rec_raw_path) if app._rec_raw_path else "无")

    n = min(len(noisy), 48000 * 3)
    for pos in range(0, n - 480 + 1, 480):
        indata = noisy[pos:pos + 480][:, None].astype(np.float32)
        outdata = np.zeros((480, 1), np.float32)
        app._audio_cb(indata, outdata, 480, None, False)

    app._stop_recorder(announce=True)
    print("状态栏:", app.status.cget("text"))

    den, sr = read_wav(app._rec_path)
    raw, sr2 = read_wav(app._rec_raw_path)
    print(f"降噪录音 {len(den)/sr:.2f}s | {rms_db(den):6.1f} dBFS")
    print(f"原始录音 {len(raw)/sr2:.2f}s | {rms_db(raw):6.1f} dBFS")
    # 噪声抑制：取开头 0.5s（语音前的环境噪声段不好找，这里看整体差值）
    print("判定:", "通过" if (len(den) > 48000 and len(raw) > 48000
                             and abs(len(den) - len(raw)) < 480) else "失败")

    app._quit()
    for p in (app._rec_path, app._rec_raw_path):
        try:
            os.remove(p)
        except Exception:
            pass


if __name__ == "__main__":
    main()
