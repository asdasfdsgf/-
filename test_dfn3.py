# -*- coding: utf-8 -*-
"""对比 dfn3.py（numpy+ORT，主环境 3.13）与 golden.npz（torch+libdf 黄金基准）。

逐帧比对：ERB 特征 / LSNR / 掩码 / DF 系数 / 增强谱 / 输出波形。
"""
import os
import time

import numpy as np

import dfn3

HERE = os.path.dirname(os.path.abspath(__file__))


def db(x):
    x = np.asarray(x)
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
    return 20.0 * np.log10(r)


def main():
    print("available:", dfn3.available(), "| model:", dfn3.find_model_dir())
    g = np.load(os.path.join(HERE, "models", "df3", "golden.npz"))
    x = g["x"]
    g_out = g["out"]
    hop = 480
    nblk = len(x) // hop

    eng = dfn3.Dfn3Engine()
    eng.start_debug()
    outs = []
    t0 = time.perf_counter()
    for i in range(nblk):
        outs.append(eng.process(x[i * hop:(i + 1) * hop]))
    dt = time.perf_counter() - t0
    out = np.concatenate(outs)

    dur = nblk * hop / 48000.0
    print(f"app engine: {dur:.2f}s audio in {dt:.3f}s | RTF={dt/dur:.4f} "
          f"(约 {dt/dur*100:.1f}% 单核)")

    d_out = np.abs(out[:len(g_out)] - g_out[:len(outs) * hop])
    print(f"[波形]  max|diff|={d_out.max():.3e}  rms={np.sqrt((d_out**2).mean()):.3e}")

    def cmp(name, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        n = min(len(a), len(b))
        d = np.abs(a[:n] - b[:n])
        print(f"[{name}] max|diff|={d.max():.3e}  mean={d.mean():.3e}")

    cmp("ERB特征", np.array(eng.rec["feat"]), g["feat"])
    cmp("LSNR", np.array(eng.rec["lsnr"]), g["lsnr"])
    cmp("掩码m", np.array(eng.rec["m"]), g["m"])
    cmp("DF系数", np.array(eng.rec["coefs"]), g["coefs"])
    cmp("增强谱", np.array(eng.rec["spec"]), g["spec"])
    raw_app = np.array(eng.rec["raw"])
    cmp("分析谱", raw_app, g["raw"])
    raw_d = np.abs(raw_app - g["raw"]).max(axis=1)
    worst = np.argsort(raw_d)[-5:]
    print("分析谱差异最大的帧:", [(int(i), float(raw_d[i])) for i in worst])

    # 降噪效果：静音段（噪声底）抑制量
    print(f"\n输入噪声底(0.3-0.85s): {db(x[int(0.3*48000):int(0.85*48000)]):.1f} dBFS")
    print(f"输出噪声底(0.3-0.85s): {db(out[int(0.3*48000):int(0.85*48000)]):.1f} dBFS")
    sp = slice(int(3.2 * 48000), int(3.8 * 48000))
    print(f"输入语音段 RMS: {db(x[sp]):.1f} dBFS | 输出: {db(out[sp]):.1f} dBFS")

    # 保存可试听输出
    import wave
    wav = os.path.join(HERE, "models", "df3", "app_out.wav")
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())
    print("saved:", wav)


if __name__ == "__main__":
    main()
