# -*- coding: utf-8 -*-
"""DeepFilterNet3 流式推理的黄金基准（torch 子网 + libdf 的 STFT/ISTFT）。

严格按 libDF/src/tract.rs 的 DfTract::process 逐帧复刻：
  分析 → ERB 特征(均值归一) + 复数特征(unit norm) → enc → 按 LSNR 分档 →
  erb_dec 掩码 / df_dec 系数 → 掩码应用到 y[t-conv_lookahead] →
  低频 96 bin 用 5 阶深度滤波覆盖 → 合成。

产物：models/df3/golden.npz（逐帧中间量）+ models/df3/golden_out.wav
供主程序（纯 numpy+onnxruntime）逐项比对。
"""
import json
import os
import time
import wave

import numpy as np
import torch

from export_dfn3 import DfDecWrap, EncWrap, ErbDecWrap, HERE, MODEL_DIR, OUT_DIR

ALPHA = np.float32(0.995)


def make_test_signal(sr=48000):
    """合成测试信号：谐波"人声" + 白噪/工频哼声 + 硬静音段（触发 skip 逻辑）。"""
    rng = np.random.default_rng(7)
    dur = 8.0
    n = int(sr * dur)
    t = np.arange(n) / sr

    f0 = 118.0 + 14.0 * np.sin(2 * np.pi * 0.23 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = np.zeros(n)
    for k in range(1, 41):
        a = (1.0 / k) * (1.0 + 0.55 * np.cos(2 * np.pi * (k - 8) / 10.0))
        voice += a * np.sin(k * phase + 0.3 * k)
    voice /= np.max(np.abs(voice))
    # 音节包络：3.5Hz 开合 + 两段静音
    env = np.clip(np.sin(2 * np.pi * 3.5 * t), 0.0, None) ** 1.5
    env[np.abs(t - 2.0) < 0.24] = 0.0
    env[np.abs(t - 5.2) < 0.35] = 0.0
    voice = voice * env * 0.16

    noise = 0.012 * rng.standard_normal(n) + 0.006 * np.sin(2 * np.pi * 50 * t)
    x = (voice + noise).astype(np.float32)
    # 一段纯数字零（0.9s~1.15s），验证 skip_counter 分支
    x[int(0.9 * sr):int(1.15 * sr)] = 0.0
    return x


def main():
    import df.logger as _lg
    import df.utils as _u
    for _m in (_u, _lg):
        for _n in ("get_commit_hash", "get_branch_name", "get_git_root"):
            if hasattr(_m, _n):
                setattr(_m, _n, lambda: "local")

    from df.enhance import init_df
    from df.model import ModelParams
    from libdf import DF

    model, _, _ = init_df(MODEL_DIR, post_filter=False, log_level="ERROR", log_file=None)[:3]
    model = model.cpu().eval()
    p = ModelParams()
    meta = json.load(open(os.path.join(OUT_DIR, "df3.json"), encoding="utf-8"))
    widths = np.array(meta["erb_widths"], dtype=np.int64)
    offs = np.concatenate(([0], np.cumsum(widths)))
    nb_erb, nb_df = p.nb_erb, p.nb_df
    order, conv_la = p.df_order, p.conv_lookahead
    lookahead = max(p.conv_lookahead, p.df_lookahead)
    nfreq = p.fft_size // 2 + 1
    hop = p.hop_size

    df = DF(p.sr, p.fft_size, p.hop_size, nb_erb, 2)
    enc_w = EncWrap(model.enc).eval()
    erb_w = ErbDecWrap(model.erb_dec).eval()
    df_w = DfDecWrap(model.df_dec).eval()

    mean_state = np.linspace(-60.0, -90.0, nb_erb).astype(np.float32)
    unit_state = np.linspace(0.001, 0.0001, nb_df).astype(np.float32)
    y_buf = [np.zeros(nfreq, np.complex64) for _ in range(order + conv_la)]
    x_buf = [np.zeros(nfreq, np.complex64) for _ in range(max(order, lookahead))]
    h_enc = torch.zeros(1, 1, p.emb_hidden_dim)
    h_erb = torch.zeros(p.emb_num_layers - 1, 1, p.emb_hidden_dim)
    h_df = torch.zeros(p.df_num_layers, 1, p.df_hidden_dim)

    x = make_test_signal(p.sr)
    nblk = len(x) // hop
    out = np.zeros(nblk * hop, dtype=np.float32)

    rec_feat = np.zeros((nblk, nb_erb), np.float32)
    rec_lsnr = np.zeros(nblk, np.float32)
    rec_m = np.zeros((nblk, nb_erb), np.float32)
    rec_coefs = np.zeros((nblk, order, nb_df, 2), np.float32)
    rec_spec = np.zeros((nblk, nfreq), np.complex64)
    rec_raw = np.zeros((nblk, nfreq), np.complex64)
    rec_skip = np.zeros(nblk, np.int8)

    skip = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(nblk):
            blk = x[i * hop:(i + 1) * hop]
            rms = float(np.mean(blk.astype(np.float64) ** 2))
            if rms < 1e-7:
                skip += 1
            else:
                skip = 0
            if skip > 5:
                rec_skip[i] = 1
                rec_lsnr[i] = -15.0
                continue

            # 注意：绑定层 reset 默认 True，流式必须显式传 False
            spec = np.asarray(df.analysis(np.ascontiguousarray(blk[None, :]), False))[0, 0].astype(np.complex64)
            rec_raw[i] = spec

            band_pw = np.add.reduceat(np.abs(spec) ** 2, offs[:-1]) / widths
            band_db = (10.0 * np.log10(band_pw + 1e-10)).astype(np.float32)
            mean_state = band_db * (1 - ALPHA) + mean_state * ALPHA
            feat_erb = ((band_db - mean_state) / 40.0).astype(np.float32)

            lo = spec[:nb_df]
            unit_state = np.abs(lo) * (1 - ALPHA) + unit_state * ALPHA
            feat_cplx = (lo / np.sqrt(unit_state)).astype(np.complex64)

            y_buf.pop(0)
            x_buf.pop(0)
            y_buf.append(spec.copy())
            x_buf.append(spec.copy())

            fe = torch.from_numpy(feat_erb).view(1, 1, 1, nb_erb)
            fs_np = np.stack([feat_cplx.real, feat_cplx.imag], -1).astype(np.float32)
            fs_t = torch.from_numpy(fs_np).view(1, 1, nb_df, 2).permute(0, 3, 1, 2)

            e0, e1, e2, e3, emb, c0, lsnr, h_enc = enc_w(fe, fs_t, h_enc)
            lsnr_v = float(lsnr.reshape(-1)[0])

            m_np = None
            coefs_np = None
            if lsnr_v < -10.0:
                m_np = np.zeros(nb_erb, np.float32)
            elif lsnr_v > 30.0:
                pass
            else:
                m_t, h_erb = erb_w(emb, e3, e2, e1, e0, h_erb)
                m_np = m_t.reshape(-1).numpy().astype(np.float32)
                if lsnr_v <= 20.0:
                    coefs_t, h_df = df_w(emb, c0, h_df)
                    coefs_np = coefs_t[0, :, 0].numpy().astype(np.float32)  # [order, nb_df, 2]

            base = y_buf[order - 1]
            if m_np is None:
                spec_out = base.copy()
            else:
                spec_out = (base * np.repeat(m_np, widths)).astype(np.complex64)
                if coefs_np is not None:
                    acc = np.zeros(nb_df, np.complex64)
                    for k in range(order):
                        ck = (coefs_np[k, :, 0] + 1j * coefs_np[k, :, 1]).astype(np.complex64)
                        acc += ck * x_buf[k][:nb_df]
                    spec_out[:nb_df] = acc

            # synthesis 会原地改写输入谱，先留底再合成
            spec_rec = spec_out.copy()
            o = np.asarray(df.synthesis(np.ascontiguousarray(spec_out[None, None, :]), False))
            out[i * hop:(i + 1) * hop] = o[0]
            rec_feat[i] = feat_erb
            rec_lsnr[i] = lsnr_v
            rec_m[i] = 0.0 if m_np is None else m_np
            rec_coefs[i] = 0.0 if coefs_np is None else coefs_np
            rec_spec[i] = spec_rec

    dt = time.perf_counter() - t0
    dur = nblk * hop / p.sr
    print(f"golden done: {dur:.2f}s audio in {dt:.3f}s | RTF={dt/dur:.3f}")
    print(f"lsnr range [{rec_lsnr.min():.1f}, {rec_lsnr.max():.1f}], skipped {int(rec_skip.sum())} frames")

    np.savez_compressed(
        os.path.join(OUT_DIR, "golden.npz"),
        x=x, out=out, feat=rec_feat, lsnr=rec_lsnr, m=rec_m,
        coefs=rec_coefs, spec=rec_spec, raw=rec_raw, skip=rec_skip,
    )
    wav = os.path.join(OUT_DIR, "golden_out.wav")
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(p.sr)
        wf.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())
    print("saved:", os.path.join(OUT_DIR, "golden.npz"))


if __name__ == "__main__":
    main()
