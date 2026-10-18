# -*- coding: utf-8 -*-
"""把 DeepFilterNet3 导出为三个 ONNX 子网（enc / erb_dec / df_dec）。

与官方 export.py 的区别：GRU 隐状态显式作为图输入/输出，
这样 onnxruntime 才能跨帧流式推理（官方脚本导出的图每帧重置状态）。

运行环境：.venv-dfn（torch + deepfilternet + libdf）
"""
import json
import os

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "DeepFilterNet3")
OUT_DIR = os.path.join(HERE, "models", "df3")

from df.enhance import init_df  # noqa: E402
from df.model import ModelParams  # noqa: E402
from libdf import DF  # noqa: E402

torch.manual_seed(0)


class EncWrap(nn.Module):
    """Encoder + emb GRU，GRU 状态显式输入输出。"""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, feat_erb, feat_spec, h0):
        e = self.enc
        e0 = e.erb_conv0(feat_erb)
        e1 = e.erb_conv1(e0)
        e2 = e.erb_conv2(e1)
        e3 = e.erb_conv3(e2)
        c0 = e.df_conv0(feat_spec)
        c1 = e.df_conv1(c0)
        cemb = c1.permute(0, 2, 3, 1).flatten(2)
        cemb = e.df_fc_emb(cemb)
        emb = e3.permute(0, 2, 3, 1).flatten(2)
        emb = e.combine(emb, cemb)
        emb, h = e.emb_gru(emb, h0)
        lsnr = e.lsnr_fc(emb) * e.lsnr_scale + e.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr, h


class ErbDecWrap(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.d = dec

    def forward(self, emb, e3, e2, e1, e0, h0):
        d = self.d
        b, _, t, f8 = e3.shape
        emb, h = d.emb_gru(emb, h0)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)
        e3 = d.convt3(d.conv3p(e3) + emb)
        e2 = d.convt2(d.conv2p(e2) + e3)
        e1 = d.convt1(d.conv1p(e1) + e2)
        m = d.conv0_out(d.conv0p(e0) + e1)
        return m, h


class DfDecWrap(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.d = dec

    def forward(self, emb, c0, h0):
        d = self.d
        b, t, _ = emb.shape
        c, h = d.df_gru(emb, h0)
        if d.df_skip is not None:
            c = c + d.df_skip(emb)
        c0 = d.df_convp(c0).permute(0, 2, 3, 1)
        c = d.df_out(c)
        c = c.view(b, t, d.df_bins, d.df_out_ch) + c0
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        c = c.view(b, t, d.df_bins, d.df_order, 2).permute(0, 3, 1, 2, 4)
        return c, h


def export_one(path, model, inputs, input_names, output_names, opset=14):
    torch.onnx.export(
        model,
        inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[+] exported {os.path.basename(path)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # 本机无 git，logger 里会调 git 取 commit/branch，直接打桩
    import df.logger as _lg
    import df.utils as _u
    for _m in (_u, _lg):
        for _n in ("get_commit_hash", "get_branch_name", "get_git_root"):
            if hasattr(_m, _n):
                setattr(_m, _n, lambda: "local")
    model, df_state, suffix = init_df(
        MODEL_DIR, post_filter=False, log_level="ERROR", log_file=None
    )[:3]
    model = model.cpu().eval()
    p = ModelParams()
    print(f"model={suffix}")
    print(f"sr={p.sr} fft={p.fft_size} hop={p.hop_size} nb_erb={p.nb_erb} nb_df={p.nb_df} "
          f"df_order={p.df_order} conv_la={p.conv_lookahead} df_la={p.df_lookahead}")

    d = DF(p.sr, p.fft_size, p.hop_size, p.nb_erb, 2)
    widths = [int(x) for x in d.erb_widths()]
    win = np.asarray(d.fft_window(), dtype=np.float32)
    print("widths", widths, sum(widths))

    enc = EncWrap(model.enc).eval()
    erb_dec = ErbDecWrap(model.erb_dec).eval()
    df_dec = DfDecWrap(model.df_dec).eval()

    ch = p.conv_ch
    nb_erb, nb_df = p.nb_erb, p.nb_df
    f8 = nb_erb // 4

    with torch.no_grad():
        i_erb = torch.randn(1, 1, 1, nb_erb)
        i_spec = torch.randn(1, 2, 1, nb_df)
        i_he = torch.zeros(1, 1, p.emb_hidden_dim)
        out_enc = enc(i_erb, i_spec, i_he)
        print("enc outs:", [tuple(o.shape) for o in out_enc])
        export_one(
            os.path.join(OUT_DIR, "enc.onnx"), enc,
            (i_erb, i_spec, i_he),
            ["feat_erb", "feat_spec", "h_in"],
            ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr", "h_out"],
        )

        e0, e1, e2, e3, emb, c0, lsnr, _ = out_enc
        i_he2 = torch.zeros(2, 1, p.emb_hidden_dim)
        out_ed = erb_dec(emb, e3, e2, e1, e0, i_he2)
        print("erb_dec outs:", [tuple(o.shape) for o in out_ed])
        export_one(
            os.path.join(OUT_DIR, "erb_dec.onnx"), erb_dec,
            (emb.clone(), e3.clone(), e2.clone(), e1.clone(), e0.clone(), i_he2),
            ["emb", "e3", "e2", "e1", "e0", "h_in"],
            ["m", "h_out"],
        )

        out_dd = df_dec(emb.clone(), c0.clone(), torch.zeros(2, 1, p.df_hidden_dim))
        print("df_dec outs:", [tuple(o.shape) for o in out_dd])
        export_one(
            os.path.join(OUT_DIR, "df_dec.onnx"), df_dec,
            (emb.clone(), c0.clone(), torch.zeros(2, 1, p.df_hidden_dim)),
            ["emb", "c0", "h_in"],
            ["coefs", "h_out"],
        )

    meta = {
        "sr": int(p.sr),
        "fft_size": int(p.fft_size),
        "hop_size": int(p.hop_size),
        "nb_erb": nb_erb,
        "nb_df": nb_df,
        "df_order": int(p.df_order),
        "conv_lookahead": int(p.conv_lookahead),
        "df_lookahead": int(p.df_lookahead),
        "conv_ch": ch,
        "emb_hidden": int(p.emb_hidden_dim),
        "emb_layers": int(p.emb_num_layers),
        "df_hidden": int(p.df_hidden_dim),
        "df_layers": int(p.df_num_layers),
        "erb_widths": widths,
        "norm_alpha": 0.995,
        "lsnr_min": float(p.lsnr_min),
        "lsnr_max": float(p.lsnr_max),
        "min_db_thresh": -10.0,
        "max_db_erb_thresh": 30.0,
        "max_db_df_thresh": 20.0,
        "window": [float(x) for x in win],
    }
    with open(os.path.join(OUT_DIR, "df3.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    np.save(os.path.join(OUT_DIR, "window.npy"), win)
    print("[+] meta saved:", os.path.join(OUT_DIR, "df3.json"))

    # 用 onnxruntime 校验三个图与 torch 输出一致
    try:
        import onnxruntime as ort
    except ImportError:
        print("[!] onnxruntime 未安装，跳过数值校验")
        return
    so = ort.SessionOptions()
    so.log_severity_level = 3
    with torch.no_grad():
        s_enc = ort.InferenceSession(os.path.join(OUT_DIR, "enc.onnx"), so, ["CPUExecutionProvider"])
        got = s_enc.run(None, {
            "feat_erb": i_erb.numpy(), "feat_spec": i_spec.numpy(), "h_in": i_he.numpy(),
        })
        for name, a, b in zip(["e0", "e1", "e2", "e3", "emb", "c0", "lsnr", "h"],
                              out_enc, got):
            print(f"  enc {name}: max|diff|={np.abs(a.numpy()-b).max():.2e}")

        s_ed = ort.InferenceSession(os.path.join(OUT_DIR, "erb_dec.onnx"), so, ["CPUExecutionProvider"])
        got = s_ed.run(None, {
            "emb": emb.numpy(), "e3": e3.numpy(), "e2": e2.numpy(),
            "e1": e1.numpy(), "e0": e0.numpy(), "h_in": i_he2.numpy(),
        })
        for name, a, b in zip(["m", "h"], out_ed, got):
            print(f"  erb_dec {name}: max|diff|={np.abs(a.numpy()-b).max():.2e}")

        s_dd = ort.InferenceSession(os.path.join(OUT_DIR, "df_dec.onnx"), so, ["CPUExecutionProvider"])
        got = s_dd.run(None, {
            "emb": emb.numpy(), "c0": c0.numpy(), "h_in": np.zeros((2, 1, p.df_hidden_dim), np.float32),
        })
        for name, a, b in zip(["coefs", "h"], out_dd, got):
            print(f"  df_dec {name}: max|diff|={np.abs(a.numpy()-b).max():.2e}")


if __name__ == "__main__":
    main()
