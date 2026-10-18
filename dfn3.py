# -*- coding: utf-8 -*-
"""DeepFilterNet3 实时推理引擎（纯 numpy + onnxruntime）。

严格复刻 libDF/src/tract.rs 的 DfTract::process 逐帧流程：
  分析(vorbis窗) → ERB dB 特征(指数均值归一) + 复数特征(unit norm) →
  enc → LSNR 分档 → erb_dec 掩码 / df_dec 系数 →
  掩码作用于 y[t-conv_lookahead] → 低频 nb_df 个 bin 用 df_order 阶深度滤波覆盖 →
  合成(OLA)。

模型文件（models/df3/）：
  enc.onnx / erb_dec.onnx / df_dec.onnx / df3.json
由 export_dfn3.py 在 .venv-dfn 环境生成，GRU 隐状态显式入图。
"""
from __future__ import annotations

import json
import os

import numpy as np

try:
    import onnxruntime as ort
    _ORT_OK = True
except Exception:  # pragma: no cover
    ort = None
    _ORT_OK = False


# ---------------------------------------------------------------- 模型定位
def _candidate_dirs():
    dirs = []
    here = os.path.dirname(os.path.abspath(__file__))
    dirs.append(os.path.join(here, "models", "df3"))
    mei = getattr(__import__("sys"), "_MEIPASS", None)
    if mei:
        dirs.append(os.path.join(mei, "models", "df3"))
    exe = os.path.dirname(os.path.abspath(__import__("sys").executable))
    dirs.append(os.path.join(exe, "models", "df3"))
    dirs.append(os.path.join(exe, "_internal", "models", "df3"))
    return dirs


def find_model_dir():
    for d in _candidate_dirs():
        if os.path.isfile(os.path.join(d, "df3.json")):
            return d
    return None


def available():
    """引擎是否可用（onnxruntime + 模型文件都在）。"""
    return _ORT_OK and find_model_dir() is not None


# ---------------------------------------------------------------- DSP 基元
def vorbis_window(fft_size):
    """libDF 的 vorbis 窗：sin(pi/2 * sin^2(pi*n/N))，f64 计算后转 f32。"""
    n = np.arange(fft_size, dtype=np.float64)
    half = fft_size // 2
    s = np.sin(0.5 * np.pi * (n + 0.5) / float(half))
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


class _Stft:
    """libDF DFState 的 numpy 等价：50% 重叠、vorbis 窗、analysis 乘 wnorm。"""

    def __init__(self, fft_size, hop_size):
        assert fft_size == 2 * hop_size, "only 50% overlap supported"
        self.fft_size = fft_size
        self.hop = hop_size
        self.window = vorbis_window(fft_size)
        # rust: wnorm = 1 / (fft^2 / (2*hop))
        self.wnorm = np.float32(1.0 / (fft_size * fft_size / (2.0 * hop_size)))
        # 分析/合成各有独立缓冲（对应 rust 的 analysis_mem / synthesis_mem）
        self.analysis_mem = np.zeros(hop_size, dtype=np.float32)
        self.synthesis_mem = np.zeros(hop_size, dtype=np.float32)

    def reset(self):
        self.analysis_mem = np.zeros(self.hop, dtype=np.float32)
        self.synthesis_mem = np.zeros(self.hop, dtype=np.float32)

    def analysis(self, block):
        buf = np.concatenate((self.analysis_mem, block)) * self.window
        spec = np.fft.rfft(buf).astype(np.complex64) * self.wnorm
        self.analysis_mem = np.array(block, dtype=np.float32, copy=True)
        return spec

    def synthesis(self, spec):
        x = np.fft.irfft(spec.astype(np.complex128), n=self.fft_size)
        x = (x * self.fft_size).astype(np.float32) * self.window
        out = x[: self.hop] + self.synthesis_mem
        self.synthesis_mem = x[self.hop:]
        return out.astype(np.float32)


class Dfn3Engine:
    """逐 480 样本块流式降噪。输入输出均为 float32 单声道 numpy 数组。"""

    # LSNR 分档阈值（RuntimeParams::default_with_ch）
    MIN_DB_THRESH = -10.0
    MAX_DB_ERB_THRESH = 30.0
    MAX_DB_DF_THRESH = 20.0
    ALPHA = np.float32(0.995)

    def __init__(self, model_dir=None, threads=1):
        if not _ORT_OK:
            raise RuntimeError("onnxruntime 未安装")
        model_dir = model_dir or find_model_dir()
        if model_dir is None:
            raise RuntimeError("未找到 DeepFilterNet3 模型（models/df3）")
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "df3.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        m = self.meta
        self.sr = int(m["sr"])
        self.fft = int(m["fft_size"])
        self.hop = int(m["hop_size"])
        self.nb_erb = int(m["nb_erb"])
        self.nb_df = int(m["nb_df"])
        self.order = int(m["df_order"])
        self.conv_la = int(m["conv_lookahead"])
        self.lookahead = max(int(m["conv_lookahead"]), int(m["df_lookahead"]))
        self.emb_hidden = int(m["emb_hidden"])
        self.df_hidden = int(m["df_hidden"])
        self.widths = np.array(m["erb_widths"], dtype=np.int64)
        self.nfreq = self.fft // 2 + 1
        self.offs = np.concatenate(([0], np.cumsum(self.widths)))
        # 每个 fft bin 属于哪个 erb band（掩码逐 bin 相乘用）
        self.band_of_bin = np.repeat(np.arange(self.nb_erb), self.widths)

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        prov = ["CPUExecutionProvider"]
        self._enc = ort.InferenceSession(os.path.join(model_dir, "enc.onnx"), so, prov)
        self._erb = ort.InferenceSession(os.path.join(model_dir, "erb_dec.onnx"), so, prov)
        self._df = ort.InferenceSession(os.path.join(model_dir, "df_dec.onnx"), so, prov)

        self.reset()

    # -- 状态 --------------------------------------------------------
    def reset(self):
        self._stft = _Stft(self.fft, self.hop)
        self._mean_state = np.linspace(-60.0, -90.0, self.nb_erb).astype(np.float32)
        self._unit_state = np.linspace(0.001, 0.0001, self.nb_df).astype(np.float32)
        self._y = [np.zeros(self.nfreq, np.complex64) for _ in range(self.order + self.conv_la)]
        self._x = [np.zeros(self.nfreq, np.complex64) for _ in range(max(self.order, self.lookahead))]
        self._h_enc = np.zeros((1, 1, self.emb_hidden), np.float32)
        self._h_erb = np.zeros((self.meta["emb_layers"] - 1, 1, self.emb_hidden), np.float32)
        self._h_df = np.zeros((self.meta["df_layers"], 1, self.df_hidden), np.float32)
        self._skip = 0
        self.last_lsnr = -15.0
        self.last_mask = 1.0   # 最近一帧的 ERB 掩码均值：噪声帧低、人声帧高
        self.debug = False
        self.rec = None

    def start_debug(self):
        self.debug = True
        self.rec = {"feat": [], "lsnr": [], "m": [], "coefs": [], "spec": [], "raw": []}

    # -- 推理 --------------------------------------------------------
    def process(self, mono):
        """处理 hop_size 个样本，返回同样长度的增强信号。"""
        x = np.asarray(mono, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.hop:
            raise ValueError(f"block must be {self.hop} samples, got {x.shape[0]}")

        rms = float(np.mean(x.astype(np.float64) ** 2))
        if rms < 1e-7:
            self._skip += 1
        else:
            self._skip = 0
        if self._skip > 5:
            self.last_lsnr = -15.0
            if self.debug:
                self.rec["feat"].append(np.zeros(self.nb_erb, np.float32))
                self.rec["lsnr"].append(-15.0)
                self.rec["m"].append(np.zeros(self.nb_erb, np.float32))
                self.rec["coefs"].append(np.zeros((self.order, self.nb_df, 2), np.float32))
                self.rec["spec"].append(np.zeros(self.nfreq, np.complex64))
                self.rec["raw"].append(np.zeros(self.nfreq, np.complex64))
            return np.zeros(self.hop, np.float32)

        spec = self._stft.analysis(x)
        if self.debug:
            self.rec["raw"].append(spec.copy())

        # ---- ERB 特征：(band 平均功率 dB) 减指数均值，再 /40
        band_pw = np.add.reduceat(np.abs(spec) ** 2, self.offs[:-1]) / self.widths
        band_db = (10.0 * np.log10(band_pw + 1e-10)).astype(np.float32)
        self._mean_state = band_db * (1 - self.ALPHA) + self._mean_state * self.ALPHA
        feat_erb = ((band_db - self._mean_state) / 40.0).astype(np.float32)

        # ---- 复数特征：低频 nb_df 个 bin 做 unit norm
        lo = spec[: self.nb_df]
        self._unit_state = np.abs(lo) * (1 - self.ALPHA) + self._unit_state * self.ALPHA
        feat_cplx = (lo / np.sqrt(self._unit_state)).astype(np.complex64)

        self._y.pop(0)
        self._x.pop(0)
        self._y.append(spec.copy())
        self._x.append(spec.copy())

        fe = feat_erb.reshape(1, 1, 1, self.nb_erb)
        fs = np.stack([feat_cplx.real, feat_cplx.imag], -1).reshape(1, 1, self.nb_df, 2)
        fs = fs.transpose(0, 3, 1, 2).astype(np.float32)

        outs = self._enc.run(None, {"feat_erb": fe, "feat_spec": fs, "h_in": self._h_enc})
        e0, e1, e2, e3, emb, c0, lsnr, self._h_enc = outs
        lsnr_v = float(lsnr.reshape(-1)[0])
        self.last_lsnr = lsnr_v

        m_np = None
        coefs = None
        if lsnr_v < self.MIN_DB_THRESH:
            m_np = np.zeros(self.nb_erb, np.float32)
        elif lsnr_v > self.MAX_DB_ERB_THRESH:
            pass
        else:
            m_t, self._h_erb = self._erb.run(
                None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0, "h_in": self._h_erb}
            )
            m_np = m_t.reshape(-1).astype(np.float32)
            if lsnr_v <= self.MAX_DB_DF_THRESH:
                coefs_t, self._h_df = self._df.run(
                    None, {"emb": emb, "c0": c0, "h_in": self._h_df}
                )
                coefs = coefs_t[0, :, 0]  # [order, nb_df, 2]

        self.last_mask = 1.0 if m_np is None else float(np.mean(m_np))
        if self.debug:
            self.rec["feat"].append(feat_erb)
            self.rec["lsnr"].append(lsnr_v)
            self.rec["m"].append(np.zeros(self.nb_erb, np.float32) if m_np is None else m_np)

        base = self._y[self.order - 1]
        if m_np is None:
            spec_out = base.copy()
        else:
            spec_out = (base * m_np[self.band_of_bin]).astype(np.complex64)
            if coefs is not None:
                acc = np.zeros(self.nb_df, np.complex64)
                for k in range(self.order):
                    ck = (coefs[k, :, 0] + 1j * coefs[k, :, 1]).astype(np.complex64)
                    acc += ck * self._x[k][: self.nb_df]
                spec_out[: self.nb_df] = acc

        if self.debug:
            self.rec["coefs"].append(
                np.zeros((self.order, self.nb_df, 2), np.float32) if coefs is None else coefs
            )
            self.rec["spec"].append(spec_out)

        return self._stft.synthesis(spec_out)
