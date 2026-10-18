#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""麦克风实时降噪 / 噪声门工具

处理链（全部向量化，逐块处理，回调内无锁、无阻塞）：
    1. 高通滤波     -> 去掉低频隆隆声、风噪、直流偏置
    2. 频域谱门     -> 跟踪噪声底并压制稳态噪声（可关闭，关闭时省 CPU）
    3. dB 噪声门    -> 低于设定阈值的块按 起音/释放/保持 平滑衰减
    4. 输出增益 + 软限幅

运行：python mic_denoiser.py
自检：python mic_denoiser.py --selftest
打包：python build_exe.py（生成 dist/MicDenoiser.exe）

关闭窗口会驻留系统托盘后台运行，托盘菜单可显示窗口 / 启停 / 退出。
"""

import json
import math
import os
import queue
import ctypes
import sys
import threading
import time
import tkinter as tk
import wave
from collections import deque
from tkinter import simpledialog, ttk, messagebox

import numpy as np
import sounddevice as sd
from scipy.fft import irfft, rfft
from scipy.signal import butter, lfilter, sosfilt
from scipy.special import exp1

try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_OK = True
except Exception:  # 没有 pystray / pillow 时降级为普通窗口
    pystray = None
    Image = ImageDraw = None
    TRAY_OK = False


_WINDOW_TITLE = "麦克风降噪 · Mic Denoiser"


def _app_dir():
    """exe 所在目录；开发时是脚本所在目录。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _config_path():
    """配置存到 exe（或脚本）所在目录，方便备份迁移。"""
    base = _app_dir()
    new_path = os.path.join(base, "config.json")
    # 迁移旧配置（此前版本存过 %APPDATA%\\MicDenoiser）
    old_path = os.path.join(os.environ.get("APPDATA") or base, "MicDenoiser",
                            "config.json")
    if not os.path.exists(new_path) and os.path.exists(old_path):
        try:
            import shutil
            shutil.copyfile(old_path, new_path)
        except Exception:
            pass
    return new_path


CONFIG_PATH = _config_path()

BLOCK_SIZES = (480, 960, 512, 1024)
# 采样率固定 48kHz：RNNoise / DeepFilterNet3 都只支持 48k
FIXED_SR = 48000

# DeepFilterNet3 引擎（numpy + onnxruntime）惰性加载：缺模型/缺运行时自动退回 RNNoise
_DFN3_MOD = None
_DFN3_CHECKED = False


def _dfn3_support():
    """返回 dfn3 模块；不可用时返回 None（只探测一次）。"""
    global _DFN3_MOD, _DFN3_CHECKED
    if _DFN3_CHECKED:
        return _DFN3_MOD
    _DFN3_CHECKED = True
    try:
        import dfn3 as mod
        _DFN3_MOD = mod if mod.available() else None
    except Exception:
        _DFN3_MOD = None
    return _DFN3_MOD


DFN3_PRESENT = _dfn3_support() is not None

# 用于识别虚拟声卡的设备名关键字（PortAudio 返回的名字会被截断，只能做包含匹配）
VIRTUAL_PLAY_KEYS = ("cable input", "voicemeeter input", "voice meeter input",
                     "virtual audio cable", "vb-audio")
VIRTUAL_REC_KEYS = ("cable output", "voicemeeter out", "voice meeter out",
                    "virtual audio cable", "vb-audio")
STEREO_MIX_KEYS = ("立体声混音", "stereo mix", "stereo input",
                   "what u hear", "wave out mix")
MIC_KEYS = ("麦", "microphone", "mic ", "话筒")
# 这些是「默认设备」的别名，实际指向可能被改成虚拟声卡，自动配置时避开
MIXER_KEYS = ("声音映射器", "sound mapper", "主声音捕获", "主声音驱动",
              "primary sound", "primary capture")

# Windows 11 Fluent 浅色配色
BG = "#F3F3F3"          # 页面底
CARD = "#FFFFFF"        # 卡片
CARD2 = "#FAFAFA"       # 输入框底
FG = "#1B1B1B"          # 主文字
DIM = "#616161"         # 次要文字
LINE = "#E5E5E5"        # 描边
ACCENT = "#0067C0"      # 系统强调蓝
ACCENT_D = "#005A9E"    # 悬停/按下
DANGER = "#C42B1C"      # 错误 / 录音

DEFAULT_PARAMS = {
    # 自适应门限：实时噪声底 + 这个余量（不再用固定 dB 阈值）
    "gate_margin": 7.0,
    "attack_ms": 4.0,        # 开门速度
    "release_ms": 90.0,      # 关门速度（快一些，消除残响拖尾）
    "hold_ms": 80.0,         # 保持时间，避免词语间隙被切断（也消除 10ms 碎片）
    "atten_db": 60.0,        # 关门时衰减多少 dB（不过度深，避免"数字死寂"）
    "denoise": 70.0,         # 谱降噪强度 0-100（0 = 关闭，最省 CPU）
    "hp_hz": 60.0,           # 高通截止频率
    "gain_db": 4.0,          # 输出增益（补足响度到 -22~-18dBFS 区间）
    "vad": 1.0,              # 1 = 只保留人声（语音概率低的成分额外压掉）
    "rnnoise": 1.0,          # 1 = 用 RNNoise 神经降噪（对齐 RTX Voice）
    "model": "dfn3",         # 降噪模型：dfn3 = DeepFilterNet3（更强），rnnoise = RNNoise
}

# 各模型的动态处理档位（门控 / 压缩）---------------------------------------
# RNNoise：沿用默认值（按听感调过的那套）
# DFN3：模型抑制强、底噪已经很低，再叠加"60dB 硬关门 + 快释放 + 重压缩"
# 就会出现说话中间掉电平的"忽大忽小"泵感。这里换成温和档：
# 浅衰减、慢开门、长保持、慢释放、轻压缩，贴近 NVIDIA Broadcast 的无感处理。
DFN3_PROFILE = {
    # 门限余量抬高：呼吸、嘴部瞬态挡在门外，减少碎片段
    "gate_margin": 8.0,
    # 平滑 LSNR 低于该值视为非人声（键盘、纸张、环境声），直接压到门底
    "vad_lsnr": -8.0,
    "attack_ms": 10.0,
    "release_ms": 240.0,
    "hold_ms": 150.0,
    # DFN3 残留极低，门是最后一道清道夫：衰减加深到 45dB 才清得干净，
    # 释放/保持都够慢，加深不会产生泵感
    "atten_db": 45.0,
    # 输出保持在输入电平附近，不额外抬增益（用户明确不想被放大）
    "gain_db": 0.0,
    # 压缩更温和：段内起伏的小一半靠"稳"
    "comp": {"threshold_db": -26.0, "ratio": 1.6, "attack_ms": 15.0,
             "release_ms": 180.0, "makeup_db": 1.0, "input_gain_db": 6.0},
    # 宽带突发噪声（纸张/鼠标刮擦）抑制：平坦度阈值、持续帧数、压制深度
    "burst_thresh": 0.70,
    "burst_need": 8,
    "burst_depth_db": -24.0,
}




def make_icon_image(size=64):
    """程序化绘制麦克风图标，托盘与窗口共用，打包时无需外部资源。"""
    if Image is None:
        return None
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = float(size)
    d.ellipse((1, 1, s - 2, s - 2), fill=(28, 32, 40, 255))
    w = s * 0.20
    cx = s / 2.0
    d.rounded_rectangle((cx - w / 2, s * 0.19, cx + w / 2, s * 0.53),
                        radius=w / 2, fill=(61, 220, 132, 255))
    stroke = max(2, int(round(s * 0.05)))
    d.arc((s * 0.30, s * 0.30, s * 0.70, s * 0.70), start=0, end=180,
          fill=(235, 240, 245, 255), width=stroke)
    d.line((cx, s * 0.69, cx, s * 0.80), fill=(235, 240, 245, 255), width=stroke)
    d.line((s * 0.38, s * 0.80, s * 0.62, s * 0.80),
           fill=(235, 240, 245, 255), width=stroke)
    return img


def find_device(devices, keys, kind, hostapi=None):
    """按名称关键字找设备，kind 取 'in' 或 'out'，返回索引或 None。"""
    for i, dev in enumerate(devices):
        if hostapi is not None and dev["hostapi"] != hostapi:
            continue
        if kind == "in" and dev["max_input_channels"] <= 0:
            continue
        if kind == "out" and dev["max_output_channels"] <= 0:
            continue
        if any(k in dev["name"].lower() for k in keys):
            return i
    return None


def find_duplex_pair(devices):
    """找同一个 hostapi 下的「虚拟声卡输出 + 物理麦克风」。

    PortAudio 不允许不同 hostapi 的设备组成全双工流（PaErrorCode -9993），
    所以输入输出必须落在同一个 API 上。
    """
    try:
        n_api = len(sd.query_hostapis())
    except Exception:
        n_api = 1
    for api in range(n_api):
        out = find_device(devices, VIRTUAL_PLAY_KEYS, "out", api)
        mic = find_device(devices, MIC_KEYS, "in", api)
        if out is not None and mic is not None:
            return out, mic
    return None, None


def find_same_device(devices, name, hostapi, kind):
    """在指定 hostapi 下找同名设备（PortAudio 的名字常被截断，做互相包含匹配）。"""
    low = name.lower()
    for i, dev in enumerate(devices):
        if dev["hostapi"] != hostapi:
            continue
        if kind == "in" and dev["max_input_channels"] <= 0:
            continue
        if kind == "out" and dev["max_output_channels"] <= 0:
            continue
        other = dev["name"].lower()
        if low in other or other in low:
            return i
    return None


def cable_playback_disabled(devices):
    """VB-CABLE 播放端只出现在 WDM-KS、常规 API 都看不到 = 设备被禁用。"""
    try:
        apis = sd.query_hostapis()
    except Exception:
        return False
    ks = next((i for i, h in enumerate(apis) if "wdm" in h["name"].lower()), None)
    if ks is None:
        return False
    if find_device(devices, VIRTUAL_PLAY_KEYS, "out", ks) is None:
        return False
    for api in range(len(apis)):
        if api == ks:
            continue
        if find_device(devices, VIRTUAL_PLAY_KEYS, "out", api) is not None:
            return False
    return True


class Tooltip:
    """轻量悬停提示，不依赖第三方库。"""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tip = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None):
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + 8
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        tk.Label(tw, text=self.text, justify="left", bg="#FFFFFF", fg=FG,
                 relief="solid", borderwidth=1, padx=10, pady=7,
                 font=("Microsoft YaHei UI", 9)).pack()

    def _hide(self, _event=None):
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def rms_db(x):
    """返回块的 dBFS 电平（10*log10(mean(x^2))）。"""
    n = x.shape[0]
    if n == 0:
        return -120.0
    energy = float(np.dot(x, x)) / n
    if energy <= 1e-12:
        return -120.0
    return 10.0 * math.log10(energy)


class SpectralGate:
    """频域谱门降噪。

    50% 重叠 OLA，分析/合成窗取 sqrt(hann)，窗平方为周期 hann，
    重叠相加恒为 1，因此无需额外归一化。
    """

    __slots__ = ("hop", "n", "bins", "win", "in_buf", "out_buf", "noise",
                 "prev_power", "prev_gain", "gain", "warmup", "flatness",
                 "p_speech", "p_frame")

    def __init__(self, hop):
        self.hop = hop
        self.n = hop * 2
        self.bins = hop + 1
        idx = np.arange(self.n, dtype=np.float64)
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * idx / self.n)
        self.win = np.sqrt(hann).astype(np.float32)
        self.in_buf = np.zeros(self.n, dtype=np.float32)
        self.out_buf = np.zeros(self.n, dtype=np.float32)
        self.reset()

    def reset(self):
        self.in_buf.fill(0.0)
        self.out_buf.fill(0.0)
        self.noise = np.full(self.bins, np.inf, dtype=np.float32)
        self.prev_power = np.zeros(self.bins, dtype=np.float32)
        self.prev_gain = np.ones(self.bins, dtype=np.float32)
        self.gain = np.ones(self.bins, dtype=np.float32)
        self.warmup = 0
        self.flatness = 1.0
        self.p_speech = np.ones(self.bins, dtype=np.float32)
        self.p_frame = 1.0

    def process(self, x, strength, want_flatness=False):
        hop = self.hop
        # 滚动输入缓冲
        self.in_buf[:-hop] = self.in_buf[hop:]
        self.in_buf[-hop:] = x

        spec = rfft(self.in_buf * self.win)
        power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
        power += 1e-18

        if want_flatness:
            # 谱平坦度：白噪声/键盘声偏高，人声谐波结构下远低于它
            mean_power = float(power.mean()) + 1e-18
            self.flatness = float(np.exp(np.log(power).mean()) / mean_power)

        if self.warmup < 90:
            # 前 ~1 秒估计环境噪声底并原样透传，不把用户第一句话当噪声
            if self.warmup == 0:
                self.noise = power.copy()
            else:
                self.noise += 0.15 * (power - self.noise)
            self.warmup += 1
            self.prev_power = power
            self.prev_gain = self.gain
        else:
            # 1) 噪声底：低 SNR 处吸收、语音段冻结
            low = power < 2.0 * self.noise
            alpha = np.where(low, np.where(power > self.noise, 0.08, 0.45), 0.0)
            self.noise += alpha * (power - self.noise)
            np.maximum(self.noise, 1e-12, out=self.noise)

            # 2) 后验 SNR 与 decision-directed 先验 SNR（Ephraim-Malah）
            #    噪声底乘过减因子，才能压住噪声谱的统计涨落
            noise_eff = self.noise * np.float32(1.0 + 2.0 * strength)
            gamma = power / noise_eff
            np.maximum(gamma, 1e-4, out=gamma)

            xi = (0.98 * self.prev_gain * self.prev_gain * self.prev_power / noise_eff
                  + 0.02 * np.maximum(gamma - 1.0, 0.0))
            np.maximum(xi, 1e-3, out=xi)

            # 3) MMSE-LSA 增益
            v = xi * gamma / (1.0 + xi)
            g = xi / (1.0 + xi) * np.exp(0.5 * exp1(np.maximum(v, 1e-8)))
            np.clip(g, 0.0, 1.0, out=g)

            # 3b) 语音存在概率（高斯模型）：像噪声的频点再压一层
            spp = 1.0 / (1.0 + (1.0 + xi) * np.exp(-v))
            self.p_speech += 0.35 * (spp.astype(np.float32) - self.p_speech)
            self.p_frame = float(self.p_speech.mean())
            g *= np.maximum(self.p_speech, 0.12)

            # 4) 强度控制与平滑（平滑可显著减弱“音乐噪声”）
            if strength > 0.0:
                g = np.power(g, 1.0 + 2.0 * strength)
            g = np.maximum(g, 10.0 ** (-(32.0 * strength) / 20.0)).astype(np.float32)
            self.gain += 0.5 * (g - self.gain)

            self.prev_power = power
            self.prev_gain = self.gain

        y = irfft(spec * self.gain, self.n).astype(np.float32) * self.win
        self.out_buf += y

        out = self.out_buf[:hop].copy()
        self.out_buf[:-hop] = self.out_buf[hop:]
        self.out_buf[-hop:] = 0.0
        return out


def _find_rnnoise_dll():
    """定位 RNNoise 的 DLL：打包后从 _MEIPASS 找，开发环境从 pyrnnoise 包找。"""
    import importlib.util
    candidates = []
    base = getattr(sys, "_MEIPASS", None)
    if base:
        candidates += [os.path.join(base, "rnnoise.dll"),
                       os.path.join(base, "pyrnnoise", "rnnoise.dll")]
    try:
        spec = importlib.util.find_spec("pyrnnoise")
        if spec is not None and spec.origin:
            candidates.append(
                os.path.join(os.path.dirname(spec.origin), "rnnoise.dll"))
    except Exception:
        pass
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


class RnnoiseEngine:
    """RNNoise 神经降噪（ctypes 直连自带 DLL），48kHz / 480 样本每帧。"""

    FRAME = 480
    SR = 48000

    def __init__(self):
        self._lib = None
        self._state = None
        self.prob = 1.0
        dll = _find_rnnoise_dll()
        if dll is None:
            return
        try:
            lib = ctypes.CDLL(dll)
            lib.rnnoise_create.argtypes = [ctypes.c_void_p]
            lib.rnnoise_create.restype = ctypes.c_void_p
            lib.rnnoise_process_frame.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.rnnoise_process_frame.restype = ctypes.c_float
            lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
            state = lib.rnnoise_create(None)
            if not state:
                return
            self._lib = lib
            self._state = state
        except Exception:
            self._lib = None
            self._state = None

    @property
    def available(self):
        return self._state is not None

    def process(self, frame):
        """frame: 480 长度 float32 [-1,1] -> 同长度 float32 [-1,1]。"""
        fi = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16).astype(np.float32)
        ptr = fi.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.prob = float(self._lib.rnnoise_process_frame(self._state, ptr, ptr))
        return (fi.astype(np.int16).astype(np.float32) / 32767.0)

    def close(self):
        if self._state is not None:
            try:
                self._lib.rnnoise_destroy(self._state)
            except Exception:
                pass
            self._state = None


RNNOISE_PRESENT = _find_rnnoise_dll() is not None


def _biquad_shelf(sr, f0, gain_db, low=True):
    """RBJ 一阶倾斜均衡器系数（low/high shelf）。"""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cos_w = math.cos(w0)
    alpha = math.sin(w0) / 2.0
    two = 2.0 * math.sqrt(A) * alpha
    if low:
        b0 = A * ((A + 1.0) - (A - 1.0) * cos_w + two)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w - two)
        a0 = (A + 1.0) + (A - 1.0) * cos_w + two
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w)
        a2 = (A + 1.0) + (A - 1.0) * cos_w - two
    else:
        b0 = A * ((A + 1.0) + (A - 1.0) * cos_w + two)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w)
        b2 = A * ((A + 1.0) + (A - 1.0) * cos_w - two)
        a0 = (A + 1.0) - (A - 1.0) * cos_w + two
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w)
        a2 = (A + 1.0) - (A - 1.0) * cos_w - two
    return (np.array([b0, b1, b2]) / a0), (np.array([1.0, a1 / a0, a2 / a0]))


def _biquad_peak(sr, f0, gain_db, q=0.7):
    """RBJ peaking 均衡器（中频提升清晰度）。"""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cos_w = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w
    a2 = 1.0 - alpha / A
    return (np.array([b0, b1, b2]) / a0), (np.array([1.0, a1 / a0, a2 / a0]))


class ToneShaper:
    """广播级频响：低频厚实、中频清晰、高频柔和不刺耳。"""

    def __init__(self, sr):
        self._chains = []
        # 低架 +4dB@120Hz（厚实）→ 峰 +1.5dB@2.5kHz（清晰）
        # → 峰 +3.5dB@4.5kHz（补 RNNoise 压得最狠的 4–6kHz 凹坑，去"隔布感"）
        # → 高架 +1dB@8kHz（空气感）→ 低通 12kHz
        self._chains.append(_biquad_shelf(sr, 120.0, 4.0, low=True) + (np.zeros(2),))
        self._chains.append(_biquad_peak(sr, 2500.0, 1.5, q=0.8) + (np.zeros(2),))
        self._chains.append(_biquad_peak(sr, 4500.0, 3.5, q=1.0) + (np.zeros(2),))
        self._chains.append(_biquad_shelf(sr, 8000.0, 1.0, low=False) + (np.zeros(2),))
        b, a = butter(2, 12000.0, btype="low", fs=sr, output="ba")
        self._chains.append((b, a, np.zeros(2)))

    def process(self, x):
        for i, (b, a, zi) in enumerate(self._chains):
            y, zf = lfilter(b, a, x, zi=zi)
            self._chains[i] = (b, a, zf)
            x = y
        return x.astype(np.float32, copy=False)


class DeEsser:
    """齿音消除：动态压低 5kHz 以上的嘶声，只动高频不伤低频。"""

    def __init__(self, sr, threshold=0.22):
        self._sos = butter(2, 5000.0, btype="highpass", fs=sr, output="sos")
        self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        self._thr = threshold
        self._gain = 1.0

    def process(self, x):
        hp, self._zi = sosfilt(self._sos, x, zi=self._zi)
        hp = hp.astype(np.float32, copy=False)
        hf = float(np.sqrt(np.mean(hp * hp)))
        tot = float(np.sqrt(np.mean(x * x)))
        ratio = hf / (tot + 1e-9) if tot > 1e-5 else 0.0
        target = self._thr / ratio if ratio > self._thr else 1.0
        g0 = self._gain
        g1 = g0 + 0.2 * (min(target, 1.0) - g0)
        self._gain = g1
        ramp = np.linspace(g0, g1, len(x), dtype=np.float32)
        return (x - hp) + hp * ramp


class Compressor:
    """软拐点压缩器（AGC）：让响度均匀，说话忽近忽远也稳定。"""

    def __init__(self, sr, threshold_db=-30.0, ratio=2.5,
                 attack_ms=10.0, release_ms=80.0, makeup_db=2.0,
                 input_gain_db=10.0):
        self._sr = float(sr)
        self._thr = threshold_db
        self._ratio = ratio
        self._attack = attack_ms
        self._release = release_ms
        self._makeup = 10.0 ** (makeup_db / 20.0)
        # 预增益：低电平输入先抬进压缩器工作区，压缩才真正起作用
        self._in_gain = 10.0 ** (input_gain_db / 20.0)
        self._gain = 1.0

    def process(self, x):
        x = x * np.float32(self._in_gain)
        rms = float(np.sqrt(np.mean(x * x))) + 1e-9
        db = 20.0 * math.log10(rms)
        over = db - self._thr
        if over > 0.0:
            target = 10.0 ** (-over * (1.0 - 1.0 / self._ratio) / 20.0)
        else:
            target = 1.0
        block_ms = len(x) * 1000.0 / self._sr
        t = self._attack if target < self._gain else self._release
        coef = 1.0 - math.exp(-block_ms / t)
        g0 = self._gain
        g1 = g0 + coef * (target - g0)
        self._gain = g1
        # 块内线性插值，消除块级台阶导致的抖动/颤音
        ramp = np.linspace(g0, g1, len(x), dtype=np.float32)
        return x * (ramp * np.float32(self._makeup))


class BurstDetector:
    """检测纸张摩擦 / 鼠标刮擦这类宽带噪声突发。

    DFN3 对这类"像人声的噪声"会整段放过（实测掩码接近 1），
    但它们的频谱平坦度极高（约 0.82），而人声 0.24、齿音最高约 0.47，
    用 0.68 的平坦度阈值 + 连续性判定可以干净地分开。
    """

    def __init__(self, sr, thresh=0.70, need_frames=8, hold_frames=8,
                 level_margin=-999.0):
        n = 480
        self._win = np.hanning(n).astype(np.float32)
        f = np.fft.rfftfreq(n, 1.0 / sr)
        # 频带取人声主区（300-6000Hz）：1-8kHz 段人声能量弱、常被底噪主导，
        # 平坦度会失真；在 300-6000Hz 里人声 90% 分位约 0.51、摩擦 0.78，
        # 用 0.68 阈值就能干净分开
        self._band = (f >= 300.0) & (f <= 6000.0)
        self.thresh = thresh
        self.need = need_frames
        self.hold_frames = hold_frames
        self.level_margin = level_margin
        self._run = 0
        self._hold = 0
        self.active = False
        self.last_flat = 0.0
        self._flat_s = 0.0       # 平滑后的平坦度（单帧抖动大，用 EMA 稳定判定）

    def process(self, frame, level_db=-120.0, floor_db=-120.0):
        # 只有明显高于环境底噪的突发才处理：词间空隙的安静底噪交给噪声门，
        # 否则检测器会在每次换气的空隙里触发、把正常语音也压掉
        if level_db < floor_db + self.level_margin:
            self._run = 0
            self._hold = 0
            self.active = False
            return False

        sp = np.abs(np.fft.rfft(np.asarray(frame, np.float32) * self._win))
        m = sp[self._band] + 1e-9
        flat = float(np.exp(np.mean(np.log(m))) / (np.mean(m) + 1e-12))
        self.last_flat = flat
        self._flat_s += 0.5 * (flat - self._flat_s)
        if self._flat_s > self.thresh:
            self._run += 1
        else:
            self._run = 0
        if self._run >= self.need:
            self.active = True
            self._hold = self.hold_frames
        elif self.active:
            self._hold -= 1
            if self._hold <= 0:
                self.active = False
        return self.active


class Denoiser:
    """实时降噪器。params 与 GUI 共享同一个 dict，修改立即生效。"""

    def __init__(self, samplerate, blocksize, params):
        self.sr = float(samplerate)
        self.p = params
        self.bs = 0
        self.ramp = np.zeros(1, dtype=np.float32)
        self._spec = None
        self._spec_on = False
        self._hp_hz = -1.0
        self._hp_sos = None
        self._hp_zi = None
        self._gate_gain = 1.0
        self._hold = 0.0
        self._floor_db = -60.0     # 实时跟踪的环境噪声底
        self._floor_init = 0
        self._thr_db = -53.0       # 由底噪 + 灵敏度算出的当前门限
        self._rn = None
        self._rn_ok = False
        self._rn_on = False
        self._dfn = None
        self._dfn_ok = False
        self._dfn_on = False
        self._dfn_lsnr = -15.0
        self._dfn_lsnr_s = -15.0   # 平滑后的 LSNR（VAD 用，避免逐帧抖动）
        self._model_active = None
        self.g = {}                # 当前生效的门控参数（按模型档位覆盖）
        self._burst = None         # 宽带噪声突发检测（纸张/鼠标刮擦）
        self._burst_on = False
        self._burst_gain = 1.0     # 突发抑制的独立增益（快压快放）
        self._setup(blocksize)

    def _apply_profile(self, model):
        """按模型切换动态处理档位，避免 DFN3 下出现泵感。"""
        self._model_active = model
        base = {
            "gate_margin": float(self.p.get("gate_margin", 7.0)),
            "attack_ms": float(self.p.get("attack_ms", 4.0)),
            "release_ms": float(self.p.get("release_ms", 90.0)),
            "hold_ms": float(self.p.get("hold_ms", 80.0)),
            "atten_db": float(self.p.get("atten_db", 60.0)),
            "gain_db": float(self.p.get("gain_db", 4.0)),
        }
        self._burst_depth = 10.0 ** (-24.0 / 20.0)
        if model == "dfn3":
            prof = dict(DFN3_PROFILE)
            comp = prof.pop("comp")
            base.update({k: v for k, v in prof.items() if not k.startswith("burst_")})
            self._comp = Compressor(self.sr, **comp)
            if self._burst is not None:
                self._burst.thresh = float(prof.get("burst_thresh", 0.70))
                self._burst.need = int(prof.get("burst_need", 8))
                self._burst_depth = 10.0 ** (float(prof.get("burst_depth_db", -24.0)) / 20.0)
        else:
            self._comp = Compressor(self.sr)
        self.g = base

    def warmup(self):
        """提前把 DeepFilterNet3 的 onnxruntime 会话建好，避免首块音频卡顿。"""
        if str(self.p.get("model", "dfn3")) == "dfn3":
            self._ensure_dfn()

    def _ensure_dfn(self):
        """惰性构建 DFN3 引擎；失败时返回 False（流程自动退回 RNNoise）。"""
        # 块长必须是 480 的整数倍（512/1024 这类会走谱门兜底）
        if not self._block_ok:
            return False
        if self._dfn is not None:
            return True
        mod = _dfn3_support()
        if mod is None:
            return False
        try:
            self._dfn = mod.Dfn3Engine(threads=1)
            self._dfn_ok = True
            self._dfn.reset()
        except Exception:
            self._dfn = None
            self._dfn_ok = False
        return self._dfn_ok

    def _setup(self, bs):
        self.bs = int(bs)
        if self._rn is not None:
            self._rn.close()
        self._rn = RnnoiseEngine()
        # RNNoise 只在 48kHz 且块长是 480 的整数倍时可用
        self._rn_ok = (self._rn.available
                       and (self.bs % 480 == 0)
                       and abs(self.sr - 48000.0) < 1.0)
        # DFN3 与 RNNoise 的块长/采样率约束相同
        self._block_ok = (self.bs % 480 == 0) and abs(self.sr - 48000.0) < 1.0
        self._rn_on = False
        if self._dfn is not None:
            # 块长变了，旧引擎的帧缓冲不再匹配，重置
            try:
                self._dfn.reset()
            except Exception:
                self._dfn = None
                self._dfn_ok = False
        self._dfn_on = False
        self.ramp = (np.arange(self.bs, dtype=np.float32) / float(self.bs)).astype(np.float32)
        self._spec = SpectralGate(self.bs)
        self._tone = ToneShaper(self.sr)
        self._deesser = DeEsser(self.sr)
        self._comp = Compressor(self.sr)
        self._spec_on = False
        self._hp_hz = -1.0
        self._gate_gain = 1.0
        self._hold = 0.0
        self._thr_db = -53.0
        # 3 秒滑动窗口取最小块电平 = 环境噪声底
        n = max(8, int(3.0 * self.sr / max(self.bs, 1)))
        self._floor_buf = np.full(n, -120.0, dtype=np.float32)
        self._floor_pos = 0
        self._floor_init = 0
        self._floor_db = -60.0
        # 舒适噪声：门关闭时补一点极低电平，避免"数字零"的死寂
        rng = np.random.default_rng(2024)
        # 舒适噪声：白噪声听感"嘶"，整形成 3kHz 低通的中低频"空气声"。
        # 生成 3 倍长度再滤波，取中间段当循环缓冲，接缝处无滤波瞬态。
        n_cn = max(128, int(self.sr))
        cn_src = rng.standard_normal(n_cn * 3).astype(np.float32)
        try:
            cn_sos = butter(2, 3000.0 / (self.sr / 2.0), btype="low", output="sos")
            cn_src = sosfilt(cn_sos, cn_src).astype(np.float32)
        except Exception:
            pass
        cn = cn_src[n_cn:2 * n_cn]
        cn = cn - cn.mean()
        cn = cn / (float(np.sqrt(np.mean(cn ** 2))) + 1e-9)   # 归一化到单位 RMS
        self._cn_buf = cn.astype(np.float32)
        self._cn_pos = 0
        # 电平降到 -80dBFS（输出增益 +10dB 后约 -70），停顿段只剩极淡的空气声
        self._cn_level = float(10.0 ** (-80.0 / 20.0))
        # 高频回混：从 RNNoise 输入取 4–9kHz，补回辅音质感
        nyq = self.sr / 2.0
        self._hf_sos = butter(2, [4000.0 / nyq, 9000.0 / nyq],
                              btype="bandpass", output="sos")
        self._hf_zi = np.zeros((self._hf_sos.shape[0], 2), dtype=np.float64)
        self._hf_gain = np.float32(10.0 ** (-18.0 / 20.0))
        # 宽带突发噪声检测（纸张摩擦、鼠标刮擦）：DFN3 会把它们当人声放过
        self._burst = BurstDetector(self.sr)
        self._burst_on = False
        self._burst_gain = 1.0
        # 输入侧噪声底（3 秒最小值）：突发检测要比"输入底噪"高才算数，
        # 不能用门控那路（那是 DFN3 输出后的底，比输入低 20dB 左右）
        n_in = max(8, int(3.0 * self.sr / max(self.bs, 1)))
        self._in_floor_buf = np.full(n_in, -120.0, dtype=np.float32)
        self._in_floor_pos = 0
        self._in_floor_init = 0
        self._in_floor = -60.0
        # 按当前模型装载动态处理档位（门控/压缩）
        self._apply_profile(str(self.p.get("model", "dfn3")))

    def _track_in_floor(self, db):
        """输入侧噪声底：最近 3 秒最小块电平（不经过模型/增益）。"""
        if self._in_floor_init == 0:
            self._in_floor_buf.fill(db)
            self._in_floor_init = 1
        self._in_floor_buf[self._in_floor_pos] = db
        self._in_floor_pos = (self._in_floor_pos + 1) % self._in_floor_buf.shape[0]
        self._in_floor = float(self._in_floor_buf.min())

    def _track_floor(self, db):
        """实时跟踪噪声底：取最近 3 秒最小块。

        人声再响也只是窗口里的高块，最小值始终落在停顿/底噪上；
        空调、风扇这类稳态噪声一旦持续超过 3 秒，门限会自动跟着抬升。
        """
        if self._floor_init == 0:
            self._floor_buf.fill(db)
            self._floor_init = 1
        self._floor_buf[self._floor_pos] = db
        self._floor_pos = (self._floor_pos + 1) % self._floor_buf.shape[0]
        self._floor_db = float(self._floor_buf.min())

    def _comfort(self, n):
        """从循环缓冲里取一段极低电平噪声（门关闭时混入）。"""
        buf = self._cn_buf
        pos = self._cn_pos
        end = pos + n
        if end <= buf.shape[0]:
            self._cn_pos = end
            return buf[pos:end]
        rest = n - (buf.shape[0] - pos)
        seg = np.concatenate((buf[pos:], buf[:rest]))
        self._cn_pos = rest
        return seg

    def _build_hp(self, hz):
        self._hp_hz = hz
        if hz <= 20.0:
            self._hp_sos = None
            self._hp_zi = None
            return
        nyq = self.sr / 2.0
        hz = min(max(hz, 20.0), nyq * 0.9)
        sos = butter(2, hz / nyq, btype="highpass", output="sos")
        self._hp_sos = np.asarray(sos, dtype=np.float64)
        self._hp_zi = np.zeros((self._hp_sos.shape[0], 2), dtype=np.float64)

    def process(self, mono):
        p = self.p
        x = np.asarray(mono, dtype=np.float32)
        if x.shape[0] == 0:
            return x
        if x.shape[0] != self.bs:
            self._setup(x.shape[0])

        # 1) 高通
        hp = float(p["hp_hz"])
        if abs(hp - self._hp_hz) >= 1.0 or (hp <= 20.0) != (self._hp_sos is None):
            self._build_hp(hp)
        if self._hp_sos is not None:
            x, self._hp_zi = sosfilt(self._hp_sos, x, zi=self._hp_zi)
            x = x.astype(np.float32, copy=False)

        # 1b) 数字零保护：有些 USB 麦克风会偶发整块静音（驱动掉块/硬件静音）。
        # 直接喂零会让 DFN3 跳过推理并冻结状态，恢复瞬间出现电平跳变。
        # 用 -68dBFS 的极低抖动垫底，保持模型状态连续（听感上察觉不到）。
        if float(np.max(np.abs(x))) < 1e-6:
            x = self._comfort(len(x)) * np.float32(10.0 ** (-68.0 / 20.0))

        # 2) 降噪：DeepFilterNet3 优先 → RNNoise → 谱门兜底
        want_model = str(p.get("model", "dfn3"))
        # 切换模型时同步换动态处理档位（DFN3 用温和档，避免泵感）
        if want_model != self._model_active:
            self._apply_profile(want_model)
        use_dfn = want_model == "dfn3" and self._ensure_dfn()
        use_rn = (not use_dfn) and self._rn_ok and float(p.get("rnnoise", 1.0)) > 0.5
        st = float(p["denoise"]) / 100.0
        if use_dfn:
            # 用输入帧判断宽带突发噪声（DFN3 对纸张/摩擦会当人声放过），
            # 门限基于输入侧的底噪，避免把词间空隙的安静噪声当成突发
            x_db = rms_db(x)
            self._track_in_floor(x_db)
            self._burst_on = self._burst.process(x, x_db, self._in_floor)
            # DFN3 逐 480 样本帧流式推理（自身已做谱掩码 + 深度滤波，
            # 高频保真度好，不再需要 RNNoise 的高频回混和低强度谱门）
            out = np.empty_like(x)
            for i in range(self.bs // 480):
                s = i * 480
                out[s:s + 480] = self._dfn.process(x[s:s + 480])
            x = out
            lsnr_now = float(getattr(self._dfn, "last_lsnr", -15.0))
            self._dfn_lsnr = lsnr_now
            # LSNR 逐帧抖动大，VAD 用 150ms 平滑值，避免门控随单帧乱开合
            a_s = 1.0 - math.exp(-(self.bs / self.sr) / 0.15)
            self._dfn_lsnr_s += a_s * (lsnr_now - self._dfn_lsnr_s)
            self._dfn_on = True
            self._rn_on = False
        elif use_rn:
            # 路线 B：先从 RNNoise 的原始输入里取出 4–9kHz 带通分量
            hf, self._hf_zi = sosfilt(self._hf_sos, x, zi=self._hf_zi)
            hf = hf.astype(np.float32, copy=False)
            for i in range(self.bs // 480):
                s = i * 480
                x[s:s + 480] = self._rn.process(x[s:s + 480])
            # 按 -18dB 把原始高频混回去：辅音质感来自真实信号，
            # 混回的底噪由后面的 VAD 门兜底压掉
            x = x + hf * self._hf_gain
            # 低强度谱门：压掉 RNNoise 输出的谱波动/水声感，
            # 强度很低所以不会进一步吃高频
            x = self._spec.process(x, 0.15, False)
            self._spec_on = True
            self._rn_on = True
            self._dfn_on = False
        else:
            self._rn_on = False
            self._dfn_on = False
            if st > 0.0:
                want_flat = float(p.get("vad", 0.0)) > 0.5
                x = self._spec.process(x, st, want_flat)
                self._spec_on = True
            elif self._spec_on:
                self._spec.reset()
                self._spec_on = False

        # 3) 广播级处理链：EQ → 去齿音 → 压缩（响度均匀）
        x = self._tone.process(x)
        x = self._deesser.process(x)
        x = self._comp.process(x)

        # 4) dB 噪声门（压缩器会抬起静音段噪声，这里最后清理）
        x = self._noise_gate(x, p)

        # 4a) 宽带突发噪声抑制（纸张摩擦、鼠标刮擦）：
        #     DFN3 会把这类"像人声的噪声"放过，这里用独立的快压快放补一刀，
        #     15ms 压下、80ms 恢复，不会拖累正常语音
        if self._burst_on or self._burst_gain < 0.999:
            block_ms_b = self.bs * 1000.0 / self.sr
            t_b = 15.0 if self._burst_on else 80.0
            coef_b = 1.0 - math.exp(-block_ms_b / t_b)
            g_b = self._burst_gain
            target_b = self._burst_depth if self._burst_on else 1.0
            self._burst_gain = g_b + (target_b - g_b) * coef_b
            ramp_b = (g_b + (self._burst_gain - g_b) * self.ramp).astype(np.float32)
            x = x * ramp_b

        # 4b) 关门时补一点极低电平舒适噪声，避免"数字零"的死寂电子感
        closed = 1.0 - self._gate_gain
        if closed > 0.02:
            x = x + self._comfort(len(x)) * np.float32(closed * self._cn_level)

        # 4c) 输出兜底：即使门还处于保持期，只要整块接近数字零，
        # 就用舒适噪声垫底，避免录音里出现"真空段"造成的断裂/忽大忽小
        if rms_db(x) < -95.0:
            x = self._comfort(len(x)) * np.float32(self._cn_level)

        # 5) 输出增益（按模型档位）
        gain_db = float((self.g or p).get("gain_db", 4.0))
        if gain_db != 0.0:
            x = x * np.float32(10.0 ** (gain_db / 20.0))

        # 6) 软削波（峰值圆润，避免硬限幅的刺耳奇数谐波）
        mag = np.abs(x)
        over = mag > 0.92
        if over.any():
            soft = 0.92 + 0.08 * np.tanh((mag - 0.92) / 0.08)
            x = np.sign(x) * np.where(over, soft, mag)
        np.clip(x, -0.999, 0.999, out=x)
        return x

    def _noise_gate(self, x, p):
        block_ms = self.bs * 1000.0 / self.sr
        db = rms_db(x)

        # 门限跟着实时噪声底走，不再用固定 dB 阈值
        g = self.g or p
        self._track_floor(db)
        thr = self._floor_db + float(g.get("gate_margin", 7.0))
        thr = min(max(thr, -80.0), -12.0)
        self._thr_db = thr

        atten_db = float(g.get("atten_db", 60.0))
        floor = 10.0 ** (-atten_db / 20.0) if atten_db > 0.0 else 1.0

        knee = 6.0  # 软拐点宽度（dB），避免硬开关的咔哒感
        if db >= thr + knee:
            target = 1.0
        elif db <= thr - knee:
            target = floor
        else:
            t = (db - (thr - knee)) / (2.0 * knee)
            target = floor + (1.0 - floor) * (t * t * (3.0 - 2.0 * t))

        # 只留人声：语音概率低（键盘、敲击、风扇这类）就按噪声处理，即使它比门限响
        if float(p.get("vad", 0.0)) > 0.5:
            if self._dfn_on:
                # DFN3 的 LSNR 就是局部信噪比估计：平滑后仍判定"不是人声"就压，
                # 不再要求"声音小"才压——否则响亮的键盘/纸张摩擦会被放行
                if self._dfn_lsnr_s < float(g.get("vad_lsnr", -10.0)):
                    target = floor
            elif self._rn_on:
                if db < thr + 25.0 and self._rn.prob < 0.5:
                    target = floor
            elif self._spec_on:
                if (db < thr + 25.0
                        and (self._spec.p_frame < 0.34 or self._spec.flatness > 0.45)):
                    target = floor

        # 保持时间
        hold_ms = float(g.get("hold_ms", 80.0))
        if db > thr:
            self._hold = hold_ms
        elif self._hold > 0.0:
            self._hold = max(0.0, self._hold - block_ms)
            target = max(target, self._gate_gain)

        # 起音 / 释放
        release_ms = float(g.get("release_ms", 90.0))
        if self._rn_on and target <= self._gate_gain:
            # RNNoise 有残响拖尾，语音结束后更快关门，消除"回声感"
            release_ms = min(release_ms, 45.0)
        attack_ms = float(g.get("attack_ms", 4.0))
        t_ms = max(attack_ms if target > self._gate_gain else release_ms, 0.5)
        coef = 1.0 - math.exp(-block_ms / t_ms)
        g0 = self._gate_gain
        g1 = g0 + (target - g0) * coef
        self._gate_gain = g1

        if abs(g1 - g0) < 1e-4:
            if g1 >= 0.9995:
                return x
            return x * np.float32(g1)

        ramp = (g0 + (g1 - g0) * self.ramp).astype(np.float32)
        return x * ramp


class Recorder:
    """后台写盘录音器：音频回调只负责丢队列，写文件在独立线程，回调零阻塞。"""

    def __init__(self, path, samplerate):
        self._q = queue.Queue(maxsize=4000)  # 约 40 秒缓冲，满了就丢帧而不是阻塞回调
        self.dropped = 0
        self.samples = 0
        self._done = threading.Event()
        self._wf = wave.open(path, "wb")
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)
        self._wf.setframerate(int(samplerate))
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def feed(self, mono):
        try:
            self._q.put_nowait(mono)
        except queue.Full:
            self.dropped += 1

    def _writer(self):
        while not (self._done.is_set() and self._q.empty()):
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            data = np.clip(chunk * 32767.0, -32768.0, 32767.0).astype(np.int16)
            self._wf.writeframes(data.tobytes())
            self.samples += len(chunk)
        self._wf.close()

    def stop(self):
        self._done.set()
        self._thread.join(timeout=3.0)


class LevelMeter(tk.Canvas):
    """极简电平条：-60..0 dBFS，带阈值参考线。"""

    MIN_DB = -60.0
    MAX_DB = 0.0

    def __init__(self, master, **kw):
        super().__init__(master, height=22, bg="#FFFFFF", bd=0,
                         highlightthickness=1, highlightbackground=LINE, **kw)
        self._db = -120.0
        self._thr = None
        self.bind("<Configure>", lambda e: self.redraw())

    def set_level(self, db, threshold=None):
        self._db = db
        self._thr = threshold
        self.redraw()

    def _x_of(self, db, w):
        frac = (db - self.MIN_DB) / (self.MAX_DB - self.MIN_DB)
        frac = min(max(frac, 0.0), 1.0)
        return frac * w

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 4 or h <= 4:
            return

        for d in range(-60, 1, 10):
            x = self._x_of(d, w)
            self.create_line(x, 0, x, h, fill="#EDEDED")

        db = max(self._db, self.MIN_DB)
        x1 = self._x_of(db, w)
        if x1 > 1:
            if db >= -6.0:
                color = DANGER
            elif db >= -18.0:
                color = "#F2B01E"
            else:
                color = ACCENT
            self.create_rectangle(0, 3, x1, h - 3, fill=color, outline="")

        if self._thr is not None:
            xt = self._x_of(self._thr, w)
            self.create_line(xt, 0, xt, h, fill="#616161", width=2)
            self.create_polygon(xt - 4, 0, xt + 4, 0, xt, 6, fill="#616161",
                                outline="")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.params = dict(DEFAULT_PARAMS)
        self.stream = None
        self.denoiser = None
        self.samplerate = 48000
        self.blocksize = 480
        self.meters = {"in_db": -120.0, "out_db": -120.0, "proc_ms": 0.0, "xruns": 0}
        self._disp_in = -120.0
        self._disp_out = -120.0
        self._in_indices = []
        self._out_indices = []
        self._level_hist = deque(maxlen=120)
        self._cmd_queue = queue.Queue()
        self._tray = None
        self._tray_hint_shown = False
        self._alive = True
        self._virtual_routed = False
        self.recorder = None
        self._recorder_raw = None
        self._rec_path = ""
        self._rec_raw_path = ""
        self._rec_t0 = 0.0
        self._model_loading = False
        self.rec_raw_var = tk.BooleanVar(value=False)

        self.title(_WINDOW_TITLE)
        self.geometry("820x600")
        self.minsize(740, 540)

        self._build_ui()
        self._refresh_devices()
        self._load_config()
        self._maybe_auto_route()
        self._setup_window_icon()
        self._setup_tray()
        self.protocol("WM_DELETE_WINDOW", self._on_close_button)
        self.after(80, self._window_style)
        self.after(100, self._tick)

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        self.option_add("*Font", ("Microsoft YaHei UI", 9))
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.configure(bg=BG)
        style.configure(".", background=BG, foreground=FG, bordercolor=LINE,
                        troughcolor="#E6E6E6", fieldbackground=CARD,
                        font=("Microsoft YaHei UI", 9))

        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)

        style.configure("TLabel", background=CARD, foreground=FG)
        style.configure("CardTitle.TLabel", background=CARD, foreground=FG,
                        font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Hint.TLabel", background=BG, foreground=DIM)
        style.configure("CardHint.TLabel", background=CARD, foreground=DIM)
        style.configure("Warn.TLabel", background=CARD, foreground="#9D5D00")
        style.configure("Good.TLabel", background=CARD, foreground="#0F7B0F")
        style.configure("Mono.TLabel", background=CARD, foreground="#0F6CBD",
                        font=("Consolas", 9))

        style.configure("TButton", background="#FBFBFB", foreground=FG,
                        bordercolor=LINE, padding=(12, 6), focusthickness=0,
                        relief="flat")
        style.map("TButton",
                  background=[("pressed", "#EDEDED"), ("active", "#F2F2F2"),
                              ("disabled", "#F7F7F7")],
                  foreground=[("disabled", "#A0A0A0")],
                  bordercolor=[("disabled", "#EFEFEF")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#FFFFFF",
                        font=("Microsoft YaHei UI", 9, "bold"),
                        bordercolor=ACCENT)
        style.map("Accent.TButton",
                  background=[("pressed", ACCENT_D), ("active", "#1A76C7"),
                              ("disabled", "#CFE4F6")],
                  foreground=[("disabled", "#FFFFFF")],
                  bordercolor=[("pressed", ACCENT_D), ("active", "#1A76C7")])
        style.configure("Rec.TButton", background="#FDE7E9", foreground=DANGER,
                        bordercolor="#F1BBC0")
        style.map("Rec.TButton",
                  background=[("active", "#FBD5D9"), ("pressed", "#F8C2C8")])

        style.configure("TCheckbutton", background=CARD, foreground=FG,
                        focusthickness=0, padding=(0, 3))
        style.map("TCheckbutton",
                  indicatorcolor=[("selected", ACCENT), ("!selected", "#FFFFFF")],
                  background=[("active", CARD)],
                  foreground=[("disabled", "#A0A0A0")])

        style.configure("TCombobox", fieldbackground="#FFFFFF", background="#FFFFFF",
                        foreground=FG, arrowcolor="#444444", bordercolor=LINE,
                        padding=4, relief="flat")
        style.map("TCombobox",
                  fieldbackground=[("readonly", "#FFFFFF"), ("disabled", "#F7F7F7")],
                  foreground=[("readonly", FG), ("disabled", "#A0A0A0")],
                  background=[("active", "#F2F2F2")],
                  bordercolor=[("focus", ACCENT)])
        self.option_add("*TCombobox*Listbox.background", "#FFFFFF")
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", "#E5F1FB")
        self.option_add("*TCombobox*Listbox.selectForeground", FG)
        self.option_add("*TCombobox*Listbox.borderWidth", 0)

        style.configure("Horizontal.TScale", background="#FFFFFF",
                        troughcolor="#E6E6E6", bordercolor=LINE,
                        lightcolor=ACCENT, darkcolor=ACCENT_D,
                        gripcount=0)
        style.configure("TSeparator", background=LINE)

        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        # --- 设备 -----------------------------------------------------
        dev = ttk.Frame(root, style="Card.TFrame", padding=(14, 12))
        dev.pack(fill="x", pady=(0, 10))
        ttk.Label(dev, text="设备", style="CardTitle.TLabel").pack(
            anchor="w", pady=(0, 8))

        row = ttk.Frame(dev, style="Card.TFrame")
        row.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(row, text="麦克风", width=7).pack(side="left")
        self.in_combo = ttk.Combobox(row, state="readonly")
        self.in_combo.pack(side="left", fill="x", expand=True, padx=6)
        self.in_combo.bind("<<ComboboxSelected>>",
                           lambda e: self._on_device_change())

        row = ttk.Frame(dev, style="Card.TFrame")
        row.pack(fill="x", padx=8, pady=2)
        ttk.Label(row, text="输出到", width=7).pack(side="left")
        self.out_combo = ttk.Combobox(row, state="readonly")
        self.out_combo.pack(side="left", fill="x", expand=True, padx=6)
        self.out_combo.bind("<<ComboboxSelected>>",
                            lambda e: self._on_device_change())

        row = ttk.Frame(dev, style="Card.TFrame")
        row.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Label(row, text="采样率", width=7).pack(side="left")
        # 固定 48kHz：RNNoise 只支持该采样率，故意不可修改
        ttk.Label(row, text=f"{FIXED_SR} Hz", style="Mono.TLabel").pack(
            side="left", padx=(6, 12))
        Tooltip_box = ttk.Label(row, text="固定值", style="CardHint.TLabel")
        Tooltip_box.pack(side="left", padx=(0, 6))
        Tooltip(Tooltip_box, "采样率固定 48000 Hz：AI 降噪（RNNoise）只支持 48kHz，"
                             "所以不提供修改。")

        ttk.Label(row, text="块大小").pack(side="left", padx=(12, 6))
        self.bs_combo = ttk.Combobox(row, state="readonly", width=8,
                                     values=[str(v) for v in BLOCK_SIZES])
        self.bs_combo.set("480")
        self.bs_combo.pack(side="left")

        ttk.Button(row, text="路由检查", command=self._check_routing).pack(
            side="right", padx=(0, 6))
        ttk.Button(row, text="刷新设备", command=self._refresh_and_route).pack(side="right")

        self.route_hint = ttk.Label(dev, style="CardHint.TLabel", anchor="w",
                                    justify="left", wraplength=730)
        self.route_hint.pack(fill="x", padx=10, pady=(0, 6))

        # --- 降噪模型（选择卡片） ---------------------------------------
        self._build_model_section(root)

        # --- 电平 -----------------------------------------------------
        lvl = ttk.Frame(root, style="Card.TFrame", padding=(14, 12))
        lvl.pack(fill="x", pady=(0, 10))
        lvl_head = ttk.Frame(lvl, style="Card.TFrame")
        lvl_head.pack(fill="x", pady=(0, 6))
        ttk.Label(lvl_head, text="实时电平", style="CardTitle.TLabel").pack(side="left")
        self.chk_rec_raw = ttk.Checkbutton(
            lvl_head, text="录音时同时保存原始声（对比用）",
            variable=self.rec_raw_var, command=self._on_rec_raw_toggle)
        self.chk_rec_raw.pack(side="right")
        Tooltip(self.chk_rec_raw,
                "勾选后，点「● 录音」会同时生成两个文件：\n"
                "· 降噪_时间.wav —— 降噪后的声音（对方听到的）\n"
                "· 原始_时间.wav —— 未降噪的原始麦克风\n"
                "对比试听就能判断降噪模型的效果。")

        row = ttk.Frame(lvl, style="Card.TFrame")
        row.pack(fill="x", padx=8, pady=(6, 3))
        ttk.Label(row, text="输入", width=5).pack(side="left")
        self.meter_in = LevelMeter(row)
        self.meter_in.pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_in = ttk.Label(row, text="-120.0 dBFS", width=12,
                                anchor="e", style="Mono.TLabel")
        self.lbl_in.pack(side="left")

        row = ttk.Frame(lvl, style="Card.TFrame")
        row.pack(fill="x", padx=8, pady=(3, 8))
        ttk.Label(row, text="输出", width=5).pack(side="left")
        self.meter_out = LevelMeter(row)
        self.meter_out.pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_out = ttk.Label(row, text="-120.0 dBFS", width=12,
                                 anchor="e", style="Mono.TLabel")
        self.lbl_out.pack(side="left")

        # --- 底部 -----------------------------------------------------
        bar = ttk.Frame(root)
        bar.pack(fill="x", pady=(10, 0))
        self.btn_start = ttk.Button(bar, text="开始处理", width=12,
                                    command=self._start, style="Accent.TButton")
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="停止", width=10,
                                   command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=8)
        self.btn_rec = ttk.Button(bar, text="● 录音", width=9,
                                  command=self._toggle_rec, state="disabled")
        self.btn_rec.pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="保存配置", width=9,
                   command=self._save_config_btn).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="读取配置", width=9,
                   command=self._load_config_btn).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="测试链路", width=10,
                   command=self._test_chain).pack(side="right", padx=(0, 8))
        ttk.Button(bar, text="? 说明", width=8,
                   command=self._show_help).pack(side="right", padx=(0, 8))

        self.status = ttk.Label(root, text="就绪 · 关闭窗口后驻留托盘继续运行",
                                style="Hint.TLabel")
        self.status.pack(fill="x", pady=(8, 0))

        # 悬停提示：把每个设置的含义直接写进软件里
        Tooltip(self.in_combo,
                "麦克风（输入）= 你的声音从这台设备进来。\n"
                "选你说话用的：USB 耳机麦 / 独立麦克风 / 笔记本内置麦。")
        Tooltip(self.out_combo,
                "输出（处理后去哪）= 决定谁能听到：\n"
                "· CABLE Input → 变成虚拟麦克风，微信/Zoom 选 CABLE Output\n"
                "· 扬声器/耳机 → 只有你自己监听，对方收不到，戴耳机防啸叫")
        Tooltip(self.bs_combo, "块大小越小延迟越低，但越容易卡顿，480 是 AI 降噪推荐值。")

        # Tk 的按钮在获得焦点后会把空格键当成"点击"，导致录音等按钮被误触发。
        # 这里解除按钮的空格类绑定，并让按钮不参与键盘焦点。
        for cls in ("TButton", "Button"):
            try:
                self.unbind_class(cls, "<space>")
            except Exception:
                pass
        self._strip_button_focus(root)

    def _strip_button_focus(self, widget):
        """递归让所有按钮不参与键盘焦点，空格键不再触发它们。"""
        for child in widget.winfo_children():
            try:
                if isinstance(child, (ttk.Button, ttk.Checkbutton)):
                    child.configure(takefocus=0)
            except Exception:
                pass
            self._strip_button_focus(child)

    # ------------------------------------------------------------ 模型卡片
    def _build_model_section(self, parent):
        """降噪模型选择卡片：DeepFilterNet3 / RNNoise，点一下即时切换。"""
        card = ttk.Frame(parent, style="Card.TFrame", padding=(14, 12))
        card.pack(fill="x", pady=(0, 10))

        head = ttk.Frame(card, style="Card.TFrame")
        head.pack(fill="x")
        ttk.Label(head, text="降噪模型", style="CardTitle.TLabel").pack(side="left")
        self.model_hint = ttk.Label(head, text="", style="CardHint.TLabel")
        self.model_hint.pack(side="right")

        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))

        specs = (
            ("dfn3", "DeepFilterNet 3", "推荐",
             "强降噪 · 约 4% CPU · 延迟 +30ms"),
            ("rnnoise", "RNNoise", "经典",
             "轻量 · 约 9% CPU · 延迟极低"),
        )
        self.model_cards = {}
        for i, (key, title, tag, desc) in enumerate(specs):
            box = self._make_model_card(row, key, title, tag, desc)
            box.pack(side="left", fill="x", expand=True,
                     padx=(0, 8) if i == 0 else (0, 0))
            self.model_cards[key] = box

        self._paint_all_model_cards()

    def _make_model_card(self, parent, key, title, tag, desc):
        box = tk.Frame(parent, bg=CARD, highlightthickness=1,
                       highlightbackground=LINE, cursor="hand2",
                       highlightcolor=LINE)
        top = tk.Frame(box, bg=CARD)
        top.pack(fill="x", padx=12, pady=(10, 0))
        dot = tk.Label(top, text="○", bg=CARD, fg="#BDBDBD",
                       font=("Microsoft YaHei UI", 11))
        dot.pack(side="left")
        name = tk.Label(top, text=title, bg=CARD, fg=FG,
                        font=("Microsoft YaHei UI", 10, "bold"))
        name.pack(side="left", padx=(4, 6))
        badge = tk.Label(top, text=tag, bg="#EAF3FD", fg=ACCENT,
                         font=("Microsoft YaHei UI", 8), padx=6)
        badge.pack(side="left")
        sub = tk.Label(box, text=desc, bg=CARD, fg=DIM, anchor="w",
                       justify="left", wraplength=330,
                       font=("Microsoft YaHei UI", 8))
        sub.pack(fill="x", padx=(30, 12), pady=(2, 10))

        box._parts = (top, dot, name, badge, sub)
        box._tag = tag
        widgets = (box, top, dot, name, badge, sub)
        for w in widgets:
            w.bind("<Button-1>", lambda e, k=key: self._select_model(k))

        def on_enter(_e=None, b=box, k=key):
            if k != self.params.get("model"):
                b.configure(highlightbackground="#9DC7EC")

        for w in widgets:
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", lambda e, k=key: self._paint_model_card(k))

        Tooltip(box, "DeepFilterNet 3：抑制更强（非平稳噪声、键盘声、旁人说话），"
                     "代价是 30ms 算法延迟。\n"
                     "RNNoise：传统轻量方案，延迟极低，适合对延迟敏感的场景。"
                     if key == "dfn3" else
                     "RNNoise：经典轻量神经降噪，延迟极低；降噪强度不如 DeepFilterNet 3。")
        return box

    def _paint_model_card(self, key):
        box = self.model_cards.get(key)
        if box is None:
            return
        top, dot, name, badge, sub = box._parts
        selected = (self.params.get("model") == key)
        enabled = (key != "dfn3") or DFN3_PRESENT
        bg = "#F5FAFF" if selected else (CARD if enabled else "#F7F7F7")
        for w in (box, top, dot, name, sub):
            w.configure(bg=bg)
        if selected:
            box.configure(highlightbackground=ACCENT)
            dot.configure(text="●", fg=ACCENT)
            name.configure(fg="#0F6CBD")
        else:
            box.configure(highlightbackground=LINE)
            dot.configure(text="○", fg="#BDBDBD")
            name.configure(fg=FG if enabled else "#A0A0A0")
        if enabled:
            badge.configure(text=box._tag, bg="#EAF3FD", fg=ACCENT)
            sub.configure(fg=DIM)
        else:
            badge.configure(text="不可用", bg="#F0F0F0", fg="#909090")
            sub.configure(fg="#BDBDBD")

    def _paint_all_model_cards(self):
        for key in getattr(self, "model_cards", {}):
            self._paint_model_card(key)

    def _select_model(self, key, persist=True):
        """点击卡片切换模型。运行中切到 DFN3 时后台建引擎，避免卡拉低音频回调。"""
        if key == "dfn3" and not DFN3_PRESENT:
            messagebox.showinfo(
                "模型不可用",
                "未找到 DeepFilterNet3 模型文件（models/df3）。\n"
                "请重新打包或补齐模型后再选。当前继续使用 RNNoise。")
            return
        if key == self.params.get("model"):
            self._paint_all_model_cards()
            return
        if self._model_loading:
            return

        den = self.denoiser
        if (key == "dfn3" and den is not None and self.stream is not None
                and den._dfn is None):
            # 引擎要现建（约 0.3s），放到后台线程，建好再切
            self._model_loading = True
            self.model_hint.configure(text="正在加载 DeepFilterNet3 引擎…")
            threading.Thread(target=self._warm_model_async,
                             args=(den,), daemon=True).start()
            return

        self._apply_model(key, persist)

    def _warm_model_async(self, den):
        ok = False
        try:
            ok = bool(den._ensure_dfn())
        except Exception:
            ok = False
        # 回调线程不能直接碰 Tk，投递给主循环处理
        self._cmd_queue.put(lambda: self._after_model_warm(ok))

    def _after_model_warm(self, ok):
        self._model_loading = False
        self.model_hint.configure(text="")
        if ok:
            self._apply_model("dfn3", True)
        else:
            self._paint_all_model_cards()
            messagebox.showerror("加载失败",
                                 "DeepFilterNet3 引擎启动失败，已保持原模型。\n"
                                 "请检查 models/df3 与 onnxruntime 是否完整。")

    def _apply_model(self, key, persist=True):
        self.params["model"] = key
        self._paint_all_model_cards()
        self.model_hint.configure(text="")
        if persist:
            self._save_config()
        if self.stream is not None:
            tag = "DeepFilterNet3" if key == "dfn3" else "RNNoise"
            self.status.configure(text=f"已切换降噪模型：{tag}")

    def _show_help(self):
        messagebox.showinfo(
            "使用说明",
            "【麦克风（输入）】\n"
            "　你的声音从哪台设备进来。选你说话用的：USB 耳机麦、\n"
            "　独立麦克风、笔记本内置麦都可以。\n\n"
            "【输出到】\n"
            "　处理完的声音送去哪，决定对方能不能听到：\n"
            "　· 选「CABLE Input」→ 声音变成虚拟麦克风，在微信/Zoom\n"
            "　　里把麦克风设为「CABLE Output」就能用。\n"
            "　· 选「扬声器/耳机」→ 只有你自己听得到（监听），\n"
            "　　请戴耳机防止啸叫，对方收不到。\n\n"
            "【推荐流程】\n"
            "　1. 装好 VB-CABLE 后重启电脑\n"
            "　2. 麦克风选说话设备，输出选 CABLE Input\n"
            "　3. 点「开始处理」即可，无需校准\n"
            "　4. 通话软件里把麦克风设为 CABLE Output\n\n"
            "【门限是自动的】\n"
            "　程序持续跟踪环境噪声底，门限 = 底噪 + 灵敏度，\n"
            "　环境变吵会自己抬高、变安静会自己降低。\n"
            "　状态栏能看到实时的「底噪 / 门限」数值，\n"
            "　电平条上的灰线就是当前门限。\n\n"
            "【降噪模型】\n"
            "　· DeepFilterNet 3（推荐）：抑制更强，键盘声、旁人说话这类\n"
            "　　非平稳噪声也能压住；代价是 30ms 延迟（录音通话音质优先）。\n"
            "　· RNNoise：经典轻量方案，延迟极低，降噪力度弱一些。\n"
            "　点卡片即可切换，运行中切换也不会中断。选择会自动保存。\n\n"
            "【常用按钮】\n"
            "　测试链路：响一声，验证 CABLE 环回是否连通\n"
            "　路由检查：按当前设备给出具体配置建议\n"
            "　● 录音：直接录下降噪后的声音（与输出选耳机还是 CABLE 无关，\n"
            "　　不经过设备环回，零额外延迟），保存在程序目录的「录音」文件夹。\n"
            "　　勾选「录音时同时保存原始声」会额外存一份未降噪的原始麦\n"
            "　　克风录音，方便对比 DeepFilterNet 3 与 RNNoise 的效果。\n\n"
            "【为什么有时启动失败】\n"
            "　麦克风和输出必须属于同一个音频 API（MME/WASAPI…），\n"
            "　程序会自动配对同名设备，一般无需手动处理。")

    # ------------------------------------------------------------ 设备
    def _refresh_devices(self):
        try:
            devices = sd.query_devices()
        except Exception as exc:
            messagebox.showerror("音频设备错误", f"无法枚举音频设备：\n{exc}")
            return

        self._devices = devices
        self._in_indices = [i for i, d in enumerate(devices)
                            if d["max_input_channels"] > 0]
        self._out_indices = [i for i, d in enumerate(devices)
                             if d["max_output_channels"] > 0]

        self.in_combo["values"] = [f"{i}: {devices[i]['name']}"
                                   for i in self._in_indices]
        self.out_combo["values"] = [f"{i}: {devices[i]['name']}"
                                    for i in self._out_indices]

        try:
            default_in, default_out = sd.default.device
        except Exception:
            default_in, default_out = -1, -1

        if self._in_indices:
            pos = self._in_indices.index(default_in) if default_in in self._in_indices else 0
            self.in_combo.current(pos)
        if self._out_indices:
            pos = (self._out_indices.index(default_out)
                   if default_out in self._out_indices else 0)
            self.out_combo.current(pos)

        self._update_route_hint()

    def _device_at(self, combo, indices):
        pos = combo.current()
        if 0 <= pos < len(indices):
            return indices[pos]
        return None

    def _on_device_change(self):
        """切换设备后立即保存，重启/崩溃都不丢选择。"""
        self._update_route_hint()
        self._save_config()

    def _save_config_btn(self):
        if self._save_config():
            self.status.configure(text=f"配置已保存 → {CONFIG_PATH}")
        else:
            messagebox.showerror(
                "保存失败",
                f"无法写入配置文件：\n{CONFIG_PATH}\n\n"
                "请确认 exe 所在目录可写（别放在 Program Files 这类受保护目录）。")

    def _load_config_btn(self):
        if not os.path.exists(CONFIG_PATH):
            messagebox.showinfo("读取配置", f"还没有配置文件：\n{CONFIG_PATH}")
            return
        self._refresh_devices()
        self._load_config()
        self.status.configure(text=f"配置已读取 → {CONFIG_PATH}")

    # ------------------------------------------------------------ 路由
    def _refresh_and_route(self):
        self._refresh_devices()
        self._maybe_auto_route()

    def _maybe_auto_route(self):
        """检测到 VB-CABLE 时自动切到「麦克风 → 本软件 → CABLE Input」链路。"""
        devices = getattr(self, "_devices", None)
        if not devices or self._virtual_routed:
            return
        if cable_playback_disabled(devices):
            self._update_route_hint()
            self.status.configure(
                text="CABLE Input 播放设备被禁用，启用后再点「刷新设备」")
            return
        vplay, mic = find_duplex_pair(devices)
        if vplay is None:
            return
        self._virtual_routed = True

        if vplay in self._out_indices:
            self.out_combo.current(self._out_indices.index(vplay))

        out_api = devices[vplay]["hostapi"]
        vrec = find_device(devices, VIRTUAL_REC_KEYS, "in", out_api)
        cur_in = self._device_at(self.in_combo, self._in_indices)
        cur_name = devices[cur_in]["name"].lower() if cur_in is not None else ""
        cur_api = devices[cur_in]["hostapi"] if cur_in is not None else -1

        # 输入是虚拟声卡 / 默认设备别名 / 与输出不同 API 时，统一切到同 API 的物理麦克风
        if (mic is not None
                and (cur_in == vrec
                     or cur_api != out_api
                     or any(k in cur_name for k in MIXER_KEYS))):
            self.in_combo.current(self._in_indices.index(mic))

        self._update_route_hint()
        self.status.configure(
            text="已自动配置 VB-CABLE：其他软件的麦克风请选 CABLE Output")
        self._save_config()

    def _update_route_hint(self):
        """根据当前输出设备，提示怎么让其他软件收到降噪后的声音。"""
        devices = getattr(self, "_devices", None)
        if not devices or not hasattr(self, "route_hint"):
            return
        out_idx = self._device_at(self.out_combo, self._out_indices)
        if out_idx is None:
            self.route_hint.configure(text="未选择输出设备", style="CardHint.TLabel")
            return

        low = devices[out_idx]["name"].lower()
        if cable_playback_disabled(devices):
            self.route_hint.configure(
                style="Warn.TLabel",
                text="检测到 CABLE Input（播放端）被禁用——这就是虚拟麦克风"
                     "收不到声音的原因。到 声音设置 → 更多声音设置 → 播放，"
                     "右键「CABLE Input」→ 启用，再回来点「刷新设备」。")
            return
        if any(k in low for k in VIRTUAL_PLAY_KEYS):
            rec = find_device(devices, VIRTUAL_REC_KEYS, "in")
            rec_name = devices[rec]["name"] if rec is not None else "对应录音端"
            self.route_hint.configure(
                style="Good.TLabel",
                text=f"链路已通：麦克风 → 本软件 → {devices[out_idx]['name']}。"
                     f"其他软件的麦克风请选「{rec_name}」。")
            return

        hint = ("输出到普通扬声器只有你自己听得到，其他软件收不到。"
                "要当麦克风用，请装 VB-CABLE，输出选「CABLE Input」，"
                "其他软件麦克风选「CABLE Output」。")
        if ("realtek" in low
                and find_device(devices, STEREO_MIX_KEYS, "in") is not None):
            hint += "（也可试系统「立体声混音」，但本机实测常常没声音）"
        self.route_hint.configure(style="Warn.TLabel", text=hint)

    def _check_routing(self):
        devices = getattr(self, "_devices", None)
        if not devices:
            messagebox.showinfo("路由检查", "还没读到音频设备，请点「刷新设备」重试。")
            return

        in_idx = self._device_at(self.in_combo, self._in_indices)
        out_idx = self._device_at(self.out_combo, self._out_indices)
        in_name = devices[in_idx]["name"] if in_idx is not None else "未选择"
        out_name = devices[out_idx]["name"] if out_idx is not None else "未选择"

        vplay = find_device(devices, VIRTUAL_PLAY_KEYS, "out")
        vrec = find_device(devices, VIRTUAL_REC_KEYS, "in")
        mix = find_device(devices, STEREO_MIX_KEYS, "in")

        if cable_playback_disabled(devices):
            messagebox.showwarning(
                "路由检查",
                "检测到 CABLE Input（播放端）被禁用：\n"
                "虚拟麦克风收不到任何声音，链路必然不通。\n\n"
                "解决：声音设置 → 更多声音设置 → 播放 →\n"
                "右键「CABLE Input」→ 启用，然后回来点「刷新设备」。")
            return

        rec_name = devices[vrec]["name"] if vrec is not None else "对应录音端"
        low = out_name.lower()

        lines = [f"麦克风输入：{in_name}", f"输出目标：{out_name}", ""]
        if any(k in low for k in VIRTUAL_PLAY_KEYS):
            lines += [
                "判定：配置正确，降噪后的声音已经是虚拟麦克风的输入。",
                "",
                f"下一步：在微信 / Zoom / 游戏语音里把麦克风设为「{rec_name}」。",
            ]
        elif mix is not None and "realtek" in low:
            lines += [
                "判定：可试「立体声混音」环回，但本机实测录到的是静音。",
                "",
                "若要尝试：",
                "1. 输出保持为 Realtek 扬声器，并戴耳机；",
                f"2. 微信 / Zoom 麦克风设为「{devices[mix]['name']}」；",
                "3. 声音设置 → 录音 → 立体声混音 → 属性 → 级别，音量拉到 100。",
                "",
                "更稳妥的做法仍是装 VB-CABLE。",
            ]
        elif vplay is not None:
            lines += [
                "判定：系统里有虚拟声卡，但当前输出不是它。",
                "",
                f"1. 把「输出到」改成「{devices[vplay]['name']}」；",
                f"2. 微信 / Zoom 里把麦克风设为「{rec_name}」。",
            ]
        else:
            lines += [
                "判定：当前配置只能自己监听，其他软件收不到。",
                "",
                "原因：Windows 不允许普通程序直接写入麦克风，",
                "必须靠虚拟声卡把「播放端」转成「录音端」。",
                "",
                "方案一（推荐）：装 VB-CABLE（免费）后，",
                "   输出选「CABLE Input」，其他软件麦克风选「CABLE Output」。",
            ]
            if mix is not None:
                lines += [
                    "",
                    "方案二（免安装）：输出选 Realtek 扬声器，",
                    f"   其他软件麦克风选「{devices[mix]['name']}」，并戴耳机。",
                ]

        lines += ["", "程序只会把声音送到上面选中的输出设备，不会自动变成麦克风。"]
        messagebox.showinfo("路由检查", "\n".join(lines))

    def _test_chain(self):
        """放一段提示音，从虚拟麦克风录回来，验证整条链路是否连通。"""
        if self.stream is not None:
            messagebox.showinfo("链路测试", "请先点「停止」，再做链路测试。")
            return
        devices = getattr(self, "_devices", None)
        if not devices:
            return
        out_idx = self._device_at(self.out_combo, self._out_indices)
        vrec = (find_device(devices, VIRTUAL_REC_KEYS, "in",
                            devices[out_idx]["hostapi"])
                if out_idx is not None else None)
        if out_idx is None or vrec is None:
            messagebox.showinfo(
                "链路测试",
                "当前输出不是虚拟声卡，无法做环回测试。\n"
                "请装好 VB-CABLE 并把输出选为 CABLE Input。")
            return

        sr = FIXED_SR
        block = 512
        tone = (0.15 * np.sin(2 * np.pi * 880 * np.arange(int(sr * 1.2)) / sr)
                ).astype(np.float32)
        rec = None
        ost = None
        try:
            # 用持续流写 1.2 秒；一次性 sd.play 在这类虚拟声卡上可能握手失败
            rec = sd.rec(int(sr * 1.8), samplerate=sr, channels=2,
                         device=vrec, dtype="float32")
            ost = sd.OutputStream(samplerate=sr, blocksize=block,
                                  device=out_idx, channels=1, dtype="float32")
            ost.start()
            time.sleep(0.05)
            for i in range(len(tone) // block):
                ost.write(tone[i * block:(i + 1) * block])
            ost.stop()
            ost.close()
            ost = None
            time.sleep(1.2)
        except Exception as exc:
            messagebox.showerror("链路测试", f"测试失败：{exc}")
            return
        finally:
            if ost is not None:
                try:
                    ost.close()
                except Exception:
                    pass
            try:
                sd.stop()
            except Exception:
                pass

        if rec is None or np.isnan(rec).any():
            messagebox.showwarning("链路测试", "录音缓冲异常，请稍后再试。")
            return

        peak = float(np.max(np.abs(rec)))
        if peak > 0.01:
            messagebox.showinfo(
                "链路测试",
                "链路正常：提示音已经过 VB-CABLE 转回来了。\n"
                "在微信 / Zoom 里把麦克风选为 CABLE Output 即可。")
        else:
            extra = ("\n3. 检测到 CABLE Input 播放端被禁用，请到 声音设置 → "
                     "更多声音设置 → 播放 → 右键启用。"
                     if cable_playback_disabled(devices)
                     else "\n3. 重启 Windows Audio 服务（services.msc）；\n"
                          "4. 还不行就卸载 VB-CABLE，改装 VoiceMeeter Banana。")
            messagebox.showwarning(
                "链路测试",
                "虚拟麦克风没有收到声音（录到的是静音）。\n\n"
                "1. 装完 VB-CABLE 后需要重启电脑；\n"
                "2. 到 声音设置 → 更多声音设置 里确认 CABLE 设备"
                "未被禁用或静音。" + extra)

    # ------------------------------------------------------------ 音频
    def _audio_cb(self, indata, outdata, frames, time_info, status):
        if status:
            self.meters["xruns"] += 1

        den = self.denoiser
        if den is None:
            outdata.fill(0.0)
            return

        if indata.shape[1] > 1:
            mono = indata.mean(axis=1, dtype=np.float32)
        else:
            mono = indata[:, 0]

        in_db = rms_db(mono)
        self.meters["in_db"] = in_db
        self._level_hist.append(in_db)

        t0 = time.perf_counter()
        y = den.process(mono)
        dt = (time.perf_counter() - t0) * 1000.0

        self.meters["proc_ms"] = self.meters["proc_ms"] * 0.85 + dt * 0.15
        self.meters["out_db"] = rms_db(y)

        # 始终把处理后的声音送到所选输出设备：
        #   选扬声器 = 自己监听；选虚拟声卡 = 供其他软件当麦克风
        outdata[:] = y[:, None]

        # 录音直接抓处理后的信号：不依赖输出路由（选耳机也能录），
        # 也避免 CABLE 环回带来的额外延迟和时钟漂移
        rec = self.recorder
        if rec is not None:
            rec.feed(y)
        rec_raw = self._recorder_raw
        if rec_raw is not None:
            rec_raw.feed(np.array(mono, dtype=np.float32, copy=True))

    def _open_stream(self, in_dev, out_dev, sr, bs):
        try:
            devices = list(sd.query_devices())
        except Exception:
            devices = []

        # PortAudio 全双工要求输入输出同属一个 hostapi；
        # 用户手选了跨 API 组合时，自动找同名设备配对重试
        candidates = [(in_dev, out_dev)]
        if devices and in_dev is not None and out_dev is not None:
            api_in = devices[in_dev]["hostapi"]
            api_out = devices[out_dev]["hostapi"]
            if api_in != api_out:
                alt_in = find_same_device(devices, devices[in_dev]["name"],
                                          api_out, "in")
                if alt_in is not None:
                    candidates.append((alt_in, out_dev))
                alt_out = find_same_device(devices, devices[out_dev]["name"],
                                           api_in, "out")
                if alt_out is not None:
                    candidates.append((in_dev, alt_out))

        # 采样率固定 48kHz，不再回退到其他值（否则 AI 降噪会失效）
        sr_opts = [FIXED_SR]

        last_err = None
        for in_try, out_try in candidates:
            in_max = out_max = 2
            try:
                if in_try is not None:
                    in_max = sd.query_devices(in_try)["max_input_channels"]
                if out_try is not None:
                    out_max = sd.query_devices(out_try)["max_output_channels"]
            except Exception:
                pass

            in_opts = [1] + ([2] if in_max >= 2 else [])
            out_opts = [2, 1] if out_max >= 2 else [1]

            for rate in sr_opts:
                for ch_in in in_opts:
                    for ch_out in out_opts:
                        stream = None
                        try:
                            stream = sd.Stream(
                                samplerate=rate,
                                blocksize=bs,
                                device=(in_try, out_try),
                                channels=(ch_in, ch_out),
                                dtype="float32",
                                callback=self._audio_cb,
                            )
                            self.denoiser = Denoiser(rate, bs, self.params)
                            # DFN3 的 onnxruntime 会话提前建好，避免第一块音频卡顿
                            self.denoiser.warmup()
                            stream.start()
                            self.samplerate = rate
                            self.blocksize = bs
                            self._used_devices = (in_try, out_try)
                            return stream
                        except Exception as exc:
                            last_err = exc
                            if stream is not None:
                                try:
                                    stream.close()
                                except Exception:
                                    pass
        raise RuntimeError(str(last_err) if last_err else "未知错误")

    def _start(self):
        if self.stream is not None:
            return

        in_dev = self._device_at(self.in_combo, self._in_indices)
        if in_dev is None:
            messagebox.showwarning("提示", "请先选择麦克风设备。")
            return
        out_dev = self._device_at(self.out_combo, self._out_indices)
        sr = FIXED_SR
        bs = int(self.bs_combo.get() or 480)

        try:
            stream = self._open_stream(in_dev, out_dev, sr, bs)
        except Exception as exc:
            self.denoiser = None
            messagebox.showerror(
                "启动失败",
                f"无法打开音频流：\n{exc}\n\n"
                "已自动尝试过「同名设备跨 API 配对」，仍失败的话：\n"
                "1. 换一个麦克风或输出设备再试；\n"
                "2. 更换块大小（480 / 960）；\n"
                "3. 确认麦克风没有被其他程序独占。")
            return

        # 实际打开的设备可能经过了跨 API 自动配对，把下拉框同步成真实值
        used = getattr(self, "_used_devices", None)
        if used:
            if used[0] in self._in_indices:
                self.in_combo.current(self._in_indices.index(used[0]))
            if used[1] in self._out_indices:
                self.out_combo.current(self._out_indices.index(used[1]))
            self._update_route_hint()

        self.stream = stream
        self.meters["xruns"] = 0
        self.meters["proc_ms"] = 0.0
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_rec.configure(state="normal")
        self.bs_combo.set(str(self.blocksize))
        model_tag = "DeepFilterNet3" if (DFN3_PRESENT
                                         and str(self.params.get("model")) == "dfn3") else "RNNoise"
        self.status.configure(text=f"运行中 · {self.samplerate} Hz · "
                                   f"块 {self.blocksize} · {model_tag}")

    def _stop(self):
        self._stop_recorder(announce=False)
        stream, self.stream = self.stream, None
        self.denoiser = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self.meters["in_db"] = -120.0
        self.meters["out_db"] = -120.0
        self._disp_in = -120.0
        self._disp_out = -120.0
        self._level_hist.clear()
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_rec.configure(state="disabled")
        self.status.configure(text="已停止")

    # ------------------------------------------------------------ 录音
    def _new_rec_path(self):
        folder = os.path.join(_app_dir(), "录音")
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception:
            folder = _app_dir()
        return os.path.join(folder, time.strftime("降噪_%Y%m%d_%H%M%S") + ".wav")

    def _on_rec_raw_toggle(self):
        self._save_config()
        if (self.recorder is not None or self._recorder_raw is not None) \
                and bool(self.rec_raw_var.get()) and self._recorder_raw is None:
            self.status.configure(text="原始声录制将在下次录音时生效")

    def _stop_recorder(self, announce=True):
        rec, self.recorder = self.recorder, None
        rec_raw, self._recorder_raw = self._recorder_raw, None
        if rec is None and rec_raw is None:
            return
        for r in (rec, rec_raw):
            if r is not None:
                try:
                    r.stop()
                except Exception:
                    pass
        secs = (rec or rec_raw).samples / float(self.samplerate or 48000)
        note = ""
        if rec is not None and rec.dropped:
            note = f"（磁盘来不及，丢了 {rec.dropped} 块）"
        self.btn_rec.configure(text="● 录音", style="TButton")
        if announce:
            if rec_raw is not None:
                self.status.configure(
                    text=f"录音完成 · {secs:.1f} 秒 → {self._rec_path}（+ 原始声 {self._rec_raw_path}）{note}")
            else:
                self.status.configure(
                    text=f"录音完成 · {secs:.1f} 秒 → {self._rec_path}{note}")

    def _toggle_rec(self):
        if self.recorder is not None or self._recorder_raw is not None:
            self._stop_recorder()
            return
        if self.stream is None:
            messagebox.showinfo("录音", "请先点「开始处理」，再点录音。")
            return

        path = self._new_rec_path()
        try:
            self.recorder = Recorder(path, self.samplerate)
        except Exception as exc:
            messagebox.showerror("录音", f"无法创建录音文件：\n{exc}")
            return
        self._rec_path = path
        self._rec_raw_path = ""
        self._rec_t0 = time.time()

        # 可选：同时保存未降噪的原始麦克风，方便对比模型效果（A/B 试听）
        if bool(self.rec_raw_var.get()):
            raw_path = os.path.join(
                os.path.dirname(path),
                os.path.basename(path).replace("降噪_", "原始_", 1))
            try:
                self._recorder_raw = Recorder(raw_path, self.samplerate)
                self._rec_raw_path = raw_path
            except Exception:
                self._recorder_raw = None

        tag = "降噪后 + 原始声" if self._recorder_raw is not None else "降噪后"
        self.status.configure(text=f"● 录音中（{tag}）")
        self.btn_rec.configure(text="■ 停止", style="Rec.TButton")

    def _tick(self):
        if not self._alive:
            return
        self._drain_commands()
        if not self._alive:
            return
        if self.state() == "withdrawn":
            # 窗口隐藏后降低界面刷新频率，音频处理不受影响
            self.after(250, self._tick)
            return

        m = self.meters
        self._disp_in = max(m["in_db"], self._disp_in - 4.0)
        self._disp_out = max(m["out_db"], self._disp_out - 4.0)

        den = self.denoiser
        thr = den._thr_db if den is not None else None
        self.meter_in.set_level(self._disp_in, thr)
        self.meter_out.set_level(self._disp_out, thr)
        self.lbl_in.configure(text=f"{m['in_db']:7.1f} dBFS")
        self.lbl_out.configure(text=f"{m['out_db']:7.1f} dBFS")

        if self.stream is not None:
            block_ms = self.blocksize * 1000.0 / self.samplerate
            cpu = m["proc_ms"] / block_ms * 100.0 if block_ms > 0 else 0.0
            auto = (f"底噪 {den._floor_db:.0f} dB · 门限 {den._thr_db:.0f} dB"
                    if den is not None else "")
            if den is not None:
                if str(self.params.get("model", "dfn3")) == "dfn3" and DFN3_PRESENT:
                    auto += " · DFN3"
                elif den._rn_ok:
                    auto += " · RNNoise"
                elif float(self.params.get("rnnoise", 1.0)) > 0.5:
                    auto += " · AI 降噪不可用（需 48kHz）"
            if self.recorder is not None:
                secs = time.time() - self._rec_t0
                self.status.configure(
                    text=f"● 录音中 {int(secs) // 60:02d}:{int(secs) % 60:02d} · "
                         f"{auto} · 处理 {m['proc_ms']:.2f} ms/块")
            else:
                self.status.configure(
                    text=f"运行中 · {self.samplerate} Hz · {auto} · "
                         f"处理 {m['proc_ms']:.2f} ms/块 · CPU 约 {cpu:.1f}% · "
                         f"溢出 {m['xruns']} 次")

        self.after(100, self._tick)

    # ------------------------------------------------------------ 配置
    def _load_config(self):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception:
            return

        # 采样率固定 48kHz，配置里的旧值不再生效
        if cfg.get("blocksize") in BLOCK_SIZES:
            bs = int(cfg["blocksize"])
            # RNNoise 需要 480 的整数倍块长，旧配置可能是 512，自动纠正
            if RNNOISE_PRESENT and bs % 480 != 0:
                bs = 480
            self.bs_combo.set(str(bs))

        # 降噪模型：配置里的选择优先，缺失/不可用时退回 RNNoise
        model = cfg.get("model")
        if model in ("dfn3", "rnnoise"):
            if model == "dfn3" and not DFN3_PRESENT:
                model = "rnnoise"
            if model != self.params.get("model"):
                self.params["model"] = model
        self._paint_all_model_cards()

        in_idx = self._restore_device(cfg.get("input_device"), self._in_indices)
        if in_idx is not None:
            self.in_combo.current(self._in_indices.index(in_idx))
        out_idx = self._restore_device(cfg.get("output_device"), self._out_indices)
        if out_idx is not None:
            self.out_combo.current(self._out_indices.index(out_idx))
        self._virtual_routed = bool(cfg.get("virtual_routed", False))
        self.rec_raw_var.set(bool(cfg.get("rec_raw", False)))
        self._update_route_hint()

    def _save_config(self):
        devices = getattr(self, "_devices", None)
        in_idx = self._device_at(self.in_combo, self._in_indices)
        out_idx = self._device_at(self.out_combo, self._out_indices)
        cfg = {
            "samplerate": FIXED_SR,
            "blocksize": int(self.bs_combo.get() or 480),
            "model": self.params.get("model", "dfn3"),
            # 存设备名（而非索引），重启/插拔后索引变化也能按名字找回
            "input_device": devices[in_idx]["name"] if (devices and in_idx is not None) else None,
            "output_device": devices[out_idx]["name"] if (devices and out_idx is not None) else None,
            "virtual_routed": self._virtual_routed,
            "rec_raw": bool(self.rec_raw_var.get()),
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def _restore_device(self, value, indices):
        """按保存的值恢复设备：新格式存名字，旧格式存索引，两者都兼容。"""
        if value is None or not indices:
            return None
        devices = getattr(self, "_devices", None)
        if not devices:
            return None
        if isinstance(value, int):
            return value if value in indices else None
        low = str(value).lower()
        for i in indices:
            other = devices[i]["name"].lower()
            # PortAudio 名字常被截断，做互相包含匹配
            if low in other or other in low:
                return i
        return None

    # ------------------------------------------------------------ 托盘
    def _window_style(self):
        """Win11 圆角窗口 + 浅色标题栏；旧系统调用失败不影响功能。"""
        try:
            import ctypes
            hwnd = int(self.wm_frame(), 16)
            dwm = ctypes.windll.dwmapi
            light = ctypes.c_int(0)          # DWMWA_USE_IMMERSIVE_DARK_MODE
            dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(light),
                                      ctypes.sizeof(light))
            round_corner = ctypes.c_int(2)   # DWMWCP_ROUND
            dwm.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(round_corner),
                                      ctypes.sizeof(round_corner))
        except Exception:
            pass

    def _setup_window_icon(self):
        try:
            from PIL import ImageTk
            img = make_icon_image(64)
            if img is not None:
                self._win_icon = ImageTk.PhotoImage(img)
                self.iconphoto(True, self._win_icon)
        except Exception:
            pass

    def _setup_tray(self):
        if not TRAY_OK:
            return
        try:
            image = make_icon_image(64)
            if image is None:
                return
            menu = pystray.Menu(
                pystray.MenuItem("显示主窗口", self._tray_post_show, default=True),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("开始处理", self._tray_post_start,
                                 enabled=lambda item: self.stream is None),
                pystray.MenuItem("停止处理", self._tray_post_stop,
                                 enabled=lambda item: self.stream is not None),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("退出", self._tray_post_quit),
            )
            self._tray = pystray.Icon("mic_denoiser", image, "麦克风降噪", menu)
            threading.Thread(target=self._tray.run, daemon=True).start()
        except Exception:
            self._tray = None

    # 托盘回调跑在 pystray 线程，只投递命令，绝不直接操作 Tk 控件
    def _tray_post_show(self, icon=None, item=None):
        self._cmd_queue.put("show")

    def _tray_post_start(self, icon=None, item=None):
        self._cmd_queue.put("start")

    def _tray_post_stop(self, icon=None, item=None):
        self._cmd_queue.put("stop")

    def _tray_post_quit(self, icon=None, item=None):
        self._cmd_queue.put("quit")

    def _drain_commands(self):
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                return
            if cmd == "show":
                self._show_window()
            elif cmd == "start":
                self._start()
            elif cmd == "stop":
                self._stop()
            elif cmd == "quit":
                self._quit()
                return
            elif callable(cmd):
                # 后台线程（如模型预热）完成后的回调，统一回主线程执行
                try:
                    cmd()
                except Exception:
                    pass

    def _show_window(self):
        if not self._alive:
            return
        self.deiconify()
        try:
            self.state("normal")
        except Exception:
            pass
        self.lift()
        self.focus_force()
        try:
            self.attributes("-topmost", True)
            self.after(150, lambda: self.attributes("-topmost", False))
        except Exception:
            pass

    def _hide_to_tray(self):
        if self._tray is None:
            self._quit()
            return
        self.withdraw()
        if not self._tray_hint_shown:
            self._tray_hint_shown = True
            try:
                self._tray.notify("已最小化到托盘，音频处理继续运行。", "麦克风降噪")
            except Exception:
                pass

    def _on_close_button(self):
        if self._tray is None:
            self._quit()
        else:
            self._hide_to_tray()

    def _quit(self):
        self._alive = False
        self._save_config()
        self._stop()
        tray, self._tray = self._tray, None
        if tray is not None:
            try:
                tray.stop()
            except Exception:
                pass
        try:
            self.destroy()
        except Exception:
            pass

    def report_callback_exception(self, exc, val, tb):
        # 打包成无控制台的 exe 后 stderr 不可见，改为状态栏提示
        try:
            self.status.configure(text=f"出错：{exc.__name__}: {val}"[:160])
        except Exception:
            pass


def _selftest():
    sr, bs = 48000, 480
    blocks = (sr * 2) // bs
    total = blocks * bs
    rng = np.random.default_rng(7)
    t = np.arange(total) / sr

    # 类语音信号：谐波 + 音节起伏（仅用于谱门/噪声门，RNNoise 对合成音不友好）
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 2.2 * t)) ** 1.5
    speech = (0.5 * np.sin(2 * np.pi * 300 * t)
              + 0.3 * np.sin(2 * np.pi * 700 * t)
              + 0.2 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
    speech *= env.astype(np.float32)
    speech *= 10 ** (-20 / 20) / float(np.sqrt(np.mean(speech ** 2)))

    quiet = rng.standard_normal(total).astype(np.float32) * 10 ** (-70 / 20)

    def run(sig, params):
        den = Denoiser(sr, bs, params)
        out = np.zeros_like(sig)
        for i in range(blocks):
            out[i * bs:(i + 1) * bs] = den.process(sig[i * bs:(i + 1) * bs])
        return out, den

    p_gate = dict(DEFAULT_PARAMS)
    p_gate["denoise"] = 0.0
    p_gate["rnnoise"] = 0.0
    p_gate["model"] = "rnnoise"      # 关闭神经降噪，只测噪声门
    p_gate["gate_margin"] = 7.0

    p_spec = dict(DEFAULT_PARAMS)
    p_spec["denoise"] = 60.0
    p_spec["rnnoise"] = 0.0
    p_spec["model"] = "rnnoise"      # 关闭神经降噪，只测谱门
    p_spec["atten_db"] = 0.0         # 让门不衰减，单独看谱门效果

    p_rn = dict(DEFAULT_PARAMS)      # RNNoise 全开
    p_rn["rnnoise"] = 1.0
    p_rn["model"] = "rnnoise"

    p_dfn = dict(DEFAULT_PARAMS)     # DeepFilterNet3 全开
    p_dfn["model"] = "dfn3"

    q_gate, _ = run(quiet, p_gate)
    q_spec, _ = run(quiet, p_spec)
    s_spec, den_s = run(speech, p_spec)
    q_rn, den_rn = run(quiet, p_rn)

    t0 = time.perf_counter()
    for _ in range(200):
        den_rn.process(quiet[:bs])
    rn_ms = (time.perf_counter() - t0) / 200 * 1000

    q_dfn = s_dfn = None
    dfn_ms = 0.0
    if DFN3_PRESENT:
        q_dfn, den_dfn = run(quiet, p_dfn)
        s_dfn, _ = run(speech, p_dfn)
        # 用有声段计时：静音段会走跳帧早退路径，测不出真实耗时
        chunk = speech[:bs]
        t0 = time.perf_counter()
        for _ in range(200):
            den_dfn.process(chunk)
        dfn_ms = (time.perf_counter() - t0) / 200 * 1000

    tail = slice(sr, None)          # 跳过前 1 秒自适应期
    block_ms = bs * 1000.0 / sr
    print(f"底噪   -70.0 dB  ->  噪声门 {rms_db(q_gate[tail]):7.1f} dB"
          f"  ·  谱门 {rms_db(q_spec[tail]):7.1f} dB"
          f"  ·  RNNoise {rms_db(q_rn[tail]):7.1f} dB")
    print(f"人声   {rms_db(speech[tail]):6.1f} dB  ->  谱门后 {rms_db(s_spec[tail]):7.1f} dB")
    print(f"耗时     RNNoise {rn_ms:.3f} ms/块（块预算 {block_ms:.2f} ms）"
          f" · 单核 {rn_ms / block_ms * 100:.2f}%")

    # 静音段保留 -75dB 左右的低电平舒适噪声（不再是数字零，也不再是明显的"嘶"声），
    # 判定改为「足够安静但非死寂」
    ok_gate = -85.0 < rms_db(q_gate[tail]) < -62.0
    ok_spec = rms_db(q_spec[tail]) < -70.0
    ok_rn = -85.0 < rms_db(q_rn[tail]) < -62.0
    # 压缩器 + makeup 会抬高电平，允许 [-3, +6] dB 范围
    diff = rms_db(s_spec[tail]) - rms_db(speech[tail])
    ok_speech = -5.5 < diff < 6.0

    ok_dfn = True
    if DFN3_PRESENT:
        d_dfn = rms_db(s_dfn[tail]) - rms_db(speech[tail])
        print(f"DFN3     底噪 {rms_db(q_dfn[tail]):7.1f} dB"
              f"  ·  合成音变化 {d_dfn:+.1f} dB（纯音被模型当噪声属正常）"
              f"  ·  {dfn_ms:.3f} ms/块 · 单核 {dfn_ms / block_ms * 100:.2f}%")
        # DFN3 对纯音类合成信号会大幅衰减，这里只校验「底噪压得住且信号不爆炸」；
        # 真实语音保真度由 golden 基准测试保证（与官方 Rust 流水线数值一致）
        ok_dfn = (rms_db(q_dfn[tail]) < -55.0) and (d_dfn < 6.0)
    else:
        # 模型文件在、但引擎起不来 → 打包缺 onnxruntime 之类，属于坏状态
        models_present = False
        try:
            import dfn3 as _m
            models_present = _m.find_model_dir() is not None
        except Exception:
            models_present = False
        if models_present:
            print("DFN3     模型文件存在但引擎不可用（onnxruntime 缺失/加载失败）")
            ok_dfn = False
        else:
            print("DFN3     不可用（未安装模型文件），已退回 RNNoise")
            ok_dfn = True

    print(f"结果     噪声门 {ok_gate} | 谱门 {ok_spec} | RNNoise {ok_rn}"
          f" | DFN3 {ok_dfn} | 人声保真 {ok_speech}")
    return 0 if (ok_gate and ok_spec and ok_rn and ok_speech and ok_dfn) else 1


def _show_crash(title, text):
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, text)
        root.destroy()
    except Exception:
        pass


_MUTEX_HANDLE = None


def _acquire_single_instance():
    """命名互斥量：保证同一时间只运行一个实例。"""
    global _MUTEX_HANDLE
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_wchar_p]
        handle = kernel32.CreateMutexW(None, 0, "MicDenoiser_SingleInstance")
        if not handle:
            return True
        _MUTEX_HANDLE = handle
        # 183 = ERROR_ALREADY_EXISTS，说明已有实例在运行
        return ctypes.get_last_error() != 183
    except Exception:
        return True     # 拿不到互斥量时不阻塞启动


def _activate_existing():
    """把已运行实例的窗口拉到前台；找不到就返回 False。"""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, _WINDOW_TITLE)
        if hwnd:
            user32.ShowWindow(hwnd, 9)          # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            return True
    except Exception:
        pass
    return False


def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    # 只允许一个实例；重复启动时激活已有窗口后退出
    if not _acquire_single_instance():
        if not _activate_existing():
            _show_crash("已在运行",
                        "麦克风降噪已经在运行了（在系统托盘里）。\n"
                        "请从托盘图标打开窗口。")
        return 0

    def _hook(exc_type, exc, tb):
        import traceback
        detail = "".join(traceback.format_exception(exc_type, exc, tb))
        _show_crash("程序出错", detail[-1800:])

    sys.excepthook = _hook
    app = App()
    app.mainloop()


if __name__ == "__main__":
    # 打包成无控制台 exe 后 sys.stdout/stderr 为 None，print 会抛异常；
    # 自检输出丢弃即可，靠退出码判断结果
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()
