# -*- coding: utf-8 -*-
"""启动程序并截图窗口，用于检查模型卡片的视觉排版。"""
import ctypes
import subprocess
import sys
import time
from ctypes import wintypes

from PIL import ImageGrab

TITLE = "麦克风降噪 · Mic Denoiser"
user32 = ctypes.windll.user32


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "mic_denoiser.py"
    if target.endswith(".py"):
        p = subprocess.Popen([sys.executable, target])
    else:
        p = subprocess.Popen([target])
    hwnd = 0
    for _ in range(60):
        time.sleep(0.25)
        hwnd = user32.FindWindowW(None, TITLE)
        if hwnd:
            break
    if not hwnd:
        print("窗口未找到")
        p.terminate()
        return 1
    time.sleep(1.2)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.4)
    r = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    img = ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom))
    img.save("_ui.png")
    print("saved _ui.png", img.size)
    p.terminate()
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
