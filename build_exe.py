#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 mic_denoiser.py 打包成 Windows exe。

用法：
    python build_exe.py            # 单文件 dist/MicDenoiser.exe
    python build_exe.py --onedir   # 目录模式 dist/MicDenoiser/，启动更快

打包前请先装好：pip install -r requirements.txt pyinstaller
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ICON = HERE / "app.ico"


def main():
    try:
        import mic_denoiser
    except Exception as exc:
        print(f"[!] 无法导入主程序：{exc}")
        print("    请先执行：pip install -r requirements.txt")
        return 1

    img = mic_denoiser.make_icon_image(256)
    if img is not None:
        img.save(ICON, format="ICO",
                 sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                        (64, 64), (128, 128), (256, 256)])
        print(f"[+] 图标已生成：{ICON.name}")

    # 定位 RNNoise DLL 并显式打进包里（程序用 importlib 定位，PyInstaller 分析不到）
    rnnoise_dll = None
    try:
        import importlib.util
        spec = importlib.util.find_spec("pyrnnoise")
        if spec is not None and spec.origin:
            dll = Path(spec.origin).parent / "rnnoise.dll"
            if dll.exists():
                rnnoise_dll = dll
    except Exception:
        rnnoise_dll = None
    if rnnoise_dll is None:
        print("[!] 未找到 RNNoise DLL（pyrnnoise/rnnoise.dll），AI 降噪将不可用")
        print("    可先执行：pip install pyrnnoise==0.4.5 -i https://pypi.tuna.tsinghua.edu.cn/simple")

    # DeepFilterNet3：三个 ONNX 子网 + 元数据，打进 models/df3 供运行时定位
    df3_dir = HERE / "models" / "df3"
    df3_files = [df3_dir / n for n in
                 ("enc.onnx", "erb_dec.onnx", "df_dec.onnx", "df3.json")]
    missing = [p.name for p in df3_files if not p.exists()]
    if missing:
        print(f"[!] 缺少 DFN3 模型文件：{missing}（AI 降噪将退回 RNNoise）")
        print("    先执行：.venv-dfn\\Scripts\\python.exe export_dfn3.py")
    onedir = "--onedir" in sys.argv
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onedir" if onedir else "--onefile",
        "--windowed",
        "--name", "MicDenoiser",
        "--icon", str(ICON),
        "--collect-all", "sounddevice",
        "--hidden-import", "pystray._win32",
        "--hidden-import", "onnxruntime",
        "--exclude-module", "matplotlib",
        "--exclude-module", "pandas",
        "--exclude-module", "IPython",
        "--exclude-module", "PyQt5",
        "--exclude-module", "PySide2",
        "--exclude-module", "pytest",
        str(HERE / "mic_denoiser.py"),
    ]
    if rnnoise_dll is not None:
        cmd[3:3] = ["--add-binary", f"{rnnoise_dll};pyrnnoise"]
    if not missing:
        for p in df3_files:
            cmd[3:3] = ["--add-data", f"{p};models/df3"]

    print("[*] 开始打包，首次可能需要几分钟...")
    rc = subprocess.run(cmd, cwd=str(HERE)).returncode
    if rc == 0:
        target = HERE / "dist" / ("MicDenoiser" if onedir else "MicDenoiser.exe")
        print(f"[+] 打包完成：{target}")
    else:
        print("[!] 打包失败，请查看上面的 PyInstaller 日志")
    return rc


if __name__ == "__main__":
    sys.exit(main())
