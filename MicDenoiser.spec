# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('C:/Users/10998/Desktop/声音降噪软件/models/df3/df3.json', 'models/df3'), ('C:/Users/10998/Desktop/声音降噪软件/models/df3/df_dec.onnx', 'models/df3'), ('C:/Users/10998/Desktop/声音降噪软件/models/df3/erb_dec.onnx', 'models/df3'), ('C:/Users/10998/Desktop/声音降噪软件/models/df3/enc.onnx', 'models/df3')]
binaries = [('C:/Users/10998/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/local-packages/Python313/site-packages/pyrnnoise/rnnoise.dll', 'pyrnnoise')]
hiddenimports = ['pystray._win32', 'onnxruntime']
tmp_ret = collect_all('sounddevice')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:/Users/10998/Desktop/声音降噪软件/mic_denoiser.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pandas', 'IPython', 'PyQt5', 'PySide2', 'pytest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MicDenoiser',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:/Users/10998/Desktop/声音降噪软件/app.ico'],
)
