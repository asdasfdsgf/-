# -*- coding: utf-8 -*-
"""模型选择卡片的功能测试：切换、持久化、运行中异步切换。"""
import json
import os
import tempfile
import time

import mic_denoiser as md

# 用临时配置，别碰正式 config.json
md.CONFIG_PATH = os.path.join(tempfile.mkdtemp(), "config.json")

app = md.App()
app.withdraw()
app.update_idletasks()
print("默认模型:", app.params["model"], "| DFN3 可用:", md.DFN3_PRESENT)
print("卡片:", list(app.model_cards))
print("窗口需求尺寸:", app.winfo_reqwidth(), "x", app.winfo_reqheight(),
      "| 实际:", app.winfo_width(), "x", app.winfo_height())

app._select_model("rnnoise")
app.update_idletasks()
print("切到 rnnoise:", app.params["model"])
app._save_config()
print("配置已存:", json.load(open(md.CONFIG_PATH, encoding="utf-8")).get("model"))

cfg = json.load(open(md.CONFIG_PATH, encoding="utf-8"))
cfg["model"] = "dfn3"
with open(md.CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(cfg, f)
app._load_config()
print("重新加载配置后:", app.params["model"])

# 模拟运行中切换（有 stream + 引擎尚未创建 → 后台线程建引擎）
app.params["model"] = "rnnoise"
app._paint_all_model_cards()
app.denoiser = md.Denoiser(48000, 480, app.params)
app.stream = object()
app._select_model("dfn3")
t0 = time.time()
while time.time() - t0 < 5.0:
    app._drain_commands()
    app.update()
    if app.params["model"] == "dfn3":
        break
    time.sleep(0.03)
print("运行中切换:", app.params["model"],
      "| 引擎就绪:", app.denoiser._dfn is not None,
      f"| 耗时 {time.time()-t0:.2f}s")

app._quit()
print("OK")
