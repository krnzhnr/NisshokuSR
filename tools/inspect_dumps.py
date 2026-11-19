import numpy as np
from PIL import Image
import os
from pathlib import Path

td = Path(__file__).resolve().parent.parent / "testdata"
# 1) decoder NV12 1920x1080
nv = np.fromfile(os.path.join(td, "dump_dec.nv12"), dtype=np.uint8)
y = nv[:1920*1080].reshape(1080, 1920).astype(np.float32)
uv = nv[1920*1080:1920*1080 + 960*540*2].reshape(540, 960, 2).astype(np.float32)
u = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1)
v = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1)
r = np.clip(y + 1.402 * (v - 128), 0, 255).astype(np.uint8)
g = np.clip(y - 0.344136 * (u - 128) - 0.714136 * (v - 128), 0, 255).astype(np.uint8)
b = np.clip(y + 1.772 * (u - 128), 0, 255).astype(np.uint8)
Image.fromarray(np.stack([r, g, b], -1)[::2, ::2]).save(os.path.join(td, "dbg_dec.png"))
print("dec ok", float(r.mean()), float(g.mean()), float(b.mean()))
# 2) trt in: planar FP16 RGB 3x1080x1920
t = np.fromfile(os.path.join(td, "dump_trtin.f16"), dtype=np.float16).astype(np.float32)
print("trtin flat", t.shape, "minmax", float(t.min()), float(t.max()), "mean", float(t.mean()))
c, h, w = 3, 1080, 1920
t = t[:c * h * w].reshape(c, h, w)
rgb = np.stack([t[0], t[1], t[2]], -1)
Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[::2, ::2]).save(os.path.join(td, "dbg_trtin.png"))
print("trtin ok")
# 3) trt out: planar FP16 RGB 3x2160x3840
o = np.fromfile(os.path.join(td, "dump_trtout.f16"), dtype=np.float16).astype(np.float32)
print("trtout minmax", float(np.nanmin(o)), float(np.nanmax(o)), "mean", float(np.nanmean(o)), "nanfrac", float(np.isnan(o).mean()))
c, h, w = 3, 2160, 3840
o = o[:c * h * w].reshape(c, h, w)
rgb2 = np.stack([o[0], o[1], o[2]], -1)
Image.fromarray((np.clip(np.nan_to_num(rgb2, nan=0.0), 0, 1) * 255).astype(np.uint8)[::4, ::4]).save(os.path.join(td, "dbg_trtout.png"))
print("trtout ok")
