# Экспорт апскейл-модели в ONNX со static shape (NCHW, FP32; в FP16 кастит TRT).
# --arch bilinear: детерминированный билинейный апскейл (честный бейзлайн:
#   картинка корректная, скорость максимальная). Свои веса (Real-ESRGAN и т.п.)
#   подставляются через --arch tiny --ckpt model.pth.
import argparse
import torch
import torch.nn as nn


class BilinearUpscaler(nn.Module):
    """Детерминированный бейзлайн: билинейный x2 без обучаемых весов.
    Дает корректное видео и верхний предел пропускной способности тракта."""

    def __init__(self, scale=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)

    def forward(self, x):
        return self.up(x)


class TinyUpscaler(nn.Module):
    """Легкая ESPCN-подобная сеть. Без --ckpt веса случайные (выход = шум!) —
    только для проверки формы графа. Боевые веса: --ckpt model.pth."""

    def __init__(self, scale=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 3 * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        return self.body(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="bilinear", choices=["bilinear", "tiny"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"],
                    help="fp16: нативный FP16-граф (нужен для --stronglyTyped + fp16 IO)")
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--scale", type=int, default=2)
    a = ap.parse_args()
    if a.arch == "bilinear":
        m = BilinearUpscaler(a.scale).eval()
        if a.ckpt:
            raise SystemExit("--ckpt только для --arch tiny")
    else:
        m = TinyUpscaler(a.scale).eval()
        if a.ckpt:
            m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
        else:
            print("WARNING: случайные веса — выход будет шумом, только для проверки графа")
    if a.dtype == "fp16":
        m = m.half()
        x = torch.randn(1, 3, a.h, a.w, dtype=torch.float16)
    else:
        x = torch.randn(1, 3, a.h, a.w)
    torch.onnx.export(m, x, a.out, input_names=["input"], output_names=["output"],
                      opset_version=17, do_constant_folding=True)
    print("saved", a.out)


if __name__ == "__main__":
    main()
