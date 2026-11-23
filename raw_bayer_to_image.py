#!/usr/bin/env python3
"""
RAW 拜耳 -> 彩色成品图

两个独立阶段：
  ① 解拜耳 Demosaic = 数学插值，把每个像素只有一种颜色的拜耳数据还原成 RGB（答案唯一）
  ② 调色 Color grading = 白平衡 / 高光压缩 / 对比 / 饱和（审美判断，无标准答案）

依赖：
  pip install numpy opencv-python pillow

用法：
  python raw_bayer_to_image.py input.raw -o out.png --w 3840 --h 3840 --bayer GBRG

  针对怼着屏幕拍的那种图，指定中性区（白色桌面/墙）做白平衡更准：
  python raw_bayer_to_image.py input.raw -o out.png \
      --neutral 1000,2600,2800,3300 --neutral 2600,300,3400,1100

参数（--help 查看全部）：
  --black   黑电平，先从原始值里减掉（本例 60）
  --white   白点，映射到最亮（本例 2000；越大越暗）
  --ae     自动曝光：白点 = K × 全图线性中位数（K=8 适配本机 3840 鱼眼）。
           比 --auto-white 99.9 稳：画面里有太阳那种小块高光时不会把整张压暗。
  --shoulder 高光软肩，防止亮部死白（0~0.5）
  --contrast S 曲线对比强度（0~1）
  --sat      饱和度（1 = 原样，2 = 加倍）
  --vibrance 智能饱和：只增强有颜色的地方，灰色保持干净（0~1）
  --warm     色温，>1 偏暖，<1 偏冷（本例 1.04）
"""

import argparse
import numpy as np
import cv2
from PIL import Image

BAYER_CODES = {
    "RGGB": cv2.COLOR_BayerRG2RGB,
    "BGGR": cv2.COLOR_BayerBG2RGB,
    "GBRG": cv2.COLOR_BayerGB2RGB,
    "GRBG": cv2.COLOR_BayerGR2RGB,
}


def load_raw(path, w, h, bit, stride=None):
    """读裸 RAW。16-bit 小端；stride 为空则按 w*bytes 连续排列。"""
    dtype = np.uint16 if bit == 16 else np.uint8
    bpp = 2 if bit == 16 else 1
    row = stride if stride else w * bpp
    with open(path, "rb") as f:
        buf = f.read()
    arr = np.frombuffer(buf, dtype=dtype)
    if row == w * bpp:
        a = arr[: w * h].reshape(h, w)
    else:
        pixels_per_row = row // bpp
        a = arr[: pixels_per_row * h].reshape(h, pixels_per_row)[:, :w]
    return a.astype(np.float32)


def demosaic(linear, bayer):
    """① 解拜耳：把马赛克插值成每个像素的完整 RGB。"""
    return cv2.cvtColor(np.clip(linear, 0, 65535).astype(np.uint16),
                        BAYER_CODES[bayer.upper()]).astype(np.float32)


def wb_gains(rgb, patches, warm):
    """②a 白平衡：用中性区（应呈白/灰的面）反推三通道增益。"""
    if patches:
        s = np.vstack([rgb[y0:y1, x0:x1].reshape(-1, 3)
                       for x0, y0, x1, y1 in patches])
        s = s[s.mean(1) > np.percentile(s.mean(1), 40)]
        neutral = s.mean(0)
    else:
        lum = rgb.mean(2)
        s = rgb[lum > np.percentile(lum, 80)]
        neutral = s.mean(0)
    gains = neutral.mean() / np.maximum(neutral, 1)
    gains = gains * np.array([warm, 1.0, 1.0 / warm])
    return gains / gains.mean()


def grade(rgb, white, shoulder, contrast, sat, vibrance, gains, gamma=2.2):
    """②b 调色：高光软肩 -> gamma -> S 曲线对比 -> 饱和 + vibrance。"""
    c = rgb * gains[None, None, :]
    x = np.clip(c / white, 0, None)
    # 关键：white 直接映射到 1.0。切勿再除以全图 max，否则整体被压暗压灰。
    x = x * (1 + shoulder) / (1 + x * shoulder)
    x = np.clip(x, 0, 1)
    g = np.power(x, 1 / gamma)
    ss = g * g * (3 - 2 * g)
    g = np.clip(g + (ss - g) * contrast, 0, 1)

    l = (0.2126 * g[:, :, 0] + 0.7152 * g[:, :, 1]
         + 0.0722 * g[:, :, 2])[:, :, None]
    out = l + (g - l) * sat
    mx, mn = out.max(2), out.min(2)
    chroma = (mx - mn) / (mx + 1e-6)
    out = l + (out - l) * (1 + vibrance * (1 - chroma))[:, :, None]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="RAW 拜耳解码 + 调色")
    ap.add_argument("input", help="裸 RAW 文件路径")
    ap.add_argument("-o", "--output", default="out.png", help="输出 PNG")
    ap.add_argument("--w", type=int, required=True, help="图像宽度")
    ap.add_argument("--h", type=int, required=True, help="图像高度")
    ap.add_argument("--bit", type=int, default=16, choices=[8, 16], help="位深，默认 16")
    ap.add_argument("--stride", type=int, default=None, help="每行字节数，默认 w*字节数")
    ap.add_argument("--bayer", default="GBRG", choices=list(BAYER_CODES), help="拜耳排列")
    ap.add_argument("--black", type=float, default=60, help="黑电平")
    ap.add_argument("--white", type=float, default=2000, help="白点")
    ap.add_argument("--auto-white", type=float, default=None, metavar="PCT",
                    help="按本帧该百分位自动定白点（如 99.9），优先于 --white")
    ap.add_argument("--ae", type=float, default=None, metavar="K",
                    help="自动曝光：白点 = K × 全图线性中位数（本机 3840 鱼眼建议 8）。"
                         "比 --auto-white 稳：画面里有太阳这种小块高光时不会把整张压暗。"
                         "优先于 --white / --auto-white")
    ap.add_argument("--shoulder", type=float, default=0.20, help="高光软肩 0~0.5")
    ap.add_argument("--contrast", type=float, default=0.75, help="对比 0~1")
    ap.add_argument("--sat", type=float, default=2.0, help="饱和度")
    ap.add_argument("--vibrance", type=float, default=1.0, help="智能饱和 0~1")
    ap.add_argument("--gamma", type=float, default=2.2, help="gamma，越小整体越亮（不剪高光）")
    ap.add_argument("--warm", type=float, default=1.04, help="色温 >1 暖 <1 冷")
    ap.add_argument("--neutral", action="append", default=[], metavar="X0,Y0,X1,Y1",
                    help="中性区矩形，可指定多次；不指定则用亮部灰世界")
    a = ap.parse_args()

    patches = []
    for spec in a.neutral:
        x0, y0, x1, y1 = map(int, spec.split(","))
        patches.append((x0, y0, x1, y1))

    raw = load_raw(a.input, a.w, a.h, a.bit, a.stride)
    linear = np.clip(raw - a.black, 0, None)
    rgb = demosaic(linear, a.bayer)
    white = a.white
    if a.auto_white is not None:
        white = float(np.percentile(rgb.max(2), a.auto_white))
    if a.ae is not None:
        white = float(a.ae) * float(np.median(rgb.mean(2)))
    gains = wb_gains(rgb, patches, a.warm)
    result = grade(rgb, white, a.shoulder, a.contrast,
                   a.sat, a.vibrance, a.warm and gains, a.gamma)
    Image.fromarray(result).save(a.output)

    print(f"拜耳={a.bayer.upper()} 黑电平={a.black} 白点={white}"
          + (f" (自动 p{a.auto_white})" if a.auto_white is not None else ""))
    print(f"白平衡增益 RGB={gains.round(3)}")
    print(f"已保存 {a.output}  {a.w}x{a.h}")


if __name__ == "__main__":
    main()
