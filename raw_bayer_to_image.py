#!/usr/bin/env python3
"""
RAW 拜耳 -> 彩色成品图

两个独立阶段：
  ① 解拜耳 Demosaic = 数学插值，把每个像素只有一种颜色的拜耳数据还原成 RGB
  ② 调色 Color grading = 白平衡 / 高光压缩 / 对比 / 饱和（审美判断，无标准答案）

解拜耳用「方向自适应绿 + 色差插值」的浮点实现（不再用 cv2.cvtColor 的双线性）：
  双线性只做 3x3 平均，边缘上 R/B 与 G 会错开约 1 像素，于是出现彩色拉链/锯齿，
  且有效分辨率下降。方向自适应会沿着边缘插值，色差插值再保证 R-G/B-G 平滑。

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
  --wb-gain  直接固定白平衡增益 R,G,B（覆盖 --neutral/--warm）。一批片子同灯同场景时
             用它，否则 --neutral 的矩形会落到不同物体上，整批颜色忽黄忽蓝
  --denoise  色度降噪（0=关，3=轻，5=强）。只动 R-G/B-G，不动亮度，默认 5
  --out-bit  输出位深 8 或 16。16 保留完整动态范围（文件约 3 倍大），默认 8
  --resize   输出前把长边缩到 N（INTER_AREA）。别让看图软件去缩 3840 的图，
             廉价缩放会在硬边上产生彩色锯齿

命名注意：
  BAYER_CODES 里的「标签 -> OpenCV 常量」是现场实测标定的，与 OpenCV 官方文档的
  Bayer 命名正好相反（官方 BayerBG 指 RGGB，这里 RGGB 用 BayerRG）。
  换机器/换传感器时若颜色 R/B 互换，先怀疑这里，而不是调 --warm。
"""

import argparse
import time

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


# ----------------------------------------------------------------------------
# 解拜耳
# ----------------------------------------------------------------------------
def _cfa_layout(label):
    """标签 -> 直觉 2x2 布局（'GBRG' = 第0行 G,B / 第1行 R,G）。

    BAYER_CODES 的标签与 OpenCV 官方命名相反，所以这里把 R/B 互换一次，
    得到真正的传感器排布。已用合成图对 4 种排列逐一验证。
    """
    return label.upper().translate(str.maketrans("RB", "BR"))


def _channel_masks(cfa, h, w):
    """三个通道各自的采样点掩码（1 = 该像素有这个颜色）。"""
    m = np.zeros((3, h, w), np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            c = "RGB".index(cfa[dy * 2 + dx])
            m[c, dy::2, dx::2] = 1.0
    return m


def _gain_map(shape, cfa, gains):
    """每个像素按其通道取对应增益。"""
    h, w = shape
    gm = np.empty((h, w), np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            gm[dy::2, dx::2] = gains["RGB".index(cfa[dy * 2 + dx])]
    return gm


def _shift(a, dy, dx, pad=2):
    """带边缘复制的平移（避免 np.roll 的环绕）。"""
    return a[pad + dy:pad + dy + a.shape[0] - 2 * pad,
             pad + dx:pad + dx + a.shape[1] - 2 * pad]


def _demosaic_float(mos, cfa):
    """方向自适应绿(Hamilton-Adams) + 色差插值，输入/输出均为浮点。"""
    h, w = mos.shape
    pad = 2
    mp = np.pad(mos, pad, mode="edge")
    masks = _channel_masks(cfa, h, w)
    mG = masks[1]
    Gp = np.pad(mos * mG, pad, mode="edge")
    S = lambda a, dy, dx: _shift(a, dy, dx, pad)

    C = mp[pad:pad + h, pad:pad + w]
    gl, gr = S(Gp, 0, -1), S(Gp, 0, 1)
    gu, gd = S(Gp, -1, 0), S(Gp, 1, 0)
    h2 = S(mp, 0, -2) + S(mp, 0, 2)
    v2 = S(mp, -2, 0) + S(mp, 2, 0)

    # 梯度小的一侧更可信；修正项用同色 ±2 的二阶差分，天然无直流偏移
    dH = np.abs(gl - gr) + np.abs(2 * C - h2)
    dV = np.abs(gu - gd) + np.abs(2 * C - v2)
    Gh = 0.5 * (gl + gr) + 0.25 * (2 * C - h2)
    Gv = 0.5 * (gu + gd) + 0.25 * (2 * C - v2)
    G = np.where(mG > 0, C, np.where(dH <= dV, Gh, Gv))

    # 色差 (R-G)/(B-G) 平滑插值：等色相假设，边缘不会出现彩色拉链
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16

    def nconv(sparse, mask):
        den = cv2.filter2D(mask, -1, k)
        return cv2.filter2D(sparse * mask, -1, k) / np.maximum(den, 1e-6)

    Rc = nconv(np.where(masks[0] > 0, mos - G, 0), masks[0])
    Bc = nconv(np.where(masks[2] > 0, mos - G, 0), masks[2])
    R = np.where(masks[0] > 0, mos, G + Rc)
    B = np.where(masks[2] > 0, mos, G + Bc)
    return np.clip(np.dstack([R, G, B]), 0, None).astype(np.float32)


def _denoise_chroma(rgb, ksize):
    """只对色差做中值滤波：彩色噪点消失，亮度细节原样保留。"""
    if ksize < 3:
        return rgb
    g = rgb[:, :, 1]
    rg = cv2.medianBlur(rgb[:, :, 0] - g, ksize)
    bg = cv2.medianBlur(rgb[:, :, 2] - g, ksize)
    return np.dstack([g + rg, g, g + bg])


def demosaic(linear, bayer, gains=None, denoise=5):
    """① 解拜耳：把马赛克插值成每个像素的完整 RGB（浮点，保持 16bit 精度）。

    gains 给出时先做白平衡再插值：三个通道量级一致，方向判断和色差插值才准。
    返回的 rgb 已乘好增益（未做 gamma/白点）。
    """
    if gains is None:  # 兼容旧调用：退回双线性
        return cv2.cvtColor(np.clip(linear, 0, 65535).astype(np.uint16),
                            BAYER_CODES[bayer.upper()]).astype(np.float32)
    cfa = _cfa_layout(bayer)
    gains = np.asarray(gains, np.float32)
    rgb = _demosaic_float(linear * _gain_map(linear.shape, cfa, gains), cfa)
    return _denoise_chroma(rgb, denoise)


# ----------------------------------------------------------------------------
# 调色
# ----------------------------------------------------------------------------
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


def grade(rgb, white, shoulder, contrast, sat, vibrance, gamma=2.2):
    """②b 调色：高光软肩 -> gamma -> S 曲线对比 -> 饱和 + vibrance。返回 0~1 浮点。"""
    x = np.clip(rgb / white, 0, None)
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
    return np.clip(out, 0, 1).astype(np.float32)


def save_png(rgb01, path, bit=8):
    """8bit 四舍五入（不是截断）；16bit 用 OpenCV 编码后写盘，避开中文路径。"""
    if bit == 16:
        u16 = (np.clip(rgb01, 0, 1) * 65535.0 + 0.5).astype(np.uint16)
        ok, buf = cv2.imencode(".png", u16[:, :, ::-1])
        if not ok:
            raise IOError("PNG 编码失败")
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    else:
        Image.fromarray((np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8)).save(path)


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
    ap.add_argument("--denoise", type=int, default=5, choices=[0, 3, 5],
                    help="色度降噪 0/3/5：越大彩色噪点越少，默认 5")
    ap.add_argument("--out-bit", type=int, default=8, choices=[8, 16],
                    help="输出位深：8 通用；16 保留完整动态范围，默认 8")
    ap.add_argument("--resize", type=int, default=0, metavar="N",
                    help="输出前把长边缩到 N 像素（INTER_AREA，自带抗锯齿）。"
                         "看图时别让看图软件去缩 3840 的图，那样会出彩边/锯齿")
    ap.add_argument("--neutral", action="append", default=[], metavar="X0,Y0,X1,Y1",
                    help="中性区矩形，可指定多次；不指定则用亮部灰世界")
    ap.add_argument("--wb-gain", default=None, metavar="R,G,B",
                    help="直接给定白平衡增益（覆盖 --neutral / --warm）。"
                         "整批同一场景、同一盏灯时用它：批内每张的 WB 才一致，"
                         "否则 --neutral 落在不同物体上，整批颜色会忽黄忽蓝")
    a = ap.parse_args()

    patches = []
    for spec in a.neutral:
        x0, y0, x1, y1 = map(int, spec.split(","))
        patches.append((x0, y0, x1, y1))

    t0 = time.time()
    raw = load_raw(a.input, a.w, a.h, a.bit, a.stride)
    linear = np.clip(raw - a.black, 0, None)

    # 白平衡增益与白点沿用原来的双线性结果来估计：保证换解拜耳前后
    # 画面的整体亮度和色温跟旧版一致，不用重新调参。
    ref = cv2.cvtColor(np.clip(linear, 0, 65535).astype(np.uint16),
                       BAYER_CODES[a.bayer.upper()]).astype(np.float32)
    white = a.white
    if a.auto_white is not None:
        white = float(np.percentile(ref.max(2), a.auto_white))
    if a.ae is not None:
        white = float(a.ae) * float(np.median(ref.mean(2)))
    if a.wb_gain:
        gains = np.array([float(v) for v in a.wb_gain.split(",")], np.float32)
        if gains.size != 3 or not np.all(gains > 0):
            raise SystemExit("--wb-gain 需要 3 个正数，例如 --wb-gain 1.53,0.62,0.85")
        gains = gains / gains.mean()
    else:
        gains = wb_gains(ref, patches, a.warm)
    del ref

    rgb = demosaic(linear, a.bayer, gains, a.denoise)
    result = grade(rgb, white, a.shoulder, a.contrast, a.sat, a.vibrance, a.gamma)
    if a.resize and a.resize < max(result.shape[:2]):
        h, w = result.shape[:2]
        s = a.resize / float(max(h, w))
        result = cv2.resize(result, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                            interpolation=cv2.INTER_AREA)
    save_png(result, a.output, a.out_bit)

    blown = float((result >= 0.999).all(2).mean())
    print(f"拜耳={a.bayer.upper()} 黑电平={a.black} 白点={white:.0f}"
          + (f" (自动 p{a.auto_white})" if a.auto_white is not None else ""))
    print(f"白平衡增益 RGB={gains.round(3)}  色度降噪={a.denoise}  输出={a.out_bit}bit")
    if blown > 0.02:
        print(f"[提示] {blown:.1%} 的像素已压成纯白（高光细节丢失）；"
              f"可把 --ae 调大或 --white 调大")
    print(f"已保存 {a.output}  {result.shape[1]}x{result.shape[0]}  耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
