#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fisheye2erp.py — 把 DJI 195° 双鱼眼照片转换为 VR180 等距柱状投影（ERP）。

输入文件名中包含 Snapshot0 / Snapshot1：
    Snapshot0 = 右眼
    Snapshot1 = 左眼
同一张照片的左右眼按文件名末尾的时间戳配对。

用法：
    python fisheye2erp.py <input-dir> <output-dir>

默认输出：
    每个时间戳一张 SBS（左右并排）VR180 JPEG，左右眼各为 180°×180°，
    总宽高比为 2:1。左眼在左，右眼在右。195° 鱼眼边缘多出的约 7.5°
    在标准 VR180 画面之外，默认裁掉；需要保留时用 --full。

常用参数：
    --fov 195             鱼眼完整视场角（默认 195°）
    --size 3840           单眼输出边长；默认取输入图短边
    --layout sbs|separate|both
    --full                输出 360°×180° 全 ERP（背面为黑），默认只输出 VR180
    --right-rot 0.6       右眼绕光轴顺时针旋转角度（默认 0.6°）
    --max-gap 0.25        左右眼时间戳最大允许间隔（秒）
    --projection equidistant|equisolid
    --radius PX           鱼眼有效圆半径（默认 min(w,h)/2）
    --cx PX --cy PX       鱼眼光心（默认图像中心）
    --quality 95          JPEG 质量

只依赖 numpy + Pillow。
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

# 20261124_085949_974157 / 20261124-085949-974157 / 20261124085949
TS_RE = re.compile(
    r"(?<!\d)(20\d{2})[_-]?(\d{2})[_-]?(\d{2})[_-]?"
    r"(\d{2})[_-]?(\d{2})[_-]?(\d{2})(?:[_-](\d{1,6}))?(?!\d)"
)
EYE_RE = re.compile(r"Snapshot\s*([01])", re.IGNORECASE)


@dataclass
class Shot:
    path: Path
    eye: str          # "L" / "R"
    dt: datetime

    @property
    def stamp(self) -> str:
        return self.dt.strftime("%Y%m%d_%H%M%S_") + f"{self.dt.microsecond // 1000:03d}"


def parse_timestamp(name: str) -> datetime | None:
    """取文件名中最后一个时间戳，避免 req/batch 等数字误匹配。"""
    matches = list(TS_RE.finditer(name))
    if not matches:
        return None
    m = matches[-1]
    try:
        # 文件名里的微秒有时省略前导 0（如 _99868 = 099868），需左补零
        micro = int((m.group(7) or "0").rjust(6, "0")[:6])
        return datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)), int(m.group(6)), micro,
        )
    except ValueError:
        return None


def parse_eye(name: str) -> str | None:
    m = EYE_RE.search(name)
    if not m:
        return None
    return "R" if m.group(1) == "0" else "L"


def scan_folder(folder: Path) -> tuple[list[Shot], list[Path]]:
    shots: list[Shot] = []
    ignored: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        eye = parse_eye(p.name)
        dt = parse_timestamp(p.name)
        if eye is None or dt is None:
            ignored.append(p)
            continue
        shots.append(Shot(path=p, eye=eye, dt=dt))
    return shots, ignored


def pair_shots(shots: list[Shot], max_gap_s: float) -> tuple[list[tuple[Shot, Shot]], list[Shot]]:
    """按时间戳做全局最近邻一对一配对，避免 req 编号不可靠。"""
    lefts = [s for s in shots if s.eye == "L"]
    rights = [s for s in shots if s.eye == "R"]
    candidates: list[tuple[float, int, int]] = []
    for i, left in enumerate(lefts):
        for j, right in enumerate(rights):
            gap = abs((left.dt - right.dt).total_seconds())
            if gap <= max_gap_s:
                candidates.append((gap, i, j))
    candidates.sort()

    used_l: set[int] = set()
    used_r: set[int] = set()
    pairs: list[tuple[Shot, Shot]] = []
    for _, i, j in candidates:
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        pairs.append((lefts[i], rights[j]))
    pairs.sort(key=lambda p: p[0].dt)

    unpaired = [s for i, s in enumerate(lefts) if i not in used_l]
    unpaired += [s for j, s in enumerate(rights) if j not in used_r]
    unpaired.sort(key=lambda s: s.dt)
    return pairs, unpaired


def rotate_clockwise(img: Image.Image, deg: float) -> Image.Image:
    """绕图像中心顺时针旋转；Pillow 的正角度是逆时针。"""
    if abs(deg) < 1e-9:
        return img
    return img.rotate(
        -deg,
        resample=Image.Resampling.BICUBIC,
        center=((img.width - 1) / 2.0, (img.height - 1) / 2.0),
        fillcolor=(0, 0, 0),
    )


def bilinear_remap(src: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """在 float32 坐标上做双线性重采样，越界填黑。"""
    h, w = src.shape[:2]
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    fx = (map_x - x0)[..., None]
    fy = (map_y - y0)[..., None]
    valid = (map_x >= 0) & (map_x < w - 1) & (map_y >= 0) & (map_y < h - 1)

    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)

    # 用 uint8 索引取值后转 float32，避免整图 float 转换占内存
    p00 = src[y0c, x0c].astype(np.float32)
    p10 = src[y0c, x0c + 1].astype(np.float32)
    p01 = src[y0c + 1, x0c].astype(np.float32)
    p11 = src[y0c + 1, x0c + 1].astype(np.float32)

    out = (
        p00 * ((1 - fx) * (1 - fy))
        + p10 * (fx * (1 - fy))
        + p01 * ((1 - fx) * fy)
        + p11 * (fx * fy)
    )
    out[~valid] = 0.0
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def erp_unwarp(
    img: Image.Image,
    out_w: int,
    out_h: int,
    fov_deg: float,
    radius: float,
    cx: float,
    cy: float,
    projection: str,
    full: bool,
    chunk_rows: int = 256,
) -> np.ndarray:
    """
    将等距（或等立体角）鱼眼映射到 ERP。

    坐标约定：
        lon=0, lat=0 是相机光轴；x 输出向右为 lon 正，y 输出向下为 lat 负。
        鱼眼图像 +x 向右、+y 向下，因此 map_y = cy - r*sin(phi)。
    """
    src = np.asarray(img, dtype=np.uint8)
    if src.ndim == 2:
        src = np.repeat(src[..., None], 3, axis=2)
    src = src[..., :3]

    half_fov = math.radians(fov_deg) / 2.0
    lon_span = 2.0 * math.pi if full else math.pi
    xs = ((np.arange(out_w, dtype=np.float32) + 0.5) / out_w - 0.5) * lon_span
    out = np.empty((out_h, out_w, 3), dtype=np.uint8)

    for y0 in range(0, out_h, chunk_rows):
        y1 = min(y0 + chunk_rows, out_h)
        ys = (np.arange(y0, y1, dtype=np.float32) + 0.5) / out_h
        lat = (0.5 - ys)[:, None] * math.pi
        lon = xs[None, :]

        cos_lat = np.cos(lat)
        X = cos_lat * np.sin(lon)
        Y = np.sin(lat) + np.zeros_like(lon)
        Z = cos_lat * np.cos(lon)

        theta = np.arccos(np.clip(Z, -1.0, 1.0))
        if projection == "equisolid":
            r = radius * np.sin(theta / 2.0) / math.sin(half_fov / 2.0)
        else:  # equidistant
            r = radius * theta / half_fov

        phi = np.arctan2(Y, X)
        map_x = cx + r * np.cos(phi)
        map_y = cy - r * np.sin(phi)

        # 鱼眼圆外（尤其是 360° 模式下的背面）填黑
        outside = theta > half_fov + 1e-7
        map_x[outside] = -1.0
        map_y[outside] = -1.0

        out[y0:y1] = bilinear_remap(src, map_x.astype(np.float32), map_y.astype(np.float32))

    return out


def unique_path(folder: Path, stem: str, suffix: str) -> Path:
    p = folder / f"{stem}{suffix}"
    if not p.exists():
        return p
    n = 2
    while True:
        p = folder / f"{stem}_{n}{suffix}"
        if not p.exists():
            return p
        n += 1


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def convert_pair(
    left: Shot,
    right: Shot,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    img_l = load_rgb(left.path)
    img_r = load_rgb(right.path)
    if img_l.size != img_r.size:
        raise ValueError(f"左右眼分辨率不一致: L={img_l.size}, R={img_r.size}")
    w, h = img_l.size

    size = args.size or min(w, h)
    out_w, out_h = (size * 2, size) if args.full else (size, size)
    radius = args.radius if args.radius else min(w, h) / 2.0
    cx = args.cx if args.cx is not None else (w - 1) / 2.0
    cy = args.cy if args.cy is not None else (h - 1) / 2.0

    print(
        f"  输入 {w}x{h} | 输出 {'360x180' if args.full else '180x180'} "
        f"@ {out_w}x{out_h}/眼 | FOV={args.fov:g}° | 右眼旋转={args.right_rot:g}°"
    )

    img_r = rotate_clockwise(img_r, args.right_rot)

    erp_l = erp_unwarp(
        img_l, out_w, out_h, args.fov, radius, cx, cy,
        args.projection, args.full,
    )
    erp_r = erp_unwarp(
        img_r, out_w, out_h, args.fov, radius, cx, cy,
        args.projection, args.full,
    )

    written: list[Path] = []
    stamp = left.stamp

    if args.layout in ("separate", "both"):
        for tag, arr in (("L", erp_l), ("R", erp_r)):
            p = unique_path(out_dir, f"{stamp}_{tag}_ERP", ".jpg")
            Image.fromarray(arr).save(
                p, format="JPEG", quality=args.quality, subsampling=0, optimize=True
            )
            written.append(p)

    if args.layout in ("sbs", "both"):
        sbs = np.concatenate([erp_l, erp_r], axis=1)
        p = unique_path(out_dir, f"{stamp}_ERP_SBS", ".jpg")
        Image.fromarray(sbs).save(
            p, format="JPEG", quality=args.quality, subsampling=0, optimize=True
        )
        written.append(p)

    return written


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="将 Snapshot0/Snapshot1 195° 双鱼眼照片转换为 VR180 等距柱状投影（ERP）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input_dir", help="输入照片目录（会递归扫描）")
    ap.add_argument("output_dir", help="输出目录")
    ap.add_argument("--fov", type=float, default=195.0, help="鱼眼完整视场角（度）")
    ap.add_argument("--size", type=int, default=0, help="单眼输出边长；0=取输入图短边")
    ap.add_argument("--layout", choices=["sbs", "separate", "both"], default="sbs",
                    help="sbs=左右并排一张；separate=左右眼各一张；both=全部输出")
    ap.add_argument("--full", action="store_true",
                    help="输出 360°×180° 全 ERP（鱼眼未覆盖的背面为黑色），默认 180°×180°")
    ap.add_argument("--projection", choices=["equidistant", "equisolid"], default="equidistant",
                    help="鱼眼投影模型")
    ap.add_argument("--radius", type=float, default=0.0,
                    help="鱼眼有效圆半径（像素）；0=min(宽,高)/2")
    ap.add_argument("--cx", type=float, default=None, help="鱼眼光心 x（像素）")
    ap.add_argument("--cy", type=float, default=None, help="鱼眼光心 y（像素）")
    ap.add_argument("--right-rot", type=float, default=0.6,
                    help="右眼（Snapshot0）绕光轴顺时针旋转角，度")
    ap.add_argument("--max-gap", type=float, default=0.25,
                    help="同一张照片左右眼时间戳最大间隔（秒）")
    ap.add_argument("--quality", type=int, default=95, help="JPEG 质量 1-100")
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = build_argparser().parse_args(argv)
    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if not in_dir.is_dir():
        print(f"[错误] 输入目录不存在: {in_dir}", file=sys.stderr)
        return 2
    if args.size < 0:
        print("[错误] --size 必须 >= 0", file=sys.stderr)
        return 2
    if not (0 < args.fov <= 360):
        print("[错误] --fov 必须在 (0, 360] 范围内", file=sys.stderr)
        return 2
    if args.radius < 0:
        print("[错误] --radius 必须 >= 0", file=sys.stderr)
        return 2
    if args.max_gap < 0:
        print("[错误] --max-gap 必须 >= 0", file=sys.stderr)
        return 2
    if not (1 <= args.quality <= 100):
        print("[错误] --quality 必须在 1..100", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    shots, ignored = scan_folder(in_dir)
    lefts = sum(s.eye == "L" for s in shots)
    rights = sum(s.eye == "R" for s in shots)
    print(f"扫描: {in_dir}")
    print(f"  识别到左眼 Snapshot1: {lefts} 张，右眼 Snapshot0: {rights} 张")
    if ignored:
        print(f"  忽略 {len(ignored)} 个非 Snapshot 文件")

    pairs, unpaired = pair_shots(shots, args.max_gap)
    print(f"  按时间戳配对成功: {len(pairs)} 对（容差 {args.max_gap:g}s）")
    if unpaired:
        print(f"  [警告] {len(unpaired)} 张未配对：")
        for s in unpaired[:10]:
            print(f"    {s.eye} {s.dt.isoformat(timespec='milliseconds')} {s.path.name}")
        if len(unpaired) > 10:
            print(f"    ... 另有 {len(unpaired) - 10} 张")

    if not pairs:
        print("[错误] 没有找到可转换的左右眼照片。", file=sys.stderr)
        return 1

    ok = 0
    failed = 0
    for idx, (left, right) in enumerate(pairs, 1):
        gap_ms = (right.dt - left.dt).total_seconds() * 1000.0
        print(f"\n[{idx}/{len(pairs)}] L={left.dt.isoformat(timespec='milliseconds')} "
              f"R={right.dt.isoformat(timespec='milliseconds')} Δ={gap_ms:+.1f}ms")
        print(f"  L: {left.path.name}")
        print(f"  R: {right.path.name}")
        try:
            files = convert_pair(left, right, out_dir, args)
            for p in files:
                print(f"  -> {p}")
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  [错误] {exc}", file=sys.stderr)

    print(f"\n完成: 成功 {ok} 对，失败 {failed} 对，输出目录: {out_dir}")
    return 0 if ok and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
