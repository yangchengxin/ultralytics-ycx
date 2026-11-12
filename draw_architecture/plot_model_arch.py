#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Plot an Ultralytics model architecture from its model YAML.

Reads the ``backbone`` and ``head`` sections of a model YAML, resolves every layer's
``from`` field (including multi-input layers such as Concat/Detect/Pose), lays the
model out in three staged columns (Backbone on the left, Neck in the middle, output
Head on the right) and renders a PNG. Changing the YAML and re-running the script
regenerates the diagram, so the YAML stays the single source of truth.

Examples:
    >>> python plot_model_arch.py ultralytics/cfg/models/26/yolo26-pose.yaml
    >>> python plot_model_arch.py yolo26.yaml -o arch.png --imgsz 640
    >>> python plot_model_arch.py yolo26.yaml --no-head-detail --title "YOLO26 backbone"
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- palette
INK = "#202124"
SUB = "#5f6368"
ARROW = "#5f6368"
SKIP = "#1a73e8"
BG = "#ffffff"
WHITE = "#ffffff"
OUT_F, OUT_S = "#fce8e6", "#c5221f"
MERGE_F, MERGE_S = "#f1f3f4", "#5f6368"
CONV_F, CONV_S = "#e8f0fe", "#1a73e8"
CSP_F, CSP_S = "#e6f4ea", "#188038"
SPP_F, SPP_S = "#e0f2f1", "#00695c"
ATT_F, ATT_S = "#f3e8fd", "#8430ce"
HEAD_F, HEAD_S = "#fef7e0", "#e37400"
TR_F, TR_S = "#fafafa", "#9aa0a6"
PANEL_F = "#fcfcfe"

HEAD_KEYS = ("detect", "pose", "segment", "semanticsegment", "obb", "classif", "decoder", "depth", "head")


# --------------------------------------------------------------------------- fonts
_FONT_OVERRIDE = None


def _font_candidates(bold):
    if _FONT_OVERRIDE:
        return [_FONT_OVERRIDE]
    weight = "Bold" if bold else "Regular"
    candidates = [
        os.path.expanduser("~/.config/Ultralytics/Arial.Unicode.ttf"),
        "/root/.config/Ultralytics/Arial.Unicode.ttf",
        f"/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc",
        f"/usr/share/fonts/truetype/noto/NotoSansCJK-{weight}.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
    ]
    # matplotlib ships a scalable DejaVu Sans even in slim containers
    spec = importlib.util.find_spec("matplotlib")
    if spec and spec.submodule_search_locations:
        ttf = Path(list(spec.submodule_search_locations)[0]) / "mpl-data/fonts/ttf"
        candidates.append(str(ttf / f"DejaVuSans{'-Bold' if bold else ''}.ttf"))
    return candidates


def _find_font(bold=False):
    for path in _font_candidates(bold):
        try:
            if Path(path).exists():
                return path
        except OSError:  # e.g. /root is not readable for non-root users
            continue
    return None


class Painter:
    """Tiny supersampled canvas with the drawing primitives used by the diagram."""

    def __init__(self, width, height, supersample=2):
        self.ss = max(1, int(supersample))
        self.w, self.h = width, height
        self.img = Image.new("RGB", (width * self.ss, height * self.ss), BG)
        self.d = ImageDraw.Draw(self.img)
        self._fonts = {}
        self._paths = {False: _find_font(False), True: _find_font(True)}

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self._fonts:
            path = self._paths[bold] or self._paths[False]
            f = None
            if path:
                for index in ((2, 0) if path.endswith(".ttc") else (0,)):
                    try:
                        f = ImageFont.truetype(path, int(round(size * self.ss)), index=index)
                        break
                    except Exception:
                        f = None
            if f is None:  # no scalable font on this machine
                try:
                    f = ImageFont.load_default(size=int(round(size * self.ss)))
                except TypeError:  # Pillow < 10.1 bitmap default (cannot scale)
                    f = ImageFont.load_default()
            self._fonts[key] = f
        return self._fonts[key]

    def measure(self, s, size, bold=False):
        """Return (width, height) in logical pixels; supports old and new Pillow."""
        f = self.font(size, bold)
        try:
            if hasattr(f, "getlength"):
                w = f.getlength(s)
            elif hasattr(f, "getsize"):
                w = f.getsize(s)[0]
            else:  # last-resort approximation for very old/new Pillow defaults
                w = len(s) * size * 0.6
            ascent, descent = f.getmetrics()
            h = ascent + descent
        except Exception:  # bitmap default font cannot measure non-latin text
            w, h = len(s) * size * 0.6, size * 1.2
        return w / self.ss, h / self.ss

    def tw(self, s, size, bold=False):
        return self.measure(s, size, bold)[0]

    def fit(self, s, max_w, size, bold=False, min_size=9):
        """Shrink a font size until the string fits max_w."""
        while size > min_size and self.tw(s, size, bold) > max_w:
            size -= 0.5
        return size

    def th(self, s, size, bold=False):
        return self.measure(s, size, bold)[1]

    def text(self, x, y, s, size=13, bold=False, color=INK, align="left", valign="top"):
        f = self.font(size, bold)
        lines = s.split("\n")
        hs = [self.measure(line, size, bold)[1] for line in lines]
        gap = size * 0.35
        total = sum(hs) + gap * (len(lines) - 1)
        cy = y - total / 2 if valign == "center" else (y - total if valign == "bottom" else y)
        for line, line_h in zip(lines, hs):
            lw = self.measure(line, size, bold)[0]
            cx = x - lw / 2 if align == "center" else (x - lw if align == "right" else x)
            try:
                self.d.text((cx * self.ss, cy * self.ss), line, font=f, fill=color)
            except Exception:  # bitmap default font cannot draw non-latin text
                pass
            cy += line_h + gap

    # -- shapes ------------------------------------------------------------
    def dashed_line(self, x1, y1, x2, y2, color, width=1.5, dash=9, gap=6):
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        t = 0
        while t < length:
            t2 = min(t + dash, length)
            self.d.line(
                [(x1 + ux * t) * self.ss, (y1 + uy * t) * self.ss,
                 (x1 + ux * t2) * self.ss, (y1 + uy * t2) * self.ss],
                fill=color, width=max(1, int(width * self.ss)),
            )
            t = t2 + gap

    def rect(self, x, y, w, h, fill=None, outline=None, width=2, dashed=False):
        xy = [x * self.ss, y * self.ss, (x + w) * self.ss, (y + h) * self.ss]
        if not dashed:
            self.d.rectangle(xy, fill=fill, outline=outline, width=int(width * self.ss))
            return
        if fill:
            self.d.rectangle(xy, fill=fill)
        self.dashed_line(x, y, x + w, y, outline, width)
        self.dashed_line(x + w, y, x + w, y + h, outline, width)
        self.dashed_line(x + w, y + h, x, y + h, outline, width)
        self.dashed_line(x, y + h, x, y, outline, width)

    def line(self, pts, color, width=2, dashed=False):
        for a, b in zip(pts, pts[1:]):
            if dashed:
                self.dashed_line(a[0], a[1], b[0], b[1], color, width)
            else:
                self.d.line([a[0] * self.ss, a[1] * self.ss, b[0] * self.ss, b[1] * self.ss],
                            fill=color, width=max(1, int(width * self.ss)))

    def arrow(self, pts, color=ARROW, width=2, dashed=False, head=8):
        self.line(pts, color, width, dashed)
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        ang = math.atan2(y2 - y1, x2 - x1)
        ca, sa = math.cos(ang), math.sin(ang)
        base = (x2 - head * ca, y2 - head * sa)
        perp = (-sa, ca)
        p1 = (base[0] + perp[0] * head * 0.45, base[1] + perp[1] * head * 0.45)
        p2 = (base[0] - perp[0] * head * 0.45, base[1] - perp[1] * head * 0.45)
        self.d.polygon([(x2 * self.ss, y2 * self.ss), (p1[0] * self.ss, p1[1] * self.ss),
                        (p2[0] * self.ss, p2[1] * self.ss)], fill=color)

    def chip(self, x, yc, s, size=12, bold=False, fill=WHITE, stroke=SUB, pad=10, h=26,
             dashed=False, color=INK):
        w = max(self.tw(s, size, bold) + 2 * pad, 0)
        self.rect(x, yc - h / 2, w, h, fill=fill, outline=stroke, width=1.3, dashed=dashed)
        self.text(x + w / 2, yc, s, size=size, bold=bold, color=color, align="center", valign="center")
        return x + w

    def chain(self, x, yc, items, gap=12, size=12, fill=WHITE, stroke=SUB, bold=False):
        cur = x
        for i, s in enumerate(items):
            if i:
                self.arrow([(cur, yc), (cur + gap, yc)], color=ARROW, width=1.3, head=5)
                cur += gap
            cur = self.chip(cur, yc, s, size=size, bold=bold, fill=fill, stroke=stroke)
        return cur

    def chip2(self, x, y, w, h, top, bottom, fill, stroke, tsize=14, bsize=10.5):
        self.rect(x, y, w, h, fill=fill, outline=stroke, width=1.3)
        self.text(x + w / 2, y + h * 0.30, top, size=tsize, bold=True, align="center", valign="center")
        self.text(x + w / 2, y + h * 0.70, bottom, size=bsize, color=SUB, align="center", valign="center")

    def finish(self, path):
        self.img.resize((self.w, self.h), Image.LANCZOS).save(path)


# --------------------------------------------------------------------------- YAML parsing
class Layer:
    def __init__(self, index, frm, repeats, module, args):
        self.index = index
        self.frm = frm
        self.repeats = repeats
        self.module = module
        self.args = args if isinstance(args, list) else ([] if args is None else [args])
        self.sources = self._resolve(frm, index)
        self.stride = 1.0
        self.size = None  # (w, h) at the configured imgsz
        self.channels = 0  # output channels, from YAML base values

    @staticmethod
    def _resolve(frm, index):
        def one(v):
            return int(v) + index if int(v) < 0 else int(v)

        if isinstance(frm, (list, tuple)):
            return [one(v) for v in frm]
        return [one(frm)]

    @property
    def display_module(self):
        return self.module.split(".")[-1] if self.module.startswith("nn.") else self.module


def resolve_yaml(path):
    path = Path(path)
    if path.exists():
        return path
    name = path.name if path.suffix else f"{path.name}.yaml"
    here = Path(__file__).resolve().parent
    found = []
    for base in (Path.cwd(), here, here.parent):  # works from repo root or from draw_architecture/
        models = base / "ultralytics/cfg/models"
        if models.is_dir():
            found = sorted(models.rglob(name))
            if found:
                break
    if len(found) == 1:
        return found[0]
    if not found:
        raise SystemExit(f"YAML not found: {path}")
    raise SystemExit("Multiple matches, please give a full path:\n  " + "\n  ".join(str(p) for p in found))


def load_model(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    rows = list(cfg.get("backbone", [])) + list(cfg.get("head", []))
    if not rows:
        raise SystemExit(f"{path}: no 'backbone'/'head' layers found")
    layers = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            raise SystemExit(f"{path}: unsupported layer row {row!r}")
        frm, repeats, module = row[0], row[1], row[2]
        args = row[3] if len(row) > 3 else []
        layers.append(Layer(i, frm, repeats, module, args))
    return cfg, layers


def compact_args(args, limit=48):
    def fmt(v):
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{v:g}"
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return str(v)

    s = ", ".join(fmt(a) for a in args)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def compute_shapes(layers, imgsz):
    """Estimate stride and spatial size; enough for YOLO-style Conv/Upsample chains."""
    conv_mods = {"conv", "dwconv", "ghostconv", "repconv", "conv2d", "convtranspose2d", "downsample"}
    for layer in layers:
        inputs = [s for s in layer.sources if s >= 0]
        base = max((layers[s].stride for s in inputs), default=1.0)
        mod = layer.display_module.lower()
        args = layer.args
        stride = base
        if "upsample" in mod:
            scale = args[1] if len(args) > 1 and isinstance(args[1], (int, float)) else (
                args[0] if args and isinstance(args[0], (int, float)) else 1)
            stride = base / max(float(scale), 1.0)
        elif mod in conv_mods or mod.startswith("conv"):
            s = args[2] if len(args) > 2 and isinstance(args[2], int) and args[2] in (1, 2) else 1
            stride = base * s
        elif mod in ("hgstem", "focus"):
            stride = base * 2
        layer.stride = stride
        if stride >= 1 and abs(stride - round(stride)) < 1e-6:
            size = imgsz // int(round(stride))
            layer.size = (size, size) if size > 0 else None


def compute_channels(layers):
    """Estimate output channels; first arg is c2 for most blocks, Concat sums its inputs."""
    for layer in layers:
        inputs = [s for s in layer.sources if s >= 0]
        mod = layer.display_module.lower()
        args = layer.args
        if is_head(layer.module):
            ch = 0
        elif mod == "concat":
            ch = sum(layers[s].channels for s in inputs)
        elif "upsample" in mod:
            ch = layers[inputs[0]].channels if inputs else 0
        elif args and isinstance(args[0], int):
            ch = args[0]
        else:
            ch = max((layers[s].channels for s in inputs), default=0)
        layer.channels = ch


def head_dims(layer, cfg):
    """Return (nc, nk, kpt_shape) for a head layer."""
    nc = layer.args[0] if layer.args and isinstance(layer.args[0], int) else cfg.get("nc", "?")
    kpt_shape = layer.args[1] if len(layer.args) > 1 and isinstance(layer.args[1], (list, tuple)) else cfg.get(
        "kpt_shape", [17, 3])
    kpt_shape = list(kpt_shape) if isinstance(kpt_shape, (list, tuple)) else [17, 3]
    return nc, int(kpt_shape[0]) * int(kpt_shape[1]), kpt_shape


def scale_label(stride):
    labels = {4: "P2/4", 8: "P3/8", 16: "P4/16", 32: "P5/32", 64: "P6/64"}
    return labels.get(int(round(stride)), f"s{int(round(stride))}")


def module_style(module):
    m = module.lower()
    if any(k in m for k in HEAD_KEYS):
        return HEAD_F, HEAD_S
    if any(k in m for k in ("concat", "upsample", "merge", "shortcut")) or m == "add":
        return MERGE_F, MERGE_S
    if any(k in m for k in ("psa", "attention", "attn", "transformer", "aifi", "savpe")):
        return ATT_F, ATT_S
    if any(k in m for k in ("spp", "aspp")):
        return SPP_F, SPP_S
    if any(k in m for k in ("c3k2", "c2f", "c3", "c2", "bottleneck", "elan", "rep", "csp")):
        return CSP_F, CSP_S
    if m.startswith("conv") or m in ("dwconv", "ghostconv", "repconv", "conv2d", "convtranspose2d"):
        return CONV_F, CONV_S
    return MERGE_F, MERGE_S


def is_head(module):
    return any(k in module.lower() for k in HEAD_KEYS)


def head_kind(module):
    m = module.lower()
    if "semantic" in m or "decoder" in m:
        return "generic"  # dense / transformer heads have a different internal layout
    if "pose" in m:
        return "pose"
    if "segment" in m:
        return "segment"
    if "obb" in m:
        return "obb"
    if "classif" in m:
        return "classify"
    if "depth" in m:
        return "depth"
    if "detect" in m or "head" in m:
        return "detect"
    return "generic"


# --------------------------------------------------------------------------- layout helpers
NODE_W, NODE_H, NODE_GAP = 280, 64, 14
NODE_X = 70
Y_TOP = 300


# --------------------------------------------------------------------------- head detail panels
def _panel_header(p, x, y, w, title, subtitle, fill, stroke, height):
    p.rect(x, y, w, height, fill=fill, outline=stroke, width=2)
    p.text(x + 18, y + 12, title, 24, bold=True)
    p.text(x + 18, y + 48, subtitle, 14, color=SUB)


def draw_head_panel(p, x, y, w, layer, cfg, layers, max_det=300):
    kind = head_kind(layer.module)
    nc, nk, kpt_shape = head_dims(layer, cfg)
    reg_max = cfg.get("reg_max", 16)
    end2end = bool(cfg.get("end2end", False))
    nscale = len([s for s in layer.sources if s >= 0])
    mod = layer.display_module
    # parse_model keeps legacy=True unless the model uses a modern block (C3k2/A2C2f/C2fCIB)
    modern = any(l.module in ("C3k2", "A2C2f", "C2fCIB") for l in layers)
    cls_items = (["DWConv 3×3", "Conv 1×1", "DWConv 3×3", "Conv 1×1", f"Conv 1×1 → {nc}"]
                 if modern else ["Conv 3×3", "Conv 3×3", f"Conv 1×1 → {nc}"])
    anchors = sum(layers[s].size[0] * layers[s].size[1] for s in layer.sources if s >= 0 and layers[s].size)

    if kind == "pose":
        pose26 = "pose26" in mod.lower()
        h = 700 if end2end else 540
        _panel_header(p, x, y, w, f"{mod} 头详情",
                      f"nc = {nc} · kpt_shape = {kpt_shape} · reg_max = {reg_max}"
                      f"{' · end2end' if end2end else ''}", "#fdfaff", ATT_S, h)

        groups = []
        if end2end:
            groups.append(("one2many（训练分支）", y + 68, True))
            groups.append(("one2one（训练 + 推理，NMS-free）", y + 248, False))
        else:
            groups.append(("预测分支", y + 72, False))

        for title, ty, dashed in groups:
            p.text(x + 18, ty, title, 13.5, bold=True, color=SUB)
            gy = ty + 24
            p.rect(x + 12, gy, w - 24, 150, fill="#fcfcfc" if dashed else WHITE,
                   outline=ATT_S, width=1.2, dashed=dashed)

            yc = gy + 20
            p.text(x + 18, yc, "Box  cv2", 14, bold=True, valign="center")
            p.chain(x + 150, yc, ["Conv 3×3", "Conv 3×3", f"Conv 1×1 → {4 * reg_max}"],
                    size=13.5, gap=12)

            yc = gy + 58
            p.text(x + 18, yc, "Cls  cv3", 14, bold=True, valign="center")
            p.chain(x + 150, yc, cls_items, size=13.5, gap=12)

            # shared pose trunk cv4, then kpts (solid) and sigma (dashed) in parallel
            yc = gy + 96
            p.text(x + 18, yc, "Pose cv4", 14, bold=True, valign="center")
            e = p.chain(x + 150, yc, ["Conv 3×3 (c4)", "Conv 3×3 (c4)"], size=13.5, gap=12)
            p.arrow([(e, yc), (e + 24, yc)], color=ARROW, width=1.5, head=6)
            p.chip(e + 24, yc, f"cv4_kpts 1×1 → {nk}", size=13, fill=WHITE, stroke=SUB)
            yc_sigma = yc + 30
            p.dashed_line(e + 10, yc, e + 10, yc_sigma, TR_S, width=1.5, dash=6, gap=4)
            p.arrow([(e + 10, yc_sigma), (e + 24, yc_sigma)], color=TR_S, width=1.5, dashed=True, head=6)
            p.chip(e + 24, yc_sigma, f"cv4_sigma 1×1 → {int(kpt_shape[0]) * 2}（仅训练）",
                   size=12.5, fill=TR_F, stroke=TR_S, dashed=True)

        if pose26:
            p.rect(x + 12, y + 428, w - 24, 28, fill=TR_F, outline=TR_S, width=1.2, dashed=True)
            p.text(x + w / 2, y + 442,
                   "RealNVP 归一化流：仅训练，loss 中计算 log_prob；推理丢弃 cv4_sigma",
                   12, color=SUB, align="center", valign="center")

        oy = y + (470 if end2end else 260)
        p.rect(x + 12, oy, w - 24, h - (oy - y) - 14, fill=OUT_F, outline=OUT_S, width=1.8)
        nc_i = nc if isinstance(nc, int) else 0
        if end2end:
            p.text(x + 26, oy + 14, f"每 anchor：4 + {nc} + {nk} → {nscale} 尺度拼接（{anchors} anchors）"
                                      f"→ 解码 → top-k（{max_det}）", 14)
            p.text(x + 26, oy + 40, f"输出 [1, {max_det}, {6 + nk}]", 21, bold=True)
        else:
            p.text(x + 26, oy + 14, f"每 anchor：4 + {nc} + {nk} → {nscale} 尺度拼接（{anchors} anchors）"
                                      "→ 原始输出", 14)
            p.text(x + 26, oy + 40, f"输出 [1, {4 + nc_i + nk}, {anchors}]（需 NMS）", 21, bold=True)
        cx = x + 26
        for cw, top, bottom in ((170, "4", "x1,y1,x2,y2"), (100, "1", "conf"), (110, "1", "cls"),
                                (200, str(nk), f"{kpt_shape[0]} × (x, y, v)")):
            p.chip2(cx, oy + 68, cw, 42, top, bottom, WHITE, ATT_S, tsize=17, bsize=12.5)
            if cw != 200:
                p.text(cx + cw + 10, oy + 89, "+", 16, color=SUB, align="center", valign="center")
            cx += cw + 22
        p.text(x + 26, oy + 124, "• v 经 sigmoid ∈ (0,1)，表示可见度 / 关键点置信度", 12.5)
        p.text(x + 26, oy + 146, "• x,y 位于 letterbox 输入像素空间；坐标需逆 letterbox 回原图", 12.5)
        p.text(x + 26, oy + 168, f"• 固定 {max_det} 行 top-k 槽位，使用前仍需按 conf 过滤" if end2end
               else "• 原始输出需先 NMS，再按 conf 过滤", 12.5)
        return h

    if kind == "detect":
        h = 470 if end2end else 400
        _panel_header(p, x, y, w, f"{mod} 头详情",
                      f"nc = {nc} · reg_max = {reg_max} · 输出尺度 {nscale} 个"
                      f"{' · end2end' if end2end else ' · 需 NMS'}", "#fffdf6", HEAD_S, h)
        groups = []
        if end2end:
            groups.append(("one2many（训练分支）", y + 68, y + 92, y + 106, y + 144, True))
            groups.append(("one2one（训练 + 推理，NMS-free）", y + 184, y + 208, y + 222, y + 260, False))
        else:
            groups.append(("预测分支", y + 72, y + 90, y + 106, y + 144, False))
        for title, ty, gy, r1, r2, dashed in groups:
            p.text(x + 18, ty, title, 13.5, bold=True, color=SUB)
            p.rect(x + 12, gy, w - 24, 84, fill="#fcfcfc" if dashed else WHITE,
                   outline=HEAD_S, width=1.2, dashed=dashed)
            p.text(x + 18, r1, "Box  cv2", 14, bold=True, valign="center")
            p.chain(x + 150, r1, ["Conv 3×3", "Conv 3×3", f"Conv 1×1 → {4 * reg_max}"],
                    size=13.5, gap=12)
            p.text(x + 18, r2, "Cls  cv3", 14, bold=True, valign="center")
            p.chain(x + 150, r2, cls_items, size=13.5, gap=12)
        oy = y + 306 if end2end else y + 192
        oh = h - (oy - y) - 14
        p.rect(x + 12, oy, w - 24, oh, fill=OUT_F, outline=OUT_S, width=1.8)
        nc_i = nc if isinstance(nc, int) else 0
        if end2end:
            p.text(x + 26, oy + 14, f"每 anchor：4 + {nc} → {nscale} 尺度拼接（{anchors} anchors）"
                                      f"→ 解码 → top-k（{max_det}）", 14)
            p.text(x + 26, oy + 40, f"输出 [1, {max_det}, 6]", 21, bold=True)
            p.text(x + 26, oy + 74, "列顺序：x1, y1, x2, y2, conf, cls", 12.5, color=SUB)
            p.text(x + 26, oy + 96, "conf 为所选类别的概率；cls 为类别索引", 12.5, color=SUB)
        else:
            p.text(x + 26, oy + 14, f"每 anchor：4 + {nc} → {nscale} 尺度拼接（{anchors} anchors）"
                                      "→ 原始输出", 14)
            p.text(x + 26, oy + 40, f"输出 [1, {4 + nc_i}, {anchors}]（需 NMS）", 21, bold=True)
            p.text(x + 26, oy + 74, "列顺序：x1, y1, x2, y2, cls×nc（无单独 conf）", 12.5, color=SUB)
            p.text(x + 26, oy + 96, f"NMS 后才能得到 [1, {max_det}, 6] 形式的 x1,y1,x2,y2,conf,cls",
                   12.5, color=SUB)
        return h

    if kind in ("segment", "obb"):
        h = 520 if end2end else 380
        _panel_header(p, x, y, w, f"{mod} 头详情",
                      f"nc = {nc} · reg_max = {reg_max} · 输出尺度 {nscale} 个", "#fffdf6", HEAD_S, h)
        groups = []
        if end2end:
            groups.append(("one2many（训练分支）", y + 68, True))
            groups.append(("one2one（训练 + 推理，NMS-free）", y + 226, False))
        else:
            groups.append(("预测分支", y + 72, False))
        for title, ty, dashed in groups:
            p.text(x + 18, ty, title, 13.5, bold=True, color=SUB)
            gy = ty + 24
            p.rect(x + 12, gy, w - 24, 130, fill="#fcfcfc" if dashed else WHITE,
                   outline=HEAD_S, width=1.2, dashed=dashed)
            p.text(x + 18, gy + 20, "Box  cv2", 14, bold=True, valign="center")
            p.chain(x + 150, gy + 20, ["Conv 3×3", "Conv 3×3", f"Conv 1×1 → {4 * reg_max}"],
                    size=13.5, gap=12)
            p.text(x + 18, gy + 58, "Cls  cv3", 14, bold=True, valign="center")
            p.chain(x + 150, gy + 58, cls_items, size=13.5, gap=12)
            if kind == "segment":
                p.text(x + 18, gy + 96, "Mask proto", 14, bold=True, valign="center")
                p.chain(x + 150, gy + 96, ["Conv 3×3", "Upsample ×2", "Conv 3×3", "Conv 1×1 → 32"],
                        size=13.5, gap=12)
            else:
                p.text(x + 18, gy + 96, "Angle", 14, bold=True, valign="center")
                p.text(x + 150, gy + 96, "框回归含角度 θ（OBB 头）", 13, valign="center")

        oy = y + (392 if end2end else 236)
        p.rect(x + 12, oy, w - 24, h - (oy - y) - 14, fill=OUT_F, outline=OUT_S, width=1.8)
        nc_i = nc if isinstance(nc, int) else 0
        extra = "32 mask 系数" if kind == "segment" else "1 角度"
        raw_ch = 4 + nc_i + (32 if kind == "segment" else 1)
        if end2end:
            p.text(x + 26, oy + 14,
                   f"每 anchor：4 + {extra} + {nc} → 拼接（{anchors} anchors）"
                   f"→ 解码 → top-k（{max_det}）", 14)
            p.text(x + 26, oy + 40, f"输出 [1, {max_det}, {6 + (32 if kind == 'segment' else 1)}]",
                   21, bold=True)
            p.text(x + 26, oy + 74, f"固定 {max_det} 行 top-k 槽位，使用前按 conf 过滤", 12.5, color=SUB)
        else:
            p.text(x + 26, oy + 14,
                   f"每 anchor：4 + {extra} + {nc} → 拼接（{anchors} anchors）→ 原始输出", 14)
            p.text(x + 26, oy + 40, f"输出 [1, {raw_ch}, {anchors}]（需 NMS）", 21, bold=True)
            p.text(x + 26, oy + 74, f"NMS 后才能得到 [1, {max_det}, ...] 的检测结果", 12.5, color=SUB)
        return h

    # generic head (classify / depth / decoder / unknown)
    h = 280
    _panel_header(p, x, y, w, f"{mod} 头详情", f"args = [{compact_args(layer.args)}]", "#fffdf6", HEAD_S, h)
    p.text(x + 18, y + 90, f"输入来源：{[s for s in layer.sources if s >= 0]}", 14)
    p.text(x + 18, y + 118, f"nc = {nc} · 输出尺度 {nscale} 个", 14)
    p.rect(x + 12, y + 148, w - 24, 110, fill=OUT_F, outline=OUT_S, width=1.8)
    p.text(x + 26, y + 166, "该头未内置详细展开，仅显示 YAML 中的模块与参数。", 13.5)
    p.text(x + 26, y + 196, "已知 Detect / Pose / Segment / OBB 头会自动展开内部结构。", 13.5, color=SUB)
    p.text(x + 26, y + 226, "如需新头型的细节，在 draw_head_panel() 中增加分支即可。",
           13.5, color=SUB)
    return h


# --------------------------------------------------------------------------- main renderer
def render(cfg, layers, yaml_path, out_path, imgsz=640, title=None, head_detail=True, supersample=2,
           max_det=300):
    compute_shapes(layers, imgsz)
    compute_channels(layers)
    n = len(layers)
    nb = min(len(cfg.get("backbone", [])), n)

    # stage split: backbone | neck (head rows before the final output head) | head
    final_is_head = n > nb and is_head(layers[-1].module)
    head_indices = [n - 1] if final_is_head else []
    neck_indices = list(range(nb, n - 1)) if head_indices else list(range(nb, n))
    columns = [("Backbone", list(range(nb)), "#f7faff", CONV_S)]
    if neck_indices:
        columns.append(("Neck（FPN + PAN）", neck_indices, "#f6fbf7", CSP_S))
    if head_indices:
        columns.append(("Head", head_indices, "#fffbf4", HEAD_S))

    ys = [0.0] * n
    col_of = [0] * n
    for ci, (_, idxs, _, _) in enumerate(columns):
        for k, idx in enumerate(idxs):
            col_of[idx] = ci
            ys[idx] = Y_TOP + k * (NODE_H + NODE_GAP)

    # ---- edges ---------------------------------------------------------
    chain_edges, nonchain = [], []
    for layer in layers:
        if head_indices and layer.index == n - 1:
            continue  # head inputs are expanded into one branch per scale below
        for src in layer.sources:
            if src < 0:
                continue
            if src == layer.index - 1 and col_of[src] == col_of[layer.index]:
                chain_edges.append((src, layer.index))
            else:
                nonchain.append((src, layer.index))

    # entry y along the target edge, spread when several inputs merge
    incoming = {}
    for src, dst in nonchain:
        incoming.setdefault(dst, []).append(src)
    entry = {}
    for dst, srcs in incoming.items():
        for k, src in enumerate(srcs):
            entry[(src, dst)] = ys[dst] + NODE_H * (k + 1) / (len(srcs) + 1)

    # each non-chain edge uses the routing gutter right of its source column
    gutters = {}
    for src, dst in nonchain:
        c = col_of[src]
        kind = "within" if col_of[dst] == c else ("cross" if col_of[dst] == c + 1 else "jump")
        gutters.setdefault(c, []).append([src, dst, kind, entry[(src, dst)], None])
    lane_counts = {}
    for c, edges in gutters.items():
        edges.sort(key=lambda e: min(ys[e[0]], e[3]))
        lane_end = []
        for e in edges:
            a = min(ys[e[0]] + NODE_H / 2, e[3])
            b = max(ys[e[0]] + NODE_H / 2, e[3])
            lane = next((k for k, end in enumerate(lane_end) if end < a), None)
            if lane is None:
                lane = len(lane_end)
                lane_end.append(b)
            else:
                lane_end[lane] = b
            e[4] = lane
        lane_counts[c] = len(lane_end)

    # ---- canvas geometry ----------------------------------------------
    xs = [NODE_X]
    for c in range(len(columns) - 1):
        xs.append(xs[c] + NODE_W + max(80, 24 + lane_counts.get(c, 0) * 15 + 16))

    end2end = bool(cfg.get("end2end", False))
    head_layer = layers[-1] if head_indices else None
    head_sources = sorted([s for s in head_layer.sources if s >= 0], key=lambda s: ys[s]) if head_layer else []
    head_x = xs[-1] if head_layer else 0
    panel_layer = head_layer if (head_detail and head_layer) else None
    panel_w = 860
    panel_h = 0
    if panel_layer:
        panel_h = {"pose": 700 if end2end else 540,
                   "detect": 470 if end2end else 400,
                   "segment": 520 if end2end else 380,
                   "obb": 520 if end2end else 380,
                   "classify": 280, "depth": 280, "generic": 280}[head_kind(panel_layer.module)]

    node_bottom = max([ys[i] + NODE_H for i in range(n) if not (head_indices and i == n - 1)]
                      + [ys[s] + NODE_H for s in head_sources])
    width = max(900, xs[-1] + NODE_W + 40)
    height = max(760, node_bottom + 50)
    if panel_layer:
        # landscape layout: head details sit to the right of the head column
        head_band_top = min((ys[s] for s in head_sources), default=Y_TOP) - 20
        panel_x = xs[-1] + NODE_W + 50
        panel_y = head_band_top - 60
        width = max(width, panel_x + panel_w + 40)
        height = max(height, panel_y + panel_h + 40)

    p = Painter(width, height, supersample)

    # ---- title / legend ------------------------------------------------
    try:
        source_label = Path(yaml_path).resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        source_label = Path(yaml_path)
    p.text(42, 16, title or f"{Path(yaml_path).stem} 网络结构", 38, bold=True)
    p.text(44, 74, f"source: {source_label} · imgsz {imgsz} · nc={cfg.get('nc', '?')}"
                   f"{' · end2end=True' if cfg.get('end2end') else ''} · 未应用 scales", 13.5, color=SUB)
    p.text(
        44,
        98,
        "节点颜色按模块类型区分；C×H×W 为 YAML 基准值，stride 按 Conv/Upsample 链估算；"
        "nc 可被数据集 yaml 覆盖",
        13,
        color=SUB,
    )

    legend = [
        ("Conv", CONV_F, CONV_S, False),
        ("C2/C3/CSP", CSP_F, CSP_S, False),
        ("SPP", SPP_F, SPP_S, False),
        ("Attention", ATT_F, ATT_S, False),
        ("Concat/Upsample", MERGE_F, MERGE_S, False),
        ("Head", HEAD_F, HEAD_S, False),
    ]
    lx = 44
    for label, fill, stroke, dash in legend:
        p.rect(lx, 122, 24, 18, fill=fill, outline=stroke, width=1.5, dashed=dash)
        p.text(lx + 31, 132, label, 13, valign="center")
        lx += 31 + p.tw(label, 13) + 26
    p.line([(lx + 6, 132), (lx + 40, 132)], SKIP, width=2)
    p.text(lx + 46, 132, "skip / multi-input", 13, valign="center")

    # ---- stage bands and headers --------------------------------------
    for ci, (label, idxs, tint, accent) in enumerate(columns):
        is_head_col = head_layer is not None and ci == len(columns) - 1
        if is_head_col:
            hy0 = min((ys[s] for s in head_sources), default=Y_TOP)
            hy1 = max((ys[s] for s in head_sources), default=Y_TOP)
            band_top, band_bottom = hy0 - 20, hy1 + NODE_H + 14
            header_y, underline_y = band_top - 56, band_top - 18
        else:
            band_top, band_bottom = Y_TOP - 18, ys[idxs[-1]] + NODE_H + 14
            header_y, underline_y = 156, 194
        p.rect(xs[ci] - 12, band_top, NODE_W + 24, band_bottom - band_top, fill=tint)
        p.text(xs[ci] + NODE_W / 2, header_y, label, 25, bold=True, align="center")
        p.rect(xs[ci] + NODE_W / 2 - 40, underline_y, 80, 4, fill=accent)

    # ---- input ---------------------------------------------------------
    p.rect(xs[0], 212, NODE_W, NODE_H, fill=MERGE_F, outline=MERGE_S, width=2)
    p.text(xs[0] + 12, 221, "#in", 11, color=SUB)
    p.text(xs[0] + NODE_W / 2, 236, "Input", 17, bold=True, align="center")
    p.text(xs[0] + NODE_W / 2, 258, f"{imgsz}×{imgsz}×3", 14, color=SUB, align="center")
    p.arrow([(xs[0] + NODE_W / 2, 212 + NODE_H), (xs[0] + NODE_W / 2, Y_TOP)])

    # ---- layer nodes ---------------------------------------------------
    for layer in layers:
        if head_layer is not None and layer.index == n - 1:
            continue
        x, y = xs[col_of[layer.index]], ys[layer.index]
        fill, stroke = module_style(layer.module)
        head = is_head(layer.module)
        p.rect(x, y, NODE_W, NODE_H, fill=fill, outline=stroke, width=2.5 if head else 2)
        p.text(x + 9, y + 7, f"#{layer.index}", 11, color=SUB)
        title_txt = layer.display_module + (f" ×{layer.repeats}" if layer.repeats and layer.repeats > 1 else "")
        p.text(x + NODE_W / 2 + 8, y + 10, title_txt, p.fit(title_txt, NODE_W - 55, 20, True),
               bold=True, align="center")
        if head:
            strides = sorted({int(round(layers[s].stride)) for s in layer.sources if s >= 0})
            sub = "inputs: " + (" / ".join(f"s{x}" for x in strides) if strides else "-")
        elif layer.size and layer.channels:
            sub = f"{layer.channels}×{layer.size[0]}×{layer.size[1]} · s{layer.stride:g}"
        elif layer.size:
            sub = f"{layer.size[0]}×{layer.size[1]} · s{layer.stride:g}"
        elif layer.channels:
            sub = f"{layer.channels} ch"
        else:
            sub = ""
        p.text(x + NODE_W / 2, y + 38, sub, 15, color=SUB, align="center")

    # ---- chain arrows --------------------------------------------------
    for src, dst in chain_edges:
        x = xs[col_of[src]] + NODE_W / 2
        p.arrow([(x, ys[src] + NODE_H), (x, ys[dst])], width=1.8)

    # ---- routed arrows (skip / cross-stage / input merges) -------------
    for c, edges in gutters.items():
        right = xs[c] + NODE_W
        for src, dst, kind, ey, lane in edges:
            lane_x = right + 24 + lane * 15
            sy = ys[src] + NODE_H / 2
            if kind == "within":
                pts = [(right, sy), (lane_x, sy), (lane_x, ey), (right, ey)]
            elif kind == "cross":
                pts = [(right, sy), (lane_x, sy), (lane_x, ey), (xs[c + 1], ey)]
            else:  # jump over an intermediate stage through the corridor above the nodes
                corridor = Y_TOP - 32
                pts = [(right, sy), (lane_x, sy), (lane_x, corridor), (xs[col_of[dst]] - 34, corridor),
                       (xs[col_of[dst]] - 34, ey), (xs[col_of[dst]], ey)]
            p.arrow(pts, color=SKIP, width=1.5, head=6)

    # ---- head: one branch per input scale (P3 / P4 / P5) ---------------
    if head_layer:
        kind = head_kind(head_layer.module)
        nc, nk, _ = head_dims(head_layer, cfg)
        nc_i = nc if isinstance(nc, int) else 0
        head_col = col_of[head_layer.index]
        multi = len(head_sources) > 1
        for src in head_sources:
            y = ys[src]
            label = scale_label(layers[src].stride) if multi else ""
            title_txt = f"{head_layer.display_module} · {label}" if label else head_layer.display_module
            tsize = p.fit(title_txt, NODE_W - 60, 17, True)
            p.rect(head_x, y, NODE_W, NODE_H, fill=HEAD_F, outline=HEAD_S, width=2.5)
            p.text(head_x + NODE_W - 9, y + 5, f"#{head_layer.index}", 11, color=SUB, align="right")
            p.text(head_x + NODE_W / 2, y + 6, title_txt, tsize, bold=True, align="center")
            feat = layers[src]
            dims = (f"{feat.channels}×{feat.size[0]}×{feat.size[1]}" if feat.channels and feat.size
                    else (f"{feat.size[0]}×{feat.size[1]}" if feat.size else ""))
            p.text(head_x + NODE_W / 2, y + 30, f"输入 {dims}" if dims else "输入特征",
                   13.5, color=SUB, align="center")
            if kind == "detect":
                out_txt = f"每 anchor：4 + {nc} = {4 + nc_i}"
            elif kind == "pose":
                out_txt = f"每 anchor：4 + {nc} + {nk} = {4 + nc_i + nk}"
            elif kind == "segment":
                out_txt = f"每 anchor：4 + {nc} + 32 = {4 + nc_i + 32}"
            elif kind == "obb":
                out_txt = f"每 anchor：4 + {nc} + 1 = {5 + nc_i}"
            elif kind == "classify":
                out_txt = f"{nc} 类"
            else:
                out_txt = "共享输出头"
            p.text(head_x + NODE_W / 2, y + 48, out_txt, 11.5, color=SUB, align="center")
            # straight arrow from the neck feature map; longer detours go over the corridor
            src_col = col_of[src]
            right = xs[src_col] + NODE_W
            sy = y + NODE_H / 2
            if src_col == head_col - 1:
                p.arrow([(right, sy), (head_x, sy)], color=SKIP, width=1.8, head=7)
            else:
                corridor = Y_TOP - 40
                p.arrow([(right, sy), (right + 20, sy), (right + 20, corridor), (head_x - 30, corridor),
                         (head_x - 30, sy), (head_x, sy)], color=SKIP, width=1.5, head=6)

    # ---- head detail panel (right of the head column) -------------------
    if panel_layer:
        draw_head_panel(p, panel_x, panel_y, panel_w, panel_layer, cfg, layers, max_det)

    p.finish(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Plot an Ultralytics model architecture from a model YAML.")
    ap.add_argument("yaml", type=Path, help="model YAML path or name, e.g. yolo26-pose.yaml")
    ap.add_argument("-o", "--output", type=Path, help="output PNG (default: <yaml-stem>_arch.png)")
    ap.add_argument("--imgsz", type=int, default=640, help="input size used for spatial labels (default: 640)")
    ap.add_argument("--title", help="diagram title")
    ap.add_argument("--no-head-detail", action="store_true", help="do not expand Detect/Pose head internals")
    ap.add_argument("--font", type=Path,
                    help="path to a .ttf/.ttc font; use when the container has no CJK fonts installed")
    ap.add_argument("--max-det", type=int, default=300,
                    help="top-k slots shown for the NMS-free head (default: 300, same as export max_det)")
    ap.add_argument("--supersample", type=int, default=2, help="render scale before downsampling (default: 2)")
    args = ap.parse_args()

    global _FONT_OVERRIDE
    if args.font:
        if not args.font.exists():
            raise SystemExit(f"Font not found: {args.font}")
        _FONT_OVERRIDE = str(args.font)
    reg = _find_font(False)
    if not reg:
        print("WARNING: no scalable font found, text will be tiny. "
              "Install fonts-noto-cjk or pass --font /path/to/font.ttf")
    elif not any(k in Path(reg).name.lower() for k in ("cjk", "wqy", "msyh", "pingfang", "simhei", "arial")):
        print(f"WARNING: font '{reg}' has no CJK glyphs, Chinese may render as boxes. "
              "Install fonts-noto-cjk or pass --font with a CJK font.")

    yaml_path = resolve_yaml(args.yaml)
    cfg, layers = load_model(yaml_path)
    out = args.output or (Path.cwd() / f"{yaml_path.stem}_arch.png")
    if not out.suffix:
        out = out.with_suffix(".png")

    path = render(cfg, layers, yaml_path, out, args.imgsz, args.title, not args.no_head_detail,
                  args.supersample, args.max_det)
    print(f"Saved {path}")


if __name__ == "__main__":
    # 使用示例：
    # python draw_architecture/plot_model_arch.py ultralytics/cfg/models/26/yolo26-pose.yaml \
    #     -o draw_architecture/pose_arch.png
    main()
