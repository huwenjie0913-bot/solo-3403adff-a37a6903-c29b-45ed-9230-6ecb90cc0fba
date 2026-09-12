# -*- coding: utf-8 -*-
"""逐齿误差分析的 SVG 导出。

极坐标误差图：名义齿尖圆（灰虚线）、逐齿径向偏差（蓝点连线，放大
绘制）、逐齿累计节距偏差（绿色角向刻线，放大绘制）、轮心偏心向量
（红色箭头）、齿号标注，以及异常齿标记（红圈）与最差齿标记（橙色
菱形）。所有误差均按自动比例放大并在图例中注明比例。
"""
from __future__ import annotations

import math
from xml.sax.saxutils import escape

# 布局
W = H = 920
CX = CY = 430.0
R0 = 250.0          # 名义齿尖圆绘图半径 px
LABEL_R = 284.0     # 齿号标注半径
DEV_MAX_PX = 55.0   # 径向偏差最大放大到 55px
ANG_MAX_FRAC = 0.30  # 角向偏差最多放大到齿距角的 30%


def _fmt(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}"


def _pt(angle: float, r: float):
    """数学极角（度，逆时针）-> SVG 坐标（y 向下）。"""
    a = math.radians(angle)
    return CX + r * math.cos(a), CY - r * math.sin(a)


def _line(p1, p2, stroke, width=1.0, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" '
            f'y2="{p2[1]:.1f}" stroke="{stroke}" stroke-width="{width}"'
            f'{d} opacity="{opacity}"/>')


def _circle(p, r, stroke="none", fill="none", width=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="{r:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}" fill="{fill}"{d}/>')


def _text(p, s, size=11, anchor="middle", fill="#222", rotate=None):
    tr = f' transform="rotate({rotate} {p[0]:.1f} {p[1]:.1f})"' \
        if rotate is not None else ""
    return (f'<text x="{p[0]:.1f}" y="{p[1]:.1f}" font-size="{size}" '
            f'font-family="monospace" text-anchor="{anchor}" '
            f'fill="{fill}"{tr}>{escape(str(s))}</text>')


def _polygon(pts, stroke, fill="none", width=1.2):
    s = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts)
    return (f'<polygon points="{s}" stroke="{stroke}" '
            f'stroke-width="{width}" fill="{fill}"/>')


def _arrow(p1, p2, stroke, width=2.0, head=8.0):
    """带箭头线。"""
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    h1 = (p2[0] - head * math.cos(ang - 0.42),
          p2[1] - head * math.sin(ang - 0.42))
    h2 = (p2[0] - head * math.cos(ang + 0.42),
          p2[1] - head * math.sin(ang + 0.42))
    return (_line(p1, p2, stroke, width)
            + _polygon([p2, h1, h2], stroke, fill=stroke, width=1.0))


def wheel_svg(inp, table, analysis: dict) -> str:
    """生成整轮逐齿误差极坐标 SVG（只读取入参，不修改）。"""
    n = analysis["teeth"]
    pitch_deg = 360.0 / n
    eff = analysis["effective_teeth"]
    per_tooth = {row["tooth"]: row for row in analysis["per_tooth"]}
    cls = analysis["classification"]
    worst = analysis["worst_tooth"]

    dr = [eff[k]["radial_deviation_mm"] for k in range(n)]
    cum = [eff[k]["angle_deg"] - k * pitch_deg for k in range(n)]
    ecc_x, ecc_y = table.eccentricity.x, table.eccentricity.y
    ecc = math.hypot(ecc_x, ecc_y)

    # 自动比例
    max_dr = max([abs(v) for v in dr] + [ecc, 1e-9])
    scale_r = DEV_MAX_PX / max_dr
    max_cum = max([abs(v) for v in cum] + [1e-9])
    scale_a = (ANG_MAX_FRAC * pitch_deg) / max_cum
    scale_a = min(scale_a, 50.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
        f'height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#fafafa"/>',
        _text((W / 2, 30),
              f"逐齿误差极坐标图 — {escape(inp.name)} / "
              f"{escape(table.name)}", 16, fill="#111"),
        _text((W / 2, 50),
              f"{n} 齿  节距 {pitch_deg:.2f}°  "
              f"Rp={inp.wheel.pitch_radius:g} mm", 12, fill="#555"),
        # 名义齿尖圆 / 中心
        _circle((CX, CY), R0, stroke="#999", width=1.0, dash="5 4"),
        _circle((CX, CY), 2.5, fill="#333"),
    ]

    # 径向偏差曲线（蓝）：半径 = R0 + Δr*scale_r
    dev_pts = []
    for k in range(n):
        ang = k * pitch_deg
        dev_pts.append(_pt(ang, R0 + dr[k] * scale_r))
    parts.append(_polygon(dev_pts + [dev_pts[0]], "#1f6fb2", width=1.4))

    for k in range(n):
        ang = k * pitch_deg
        # 名义位置径向刻线
        parts.append(_line(_pt(ang, R0 - 4), _pt(ang, R0 + 4),
                           "#bbb", 0.8))
        # 累计节距偏差（绿）：实际角位置（放大）处的角向刻线
        if abs(cum[k]) > 1e-12:
            a_act = ang + cum[k] * scale_a
            parts.append(_line(_pt(a_act, R0 - 9), _pt(a_act, R0 + 9),
                               "#2c8c3f", 1.6))
        # 径向偏差点
        p = _pt(ang, R0 + dr[k] * scale_r)
        parts.append(_circle(p, 3.0, fill="#1f6fb2"))
        # 齿号
        parts.append(_text(_pt(ang, LABEL_R), str(k), 11, fill="#333"))
        # 异常标记
        row = per_tooth.get(k, {})
        flags = row.get("flags", [])
        if flags:
            parts.append(_circle(p, 8.0, stroke="#d62728", width=1.6))
            parts.append(_text((p[0], p[1] - 12),
                               ",".join(flags)[:18], 8, fill="#d62728"))
        if k == worst:
            d = 7.0
            parts.append(_polygon(
                [(p[0], p[1] - d), (p[0] + d, p[1]),
                 (p[0], p[1] + d), (p[0] - d, p[1])],
                "#e08a00", fill="rgba(224,138,0,0.35)", width=1.6))

    # 偏心向量（红箭头，从轮心出发，与径向偏差同比例）
    if ecc > 1e-9:
        p_ecc = (CX + ecc_x * scale_r, CY - ecc_y * scale_r)
        parts.append(_arrow((CX, CY), p_ecc, "#d62728", 2.2))
        parts.append(_text((p_ecc[0] + 8, p_ecc[1] - 8),
                           f"e={ecc:.3f} mm", 10, anchor="start",
                           fill="#d62728"))

    # 图例
    lx, ly = 30.0, H - 210.0
    legend = [
        _text((lx, ly), "图例", 13, anchor="start", fill="#111"),
        _line((lx, ly + 18), (lx + 26, ly + 18), "#999", 1.0, "5 4"),
        _text((lx + 34, ly + 22), "名义齿尖圆", 11, anchor="start"),
        _circle((lx + 13, ly + 40), 3.0, fill="#1f6fb2"),
        _text((lx + 34, ly + 44),
              f"齿尖径向偏差（1 mm = {scale_r:.0f} px）", 11,
              anchor="start"),
        _line((lx + 4, ly + 62), (lx + 22, ly + 62), "#2c8c3f", 1.6),
        _text((lx + 34, ly + 66),
              f"累计节距偏差（角向 ×{scale_a:.1f}）", 11, anchor="start"),
        _circle((lx + 13, ly + 84), 7.0, stroke="#d62728", width=1.4),
        _text((lx + 34, ly + 88), "异常齿（穿透/回退/缺拍/离群）", 11,
              anchor="start"),
    ]
    if ecc > 1e-9:
        legend.append(_arrow((lx + 4, ly + 106), (lx + 22, ly + 106),
                             "#d62728", 2.0, 6.0))
        legend.append(_text((lx + 34, ly + 110), "轮心偏心向量", 11,
                            anchor="start"))
    parts.extend(legend)

    # 摘要：最差齿 / 峰峰值 / 分类
    sx, sy = W - 330.0, H - 210.0
    stats = {s["metric"]: s for s in analysis["stats"]}
    names = {"lock_deg": "锁量", "lift_deg": "升角", "drop_deg": "落角",
             "recoil_deg": "回退", "dead_clearance_mm": "静止间隙"}
    lines = [f"最差齿: {worst}"]
    for m, label in names.items():
        st = stats.get(m)
        if st and st["p2p"] == st["p2p"]:  # not NaN
            lines.append(f"{label}峰峰值 {st['p2p']:.3f}"
                         f"（最差齿 {st['worst_tooth']}）")
    dom = cls["dominant"]
    dom_name = {"single_tooth": "单齿缺陷", "eccentricity": "轮心偏心",
                "cumulative_index": "累计分度误差", "none": "无"}[dom]
    lines.append(f"主导误差机理: {dom_name}")
    parts.append(_text((sx, sy), "摘要", 13, anchor="start", fill="#111"))
    for i, s in enumerate(lines):
        parts.append(_text((sx, sy + 20 + i * 17), s, 11, anchor="start",
                           fill="#333"))

    parts.append("</svg>")
    return "".join(parts)
