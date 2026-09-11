# -*- coding: utf-8 -*-
"""平面几何：坐标变换、齿尖圆/瓦面（直线或圆弧）求交、多边形穿透。

世界坐标中：
    wheel center  O, anchor center A
锚系向量 v（相对 A, theta=0 的名义姿态）转到世界系：
    R(theta) @ v + A

轮齿 k 的齿尖（轮转角 psi，旋转方向 s=+1 ccw / -1 cw）：
    T_k(psi) = O + Rp * ( cos(s*psi + k*dphi), sin(...) )

瓦面支持两种：
* 直线面（冲面）：端点 a, b 的线段；
* 圆弧面（死节锁面）：给定弧心 c（锚系，通常即锚轴原点），a->b 圆弧。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

D2R = math.pi / 180.0
R2D = 180.0 / math.pi


def rot(theta: float, v: np.ndarray) -> np.ndarray:
    c, sn = math.cos(theta), math.sin(theta)
    return np.array([c * v[0] - sn * v[1], sn * v[0] + c * v[1]])


def ang_between(a: float, b: float, ccw: bool) -> float:
    """从极角 a 到 b 沿指定方向的有向跨度（弧度）。"""
    d = (b - a) % (2 * math.pi)
    return d if ccw else -((-d) % (2 * math.pi))


@dataclass
class FaceGeom:
    pallet: int
    kind: str
    a: np.ndarray
    b: np.ndarray
    d: np.ndarray
    length: float
    u: np.ndarray
    is_arc: bool = False
    arc_c: Optional[np.ndarray] = None
    ra: float = 0.0
    aa: float = 0.0
    ab: float = 0.0
    arc_ccw: bool = True
    span: float = 0.0       # 弧 a->b 的有向角跨度

    def point(self, t: float) -> np.ndarray:
        if not self.is_arc:
            return self.a + self.d * t
        ang = self.aa + self.span * t
        return self.arc_c + self.ra * np.array(
            [math.cos(ang), math.sin(ang)])

    def sample(self, n: int = 16) -> np.ndarray:
        ts = np.linspace(0.0, 1.0, n)
        return np.array([self.point(t) for t in ts])


@dataclass
class PalletGeom:
    idx: int
    name: str
    lock: FaceGeom
    impulse: FaceGeom
    triangle: Tuple[np.ndarray, np.ndarray, np.ndarray]

    def faces(self) -> Tuple[FaceGeom, FaceGeom]:
        return self.lock, self.impulse


@dataclass
class WheelGeom:
    teeth: int
    Rp: float
    Rf: float
    half_tip: float
    pitch: float
    s: int

    def tip_angle(self, k: int, psi: float) -> float:
        return self.s * psi + k * self.pitch

    def tip(self, k: int, psi: float, O: np.ndarray) -> np.ndarray:
        a = self.tip_angle(k, psi)
        return O + self.Rp * np.array([math.cos(a), math.sin(a)])

    def tooth_polygon(self, k: int, psi: float,
                      O: np.ndarray) -> np.ndarray:
        """齿的近似多边形：根圆两点 + 齿尖构成的三角齿。"""
        a0 = self.tip_angle(k, psi)
        tip = O + self.Rp * np.array([math.cos(a0), math.sin(a0)])
        # 齿尖两侧斜线与根圆的交点
        pts = []
        for side in (-1, 1):
            ang = a0 + side * self.half_tip
            cdir = np.array([math.cos(ang), math.sin(ang)])
            rel = tip - O
            b2 = float(np.dot(rel, cdir))
            disc = b2 * b2 - (float(np.dot(rel, rel)) - self.Rf ** 2)
            tau = b2 - math.sqrt(max(disc, 0.0))
            pts.append(tip - tau * cdir)
        return np.array([pts[0], tip, pts[1]])


def build_pallets(inp) -> List[PalletGeom]:
    out: List[PalletGeom] = []
    for idx, p in enumerate((inp.entry_pallet, inp.exit_pallet)):
        lk = _face(idx, "lock", p.lock)
        im = _face(idx, "impulse", p.impulse)
        tri = (lk.a, lk.b, im.b)
        out.append(PalletGeom(idx, p.name, lk, im, tri))
    return out


def _face(pallet: int, kind: str, f) -> FaceGeom:
    a = np.array([f.a.x, f.a.y], dtype=float)
    b = np.array([f.b.x, f.b.y], dtype=float)
    d = b - a
    L = float(np.hypot(*d))
    fg = FaceGeom(pallet, kind, a, b, d, L, d / L)
    if f.arc_center is not None:
        c = np.array([f.arc_center.x, f.arc_center.y], dtype=float)
        ra = float(np.hypot(*(a - c)))
        rb = float(np.hypot(*(b - c)))
        aa, ab = math.atan2(a[1] - c[1], a[0] - c[0]), \
            math.atan2(b[1] - c[1], b[0] - c[0])
        # 选较短弧
        ccw = ang_between(aa, ab, True) <= math.pi
        span = ang_between(aa, ab, ccw)
        fg.is_arc, fg.arc_c, fg.ra = True, c, 0.5 * (ra + rb)
        fg.aa, fg.ab, fg.arc_ccw, fg.span = aa, ab, ccw, span
    return fg


def world_face(fg: FaceGeom, theta: float,
               A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return A + rot(theta, fg.a), A + rot(theta, fg.b)


def _circle_segment(O, R, P0, P1) -> List[Tuple[float, np.ndarray]]:
    d = P1 - P0
    f = P0 - O
    a = float(np.dot(d, d))
    if a < 1e-14:
        return []
    b = 2.0 * float(np.dot(f, d))
    c = float(np.dot(f, f)) - R * R
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    disc = math.sqrt(disc)
    res = []
    for t in ((-b - disc) / (2 * a), (-b + disc) / (2 * a)):
        if -1e-9 <= t <= 1 + 1e-9:
            tt = min(1.0, max(0.0, t))
            res.append((tt, P0 + d * tt))
    return res


def arc_intersection_t(fg: FaceGeom, theta: float, A: np.ndarray,
                       O: np.ndarray, R: float) -> List[Tuple[float, np.ndarray]]:
    """圆弧面与齿尖圆求交（返回 t∈[0,1]）。t=0 瓦跟, t=1 瓦角。"""
    if not fg.is_arc:
        P0, P1 = world_face(fg, theta, A)
        return _circle_segment(O, R, P0, P1)
    cw = A + rot(theta, fg.arc_c)
    dvec = O - cw
    d = float(np.hypot(*dvec))
    out = []
    if d > R + fg.ra + 1e-12 or d < abs(R - fg.ra) - 1e-12 or d < 1e-12:
        return out
    aa = (R * R - fg.ra * fg.ra + d * d) / (2 * d)
    h2 = R * R - aa * aa
    if h2 < 0:
        h2 = 0.0
    h = math.sqrt(h2)
    # aa 是从 O 沿 (C-O) 方向到弦垂足的距离
    base = O + aa * (cw - O) / d
    perp = np.array([-(cw - O)[1], (cw - O)[0]]) / d * h
    for Q in (base + perp, base - perp):
        local = math.atan2(Q[1] - cw[1], Q[0] - cw[0])
        # 锚系极角 = 世界极角 - theta
        la = local - theta
        dpos = (la - fg.aa) % (2 * math.pi)
        eps = 1e-7
        if fg.arc_ccw:
            if -eps <= dpos <= fg.span + eps:
                t = min(1.0, max(0.0, dpos / fg.span))
                out.append((t, Q))
        else:
            # 顺时针弧：有向跨度 span<0，等价于 dpos 在 [2π+span, 2π] 或 =0
            if dpos <= eps or 2 * math.pi + fg.span - eps <= dpos <= 2 * math.pi:
                t = 0.0 if dpos <= eps else dpos / fg.span
                t = min(1.0, max(0.0, t))
                out.append((t, Q))
    return out


# ----------------------------------------------------------------------
# 多边形碰撞
# ----------------------------------------------------------------------
def point_in_poly(q: np.ndarray, poly: np.ndarray, eps: float = 1e-9) -> bool:
    n = len(poly)
    inside = False
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (a[1] > q[1]) != (b[1] > q[1]):
            xcross = a[0] + (b[0] - a[0]) * (q[1] - a[1]) / (b[1] - a[1])
            if xcross > q[0] + eps:
                inside = not inside
    return inside


def poly_overlap_depth(polyA: np.ndarray, polyB: np.ndarray) -> float:
    """两凸多边形相交返回近似最大内深（mm），否则 0。"""
    for poly in (polyA, polyB):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            axis = np.array([-e[1], e[0]])
            na = float(np.hypot(*axis))
            if na < 1e-14:
                continue
            axis /= na
            pa = polyA @ axis
            pb = polyB @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return 0.0
    depth = 0.0
    for q in polyA:
        if point_in_poly(q, polyB):
            depth = max(depth, _inward_depth(q, polyB))
    for q in polyB:
        if point_in_poly(q, polyA):
            depth = max(depth, _inward_depth(q, polyA))
    return max(depth, 1e-6)


def _inward_depth(q: np.ndarray, poly: np.ndarray) -> float:
    best = 0.0
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        L = float(np.hypot(*e))
        if L < 1e-14:
            continue
        nrm = np.array([-e[1], e[0]]) / L
        tt = float(np.dot(q - a, e)) / (L * L)
        if -1e-6 <= tt <= 1 + 1e-6:
            best = max(best, abs(float(np.dot(q - a, nrm))))
    return best


def pallet_polygon_world(pg: PalletGeom, theta: float,
                         A: np.ndarray) -> np.ndarray:
    """瓦实体多边形（三角形；锁面为弧时沿弧加密）。"""
    verts = []
    lock_pts = pg.lock.sample(12) if pg.lock.is_arc else np.array(
        [pg.lock.a, pg.lock.b])
    for v in lock_pts:
        verts.append(A + rot(theta, v))
    verts.append(A + rot(theta, pg.impulse.b))
    return np.array(verts)
