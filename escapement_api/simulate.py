# -*- coding: utf-8 -*-
"""核心：离散锚角扫描 + brentq 求根的接触顺序追踪。

准静态模型
==========
擒纵轮总是停在“前进方向上第一道障碍”前。给定锚角 θ，沿轮旋转方向，
在不超过一个齿距的窗口内求齿尖圆与瓦面（直线或圆弧）的最早交点，
即轮平衡角 ψ：

* 同瓦面同齿连续交点 —— 锁面段（静止）或冲面段（推进）；
* 接触消失、前方 <1 齿距处落到另一瓦 —— landing，角度差=落角；
* 前方一齿距内无接触 —— not_locked；
* 两瓦同时出现等 ψ 交点 —— double_contact；
* 齿三角形与瓦多边形相交 —— tip_penetration；
* 锁面接触期间 ψ 后退 —— recoil；
* 接触段不按 锁→冲、进瓦↔出瓦 交替 —— order_jump。

离散扫描得到接触段 (run) 后，在锁→冲边界用 brentq 对
``|A+R(θ)瓦角 - O| - Rp = 0`` 求精确解锁角，在冲尾同理求释放角。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from .geometry import (
    D2R, R2D, FaceGeom, PalletGeom, WheelGeom, arc_intersection_t,
    build_pallets, pallet_polygon_world, poly_overlap_depth, rot,
    world_face,
)

EPS_MM = 1e-3
EPS_ANG = 1e-7


@dataclass
class Sample:
    i: int
    theta: float
    psi: Optional[float]
    k: Optional[int]
    pallet: Optional[int]
    face: Optional[str]
    t: float
    point: Optional[np.ndarray]
    penetration: float = 0.0


@dataclass
class Run:
    pallet: int
    face: str
    samples: List[Sample] = field(default_factory=list)

    @property
    def th0(self):
        return self.samples[0].theta

    @property
    def th1(self):
        return self.samples[-1].theta

    @property
    def psi0(self):
        return self.samples[0].psi

    @property
    def psi1(self):
        return self.samples[-1].psi


def psi_for(Q: np.ndarray, k: int, wheel: WheelGeom,
            O: np.ndarray, ref: float) -> float:
    """齿 k 齿尖位于世界点 Q 时的轮转角（展开到 ref 附近）。"""
    alpha = math.atan2(Q[1] - O[1], Q[0] - O[0])
    psi = wheel.s * (alpha - k * wheel.pitch)
    n = round((ref - psi) / wheel.pitch)
    return psi + n * wheel.pitch


class Simulation:
    def __init__(self, inp):
        self.inp = inp
        self.O = np.array([inp.wheel_center.x, inp.wheel_center.y])
        self.A = np.array([inp.anchor_center.x, inp.anchor_center.y])
        self.wheel = WheelGeom(
            inp.wheel.teeth, inp.wheel.pitch_radius,
            inp.wheel.root_radius, inp.wheel.tip_half_angle * D2R,
            2 * math.pi / inp.wheel.teeth,
            1 if inp.rotation == "ccw" else -1)
        self.pallets: List[PalletGeom] = build_pallets(inp)
        sw = inp.swing
        self.th_min = sw.theta_min * D2R
        self.th_max = sw.theta_max * D2R
        self.th_start = (sw.theta_start * D2R
                         if sw.theta_start is not None else self.th_min)
        self.psi_start = sw.initial_wheel_phase * D2R
        self.step = inp.step_deg * D2R
        self.anomalies: List[dict] = []

    # ------------------------------------------------------------------
    # 接触查询
    # ------------------------------------------------------------------
    def _face_hits(self, theta: float, ref_psi: float,
                   window: float, ahead: bool):
        """枚举所有瓦面交点对应的候选齿接触。

        返回 [{pallet, face, t, Q, k, psi}]，按 ψ 排序。
        """
        cands = []
        for pg in self.pallets:
            for fg in pg.faces():
                for t, Q in arc_intersection_t(
                        fg, theta, self.A, self.O, self.wheel.Rp):
                    alpha = math.atan2(Q[1] - self.O[1], Q[0] - self.O[0])
                    k0 = int(round((alpha - self.wheel.s * ref_psi)
                                   / self.wheel.pitch))
                    for kk in (k0 - 1, k0, k0 + 1):
                        psi = psi_for(Q, kk, self.wheel, self.O, ref_psi)
                        if ahead:
                            if not (ref_psi + EPS_ANG < psi
                                    <= ref_psi + window):
                                continue
                        else:
                            if not (ref_psi - window <= psi
                                    <= ref_psi + window):
                                continue
                        cands.append(dict(pallet=pg.idx, face=fg.kind,
                                          t=t, Q=Q, k=kk, psi=psi))
        cands.sort(key=lambda c: c["psi"])
        return cands

    def _continue_contact(self, theta: float, prev: Sample):
        """同瓦面同齿的连续接触（允许锁面上小幅度回退）。"""
        fg = self.pallets[prev.pallet].lock if prev.face == "lock" \
            else self.pallets[prev.pallet].impulse
        hits = arc_intersection_t(fg, theta, self.A, self.O,
                                  self.wheel.Rp)
        best = None
        for t, Q in hits:
            psi = psi_for(Q, prev.k, self.wheel, self.O, prev.psi)
            if abs(psi - prev.psi) > self.wheel.pitch * 0.6:
                continue
            if best is None or abs(psi - prev.psi) < abs(
                    best["psi"] - prev.psi):
                best = dict(pallet=prev.pallet, face=prev.face, t=t, Q=Q,
                            k=prev.k, psi=psi)
        return best

    def _other_pallet_hits(self, theta: float, psi: float, pallet: int):
        """另一瓦在同一 ψ（±线公差）上是否也有齿接触。"""
        hits = []
        for pg in self.pallets:
            if pg.idx == pallet:
                continue
            for fg in pg.faces():
                for t, Q in arc_intersection_t(
                        fg, theta, self.A, self.O, self.wheel.Rp):
                    alpha = math.atan2(Q[1] - self.O[1], Q[0] - self.O[0])
                    k = int(round((alpha - self.wheel.s * psi)
                                  / self.wheel.pitch))
                    for kk in (k - 1, k, k + 1):
                        qpsi = psi_for(Q, kk, self.wheel, self.O, psi)
                        if abs(qpsi - psi) * self.wheel.Rp < EPS_MM * 3:
                            hits.append((pg.idx, fg.kind, Q))
        return hits

    def _penetration(self, theta: float, psi: float):
        worst, detail = 0.0, None
        kc = int(round((-self.wheel.s * psi) / self.wheel.pitch))
        tris = [(pg.idx, pallet_polygon_world(pg, theta, self.A))
                for pg in self.pallets]
        for k in range(kc - 2, kc + 3):
            poly = self.wheel.tooth_polygon(k, psi, self.O)
            for pid, tri in tris:
                d = poly_overlap_depth(poly, tri)
                if d > worst:
                    worst, detail = d, pid
        return worst, detail

    def _min_gap_other(self, theta: float, psi: float, pallet: int):
        """锁住时对面瓦到最近齿尖的最小间隙。"""
        other = next(p for p in self.pallets if p.idx != pallet)
        best = float("inf")
        m = int(round(math.pi / self.wheel.pitch)) + 2
        kc = int(round((-self.wheel.s * psi) / self.wheel.pitch))
        for k in range(kc - m, kc + m + 1):
            T = self.wheel.tip(k, psi, self.O)
            for fg in other.faces():
                pts = arc_intersection_t(fg, theta, self.A, self.O,
                                         self.wheel.Rp)
                # 间隙 = 齿尖到面的距离（圆弧面按到弧心距离换算）
                if fg.is_arc:
                    cw = self.A + rot(theta, fg.arc_c)
                    d = abs(math.hypot(*(T - cw)) - fg.ra)
                else:
                    P0, P1 = world_face(fg, theta, self.A)
                    d, _ = _point_segment(T, P0, P1)
                best = min(best, d)
        return best

    # ------------------------------------------------------------------
    # 扫描
    # ------------------------------------------------------------------
    def run(self):
        up = np.arange(self.th_start, self.th_max + 0.5 * self.step,
                       self.step)
        down = np.arange(self.th_max - self.step,
                         self.th_min - 0.5 * self.step, -self.step)
        back = np.arange(self.th_min + self.step,
                         self.th_start + 0.5 * self.step, self.step)
        thetas = np.concatenate([up, down, back])

        samples: List[Sample] = []
        first = self._face_hits(thetas[0], self.psi_start,
                                self.wheel.pitch, ahead=True)
        first = [c for c in first if c["psi"] >= self.psi_start - EPS_ANG]
        if not first:
            first = self._face_hits(thetas[0], self.psi_start,
                                    self.wheel.pitch, ahead=False)
            first = [c for c in first if c["psi"] >= self.psi_start - EPS_ANG]
        if not first:
            self.anomalies.append(dict(
                code="bad_geometry", theta=thetas[0], psi=None,
                pallet=None, face=None, point=None,
                detail="起始锚角处没有任何齿尖与瓦面接触，机构无法锁住",
                value=None, _key=("bad_geometry", None, None)))
            cur = Sample(-1, thetas[0], self.psi_start, 0, None, None,
                         0.0, None)
        else:
            c = first[0]
            cur = Sample(-1, thetas[0], c["psi"], c["k"], c["pallet"],
                         c["face"], c["t"], c["Q"])

        pitch = self.wheel.pitch
        for i, th in enumerate(thetas):
            if i == 0:
                samples.append(cur)
                continue

            if cur.pallet is None:
                cands = self._face_hits(th, cur.psi, pitch, ahead=True)
                cands = [c for c in cands if c["psi"] > cur.psi + EPS_ANG]
                if cands:
                    c = cands[0]
                    cur = Sample(i, th, c["psi"], c["k"], c["pallet"],
                                 c["face"], c["t"], c["Q"])
                    samples.append(cur)
                    continue
                samples.append(Sample(i, th, cur.psi, cur.k, None, None,
                                      0.0, None))
                continue

            nxt = self._continue_contact(th, cur)
            landed = None
            if nxt is None:
                cands = self._face_hits(th, cur.psi, pitch * 0.999,
                                        ahead=True)
                other = [c for c in cands if c["pallet"] != cur.pallet]
                same = [c for c in cands if c["pallet"] == cur.pallet]
                if other:
                    landed = other[0]
                    drop = landed["psi"] - cur.psi
                    if drop > pitch * 0.95:
                        self._anom("order_jump", th, landed["psi"],
                                   landed["pallet"], landed["face"],
                                   landed["Q"],
                                   f"接触跨过近一个齿距（{drop*R2D:.2f}°）",
                                   drop * R2D)
                elif same:
                    # 同一个齿从一面滑到同瓦另一面，或越过一整齿距
                    c = same[0]
                    jump = c["psi"] - cur.psi
                    if jump > pitch * 0.6:
                        self._anom("order_jump", th, c["psi"], c["pallet"],
                                   c["face"], c["Q"],
                                   "接触在同一瓦上跳过一个齿距，落瓦失败",
                                   jump * R2D)
                    landed = c
                else:
                    self._anom("not_locked", th, cur.psi, cur.pallet,
                               cur.face, cur.point,
                               "齿尖脱离瓦面后一个齿距内未被任何瓦接住",
                               None)
                    samples.append(Sample(i, th, cur.psi, cur.k, None,
                                          None, 0.0, None))
                    continue

            if landed is not None:
                cur = Sample(i, th, landed["psi"], landed["k"],
                             landed["pallet"], landed["face"], landed["t"],
                             landed["Q"])
            else:
                cur = Sample(i, th, nxt["psi"], nxt["k"], nxt["pallet"],
                             nxt["face"], nxt["t"], nxt["Q"])

            # 回退（锁面上 ψ 随 θ 而后移）
            prev = samples[-1]
            if cur.face == "lock" and prev.psi is not None \
                    and cur.psi < prev.psi - EPS_ANG:
                self._anom_recoil(cur, prev)

            # 双瓦同时接触
            for pid, kind, Q in self._other_pallet_hits(
                    th, cur.psi, cur.pallet):
                self._anom("double_contact", th, cur.psi, pid, kind, Q,
                           f"进/出瓦同时接触（{pid} 号瓦 {kind} 面）", None)

            # 穿透
            pen, pid = self._penetration(th, cur.psi)
            cur.penetration = pen
            if pen > EPS_MM:
                self._anom("tip_penetration", th, cur.psi, pid,
                           cur.face, cur.point,
                           f"齿/瓦实体穿透 {pen:.3f} mm", pen)
            samples.append(cur)

        runs = self._build_runs(samples)
        runs = self._merge_seam(runs)
        self._check_sequence(runs)
        events, beats = self._postprocess(runs, thetas)
        return self._assemble(samples, runs, events, beats, thetas)

    # ------------------------------------------------------------------
    def _build_runs(self, samples):
        runs: List[Run] = []
        for s in samples:
            if s.pallet is None:
                runs.append(Run(-1, "none", [s]))
                continue
            if runs and runs[-1].pallet == s.pallet \
                    and runs[-1].face == s.face:
                runs[-1].samples.append(s)
            else:
                runs.append(Run(s.pallet, s.face, [s]))
        return runs

    def _merge_seam(self, runs):
        """扫描窗口首尾是同一连续锁面段时，合并（循环接缝）。"""
        contact = [r for r in runs if r.face != "none"]
        if len(contact) >= 2:
            fst, lst = contact[0], contact[-1]
            if fst.pallet == lst.pallet and fst.face == lst.face:
                # ψ 连续性检查（跨过 0 角度索引，只看齿号）
                if lst.samples[-1].k == fst.samples[0].k:
                    fst.samples = lst.samples + fst.samples
                    runs = [r for r in runs if r is not lst]
        return runs

    def _anom(self, code, theta, psi, pallet, face, point, detail, value,
              key=None):
        k = key or (code, pallet, face)
        for a in self.anomalies:
            if a.get("_key") == k and abs(a["theta"] - theta) * R2D < 2.0:
                if value is not None and (a["value"] is None
                                          or value > a["value"]):
                    a.update(theta=theta, psi=psi, point=point,
                             detail=detail, value=value)
                return
        self.anomalies.append(dict(code=code, theta=theta, psi=psi,
                                   pallet=pallet, face=face, point=point,
                                   detail=detail, value=value, _key=k))

    def _anom_recoil(self, cur, prev):
        amount = prev.psi - cur.psi
        for a in self.anomalies:
            if a["code"] == "recoil" and a["pallet"] == cur.pallet \
                    and abs(a["theta"] - cur.theta) * R2D < 3.0:
                if amount * R2D > (a["value"] or 0):
                    a.update(theta=cur.theta, psi=cur.psi,
                             point=cur.point,
                             detail=f"锁面推动轮回退 {amount*R2D:.3f}°",
                             value=amount * R2D)
                return
        self.anomalies.append(dict(
            code="recoil", theta=cur.theta, psi=cur.psi,
            pallet=cur.pallet, face=cur.face, point=cur.point,
            detail=f"锁面推动轮回退 {amount*R2D:.3f}°",
            value=amount * R2D,
            _key=("recoil", cur.pallet, cur.face)))

    def _check_sequence(self, runs):
        expected = "lock"
        prev_pallet = None
        contact = [r for r in runs if r.face != "none"]
        for r in contact:
            if r.face != expected:
                self._anom("order_jump", r.th0, r.psi0, r.pallet, r.face,
                           r.samples[0].point,
                           f"接触次序跳变：期望 {expected} 段，实际 "
                           f"{r.pallet} 号瓦 {r.face}", None)
            if expected == "lock" and prev_pallet is not None \
                    and r.pallet == prev_pallet:
                self._anom("order_jump", r.th0, r.psi0, r.pallet, r.face,
                           r.samples[0].point,
                           "连续两次锁在同一瓦上，落瓦次序错误", None)
            if r.face == "lock":
                prev_pallet = r.pallet
            expected = "impulse" if expected == "lock" else "lock"
        if len(contact) < 4:
            last = contact[-1].samples[-1] if contact else None
            self._anom("not_locked",
                       last.theta if last else 0.0,
                       last.psi if last else None, None, None, None,
                       f"一个锚周期内只追踪到 {len(contact)} 个接触段"
                       "（应≥4：进瓦锁/冲、出瓦锁/冲）", None,
                       key=("few_runs", None, None))

    # ------------------------------------------------------------------
    # 边界求根
    # ------------------------------------------------------------------
    def _circle_root(self, anchor_vec, lo, hi, sign=None):
        def f(th):
            Q = self.A + rot(th, anchor_vec)
            return math.hypot(Q[0] - self.O[0], Q[1] - self.O[1]) \
                - self.wheel.Rp
        try:
            if f(lo) * f(hi) > 0:
                return None
            return brentq(f, lo, hi, xtol=1e-10, rtol=1e-12)
        except ValueError:
            return None

    def _psi_on_face(self, theta, pallet, kind, k, ref):
        fg = self.pallets[pallet].lock if kind == "lock" \
            else self.pallets[pallet].impulse
        hits = arc_intersection_t(fg, theta, self.A, self.O,
                                  self.wheel.Rp)
        best = None
        for t, Q in hits:
            psi = psi_for(Q, k, self.wheel, self.O, ref)
            if best is None or abs(psi - ref) < abs(best[0] - ref):
                best = (psi, Q)
        return best

    def _first_ahead(self, theta, psi, want_pallet):
        cands = self._face_hits(theta, psi, self.wheel.pitch * 1.05,
                                ahead=True)
        sel = [c for c in cands if c["pallet"] == want_pallet]
        return sel[0] if sel else (cands[0] if cands else None)

    # ------------------------------------------------------------------
    def _postprocess(self, runs, thetas):
        contact = [r for r in runs if r.face != "none"]
        events, beats = [], []
        if not contact:
            return events, beats

        first = contact[0].samples[0]
        events.append(dict(kind="initial_lock", theta=first.theta,
                           psi=first.psi, pallet=first.pallet,
                           face="lock", point=first.point,
                           note="初始静止锁住"))

        # 每段补前后一个网格做括号
        grid = list(thetas)

        def bracket(run):
            i0, i1 = run.samples[0].i, run.samples[-1].i
            lo = grid[i0 - 1] if i0 - 1 >= 0 else grid[i0]
            hi = grid[i1 + 1] if i1 + 1 < len(grid) else grid[i1]
            return lo, hi

        n = len(contact)
        for i, r in enumerate(contact):
            if r.face != "lock":
                continue
            r_imp = contact[(i + 1) % n]
            same_next = (r_imp.pallet == r.pallet
                         and r_imp.face == "impulse")
            r_prev_imp = contact[(i - 1) % n]
            same_prev = (r_prev_imp.pallet == r.pallet
                         and r_prev_imp.face == "impulse")

            # 落瓦：前一冲面段的冲尾释放角
            th_land, psi_land = r.th0, r.psi0
            if same_prev:
                lo, hi = bracket(r_prev_imp)
                th_rel = self._circle_root(
                    self.pallets[r_prev_imp.pallet].impulse.b, lo, hi)
                if th_rel is not None:
                    rel = self._psi_on_face(
                        th_rel, r_prev_imp.pallet, "impulse",
                        r_prev_imp.samples[-1].k, r_prev_imp.psi1)
                    if rel:
                        land = self._first_ahead(th_rel, rel[0], r.pallet)
                        if land:
                            events.append(dict(
                                kind="release", theta=th_rel,
                                psi=rel[0], pallet=r_prev_imp.pallet,
                                face="impulse", point=rel[1],
                                note="齿尖离开冲尾"))
                            events.append(dict(
                                kind="landing", theta=th_rel,
                                psi=land["psi"], pallet=r.pallet,
                                face="lock", point=land["Q"],
                                note=f"落瓦，落角 "
                                f"{(land['psi']-rel[0])*R2D:.3f}°"))
                            events.append(dict(
                                kind="drop", theta=th_rel,
                                psi=land["psi"], pallet=r.pallet,
                                face="lock", point=land["Q"],
                                note=f"落角 "
                                f"{(land['psi']-rel[0])*R2D:.3f}°"))
                            th_land, psi_land = th_rel, land["psi"]

            # 解锁角：瓦角进入齿尖圆
            th_unlock, psi_unlock = r.th1, r.psi1
            if same_next:
                lo, hi = bracket(r)
                root = self._circle_root(
                    self.pallets[r.pallet].lock.b, lo, hi)
                if root is not None:
                    hit = self._psi_on_face(
                        root, r.pallet, "lock",
                        r.samples[-1].k, r.psi1)
                    if hit:
                        th_unlock, psi_unlock = root, hit[0]
            events.append(dict(
                kind="unlock", theta=th_unlock, psi=psi_unlock,
                pallet=r.pallet, face="lock",
                point=self.A + rot(
                    th_unlock, self.pallets[r.pallet].lock.b),
                note="齿尖过瓦角，锁面解锁进入冲面"))

            # 释放（本瓦冲面尾）
            th_release = psi_release = None
            if same_next:
                lo, hi = bracket(r_imp)
                th_release = self._circle_root(
                    self.pallets[r.pallet].impulse.b, lo, hi) \
                    or r_imp.th1
                rel = self._psi_on_face(
                    th_release, r.pallet, "impulse",
                    r_imp.samples[-1].k, r_imp.psi1)
                if rel:
                    psi_release = rel[0]

            lock_deg = _ang_dist(th_land, th_unlock, self.th_min,
                                 self.th_max) * R2D
            lift_deg = _ang_dist(th_unlock, th_release,
                                 self.th_min, self.th_max) * R2D \
                if th_release is not None else 0.0
            wheel_imp = ((psi_release - psi_unlock) * R2D
                         if psi_release is not None else 0.0)

            ps = [s.psi for s in r.samples if s.psi is not None]
            recoil = (max(ps) - min(ps)) * R2D if ps else 0.0
            lock_rest = self._lock_rest_depth(r, th_land, psi_land)
            clearance = self._dead_clearance(r)
            ecc = self._lock_eccentricity(r)

            beats.append(dict(
                pallet=r.pallet,
                pallet_name=self.pallets[r.pallet].name,
                lock_deg=lock_deg,
                lock_at_rest_mm=lock_rest,
                lift_deg=lift_deg,
                wheel_impulse_deg=wheel_imp,
                drop_deg=float("nan"),
                recoil_deg=max(0.0, recoil),
                advance_deg=0.0,
                dead_clearance_mm=clearance,
                lock_arc_eccentricity_mm=ecc,
                _psi_land=psi_land, _psi_release=psi_release))

        # 落角 / 净推进：按 ψ 顺序配相邻两次落瓦
        beats.sort(key=lambda b: b["_psi_land"])
        for j, b in enumerate(beats):
            if b["_psi_release"] is not None and j + 1 < len(beats):
                nb = beats[j + 1]
                b["drop_deg"] = (nb["_psi_land"]
                                 - b["_psi_release"]) * R2D
                b["advance_deg"] = (nb["_psi_land"]
                                    - b["_psi_land"]) * R2D
        if beats and beats[0]["_psi_release"] is not None \
                and math.isnan(beats[-1]["drop_deg"]):
            # 最后一拍落到下一周期的第一次落瓦
            first_next = beats[0]["_psi_land"] + self.wheel.pitch * \
                round((beats[-1]["_psi_release"] - beats[0]["_psi_land"])
                      / self.wheel.pitch + 0.5)
            beats[-1]["drop_deg"] = (first_next
                                     - beats[-1]["_psi_release"]) * R2D
            beats[-1]["advance_deg"] = (first_next
                                        - beats[-1]["_psi_land"]) * R2D
        return events, beats

    def _lock_rest_depth(self, run, th_land, psi_land):
        """落瓦静止时齿尖越过瓦角沿锁面的深度（mm）。t: 瓦跟0→瓦角1。"""
        fg = self.pallets[run.pallet].lock
        s0 = run.samples[0]
        hits = arc_intersection_t(fg, th_land, self.A, self.O,
                                  self.wheel.Rp)
        t = s0.t
        best = None
        for tt, Q in hits:
            qpsi = psi_for(Q, s0.k, self.wheel, self.O, psi_land)
            if best is None or abs(qpsi - psi_land) < abs(
                    best[0] - psi_land):
                best = (qpsi, tt)
        if best:
            t = best[1]
        return max(0.0, (1.0 - t) * (abs(fg.ra * fg.span)
                                     if fg.is_arc else fg.length))

    def _dead_clearance(self, run):
        ss = run.samples[::max(1, len(run.samples) // 10)]
        if run.samples[-1] not in ss:
            ss.append(run.samples[-1])
        worst = float("inf")
        for s in ss:
            worst = min(worst, self._min_gap_other(
                s.theta, s.psi, run.pallet))
        return worst

    def _lock_eccentricity(self, run):
        if run.face != "lock":
            return 0.0
        fg = self.pallets[run.pallet].lock
        radii = []
        for s in run.samples:
            if s.point is not None:
                if fg.is_arc:
                    radii.append(abs(math.hypot(
                        *(s.point - self.A)) - fg.ra))
                else:
                    radii.append(math.hypot(
                        s.point[0] - self.A[0], s.point[1] - self.A[1]))
        if not radii:
            return 0.0
        return max(radii) - (0.0 if fg.is_arc else min(radii))

    # ------------------------------------------------------------------
    def _assemble(self, samples, runs, events, beats, thetas):
        max_pen = max((s.penetration for s in samples), default=0.0)
        imp = [b["wheel_impulse_deg"] for b in beats]
        tot = sum(imp)
        share = [v / tot for v in imp] if tot > 1e-9 \
            else [0.0] * len(imp)
        drops = [b["drop_deg"] for b in beats
                 if not math.isnan(b["drop_deg"])]
        clears = [b["dead_clearance_mm"] for b in beats
                  if not math.isnan(b["dead_clearance_mm"])]
        metrics = dict(
            beats=beats,
            impulse_share=share,
            mean_drop_deg=float(np.mean(drops)) if drops else float("nan"),
            max_recoil_deg=max((b["recoil_deg"] for b in beats),
                               default=0.0),
            min_dead_clearance_mm=min(clears, default=float("nan")),
            total_advance_deg=sum(
                b["advance_deg"] for b in beats
                if not math.isnan(b["advance_deg"])),
            max_penetration_mm=max_pen)

        stride = max(1, len(samples) // 240)
        trace = [dict(
            pallet=s.pallet, face=s.face,
            theta_deg=s.theta * R2D,
            wheel_psi_deg=(s.psi or 0.0) * R2D,
            point=dict(x=float(s.point[0]), y=float(s.point[1])),
            penetration_mm=s.penetration)
            for s in samples[::stride] if s.pallet is not None]

        codes = {a["code"] for a in self.anomalies}
        hard = {"tip_penetration", "double_contact", "not_locked",
                "order_jump", "bad_geometry"}
        status = "fail" if codes & hard else (
            "warning" if codes or max_pen > EPS_MM else "ok")
        msg = {"ok": "一齿周期接触顺序完整，无冲突",
               "warning": "存在回退等警告，见诊断",
               "fail": "发现致命接触冲突，见诊断"}[status]

        return dict(status=status, metrics=metrics, events=events,
                    anomalies=self.anomalies, trace=trace, message=msg,
                    _thetas=thetas, _samples=samples, _runs=runs)


def _ang_dist(t0, t1, lo, hi):
    """锚角距离；t0/t1 可能跨越反向扫描，用去折叠后的路径长。"""
    d = abs(t1 - t0)
    span = hi - lo
    if d > span:
        d = 2 * span - d
    return abs(d)


def _point_segment(Q, P0, P1):
    d = P1 - P0
    L2 = float(np.dot(d, d))
    t = float(np.dot(Q - P0, d)) / L2
    tc = min(1.0, max(0.0, t))
    foot = P0 + d * tc
    return float(np.hypot(*(Q - foot))), tc


def simulate(inp) -> dict:
    return Simulation(inp).run()
