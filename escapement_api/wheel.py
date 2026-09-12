# -*- coding: utf-8 -*-
"""全轮逐齿误差分析。

误差模型
========
误差表按齿号记录：节距偏差 δ_k（齿 k 到齿 k+1 的实际角距减名义齿距）、
齿尖径向偏差 Δr_k、齿尖半角实测值，以及轮心偏心向量 e（随轮旋转）。
未录齿沿用名义值。整轮必须闭合：Σδ_k = 0（由 Pydantic 校验）。

轮心偏心折算：随轮旋转的偏心向量 e 等效于把齿 k 的齿尖向量
``(Rp+Δr_k)·dir(φ_k)`` 平移为 ``(Rp+Δr_k)·dir(φ_k) + e``，即每个齿获得
精确的等效半径 ρ_k 与等效角位置 α_k——折算是精确的，不做一阶近似。
因此整轮接触仿真仍绕固定轮心 O 准静态进行。

分析流程
========
:func:`analyze_wheel` 用 :class:`WheelSimulation`（多锚周期、非循环、
按瓦角区检查穿透）把整轮接触按实际齿序展开，每拍（每个锁面段）追踪
锁住→解锁→冲面→释放→落瓦，再把拍指标按 ``k mod N`` 归并到齿号，
计算峰峰值、周期谐波（FFT）、最差齿，并按输入表与实测序列区分
单齿缺陷 / 轮心偏心 / 累计分度误差。分析不修改传入的输入对象。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import D2R, R2D, WheelGeom, rot
from .simulate import EPS_MM, Simulation

# 逐齿统计的指标（角度度 / 长度 mm）
METRICS = ["lock_deg", "lift_deg", "drop_deg", "recoil_deg",
           "dead_clearance_mm"]
METRIC_NAMES = {
    "lock_deg": "锁量",
    "lift_deg": "升角",
    "drop_deg": "落角",
    "recoil_deg": "回退量",
    "dead_clearance_mm": "静止段间隙",
}
# 谐波分析保留的次数
MAX_HARMONIC = 8
# 单齿缺陷录入阈值（输入侧证据）
DEFECT_RADIAL_MM = 0.005
DEFECT_PITCH_DEG = 0.02
# 回退标记阈值（度）
RECOIL_FLAG_DEG = 0.05


# ----------------------------------------------------------------------
# 逐齿误差轮几何
# ----------------------------------------------------------------------
@dataclass
class ToothedWheelGeom(WheelGeom):
    """带逐齿误差的擒纵轮：齿号与绝对角位置绑定，psi 按整圈展开。"""

    base_phi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    radii: np.ndarray = field(default_factory=lambda: np.zeros(0))
    half_tips: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def phi(self, k: int) -> float:
        n = self.teeth
        return 2.0 * math.pi * (k // n) + float(self.base_phi[k % n])

    def Rp_of(self, k: int) -> float:
        return float(self.radii[k % self.teeth])

    def half_tip_of(self, k: int) -> float:
        return float(self.half_tips[k % self.teeth])

    @property
    def unwrap(self) -> float:
        return 2.0 * math.pi


def build_toothed_wheel(wheel_spec, table, rotation: str
                        ) -> Tuple[ToothedWheelGeom, List[dict]]:
    """由轮参数 + 误差表构造逐齿轮几何（纯函数，不修改入参）。

    返回 (几何, 逐齿有效参数表)。``table`` 为
    :class:`escapement_api.models.WheelErrorTable`。
    """
    n = wheel_spec.teeth
    pitch = 2.0 * math.pi / n
    delta = np.zeros(n)          # 节距偏差（弧度），齿 k -> k+1
    dr = np.zeros(n)             # 齿尖径向偏差（mm）
    half = np.full(n, wheel_spec.tip_half_angle * D2R)
    for e in table.teeth:
        delta[e.tooth] = e.pitch_deviation_deg * D2R
        dr[e.tooth] = e.tip_radial_deviation_mm
        if e.tip_half_angle_deg is not None:
            half[e.tooth] = e.tip_half_angle_deg * D2R

    # 累计角位置：齿 0 为基准，phi_k = k*pitch + Σ_{j<k} δ_j
    phi = np.array([k * pitch for k in range(n)])
    phi[1:] += np.cumsum(delta[:-1])

    # 偏心向量精确折算进逐齿半径/角度
    ex, ey = table.eccentricity.x, table.eccentricity.y
    rp = wheel_spec.pitch_radius
    vx = (rp + dr) * np.cos(phi) + ex
    vy = (rp + dr) * np.sin(phi) + ey
    radii = np.hypot(vx, vy)
    angles = np.arctan2(vy, vx)

    geom = ToothedWheelGeom(
        teeth=n, Rp=rp, Rf=wheel_spec.root_radius,
        half_tip=wheel_spec.tip_half_angle * D2R,
        pitch=pitch, s=1 if rotation == "ccw" else -1,
        base_phi=angles, radii=radii, half_tips=half)
    # 最小齿间角距（含偏心折算）：落瓦搜索窗口必须小于它，
    # 否则解锁瞬间会越过冲面直接落到对面瓦。
    # angles 来自 atan2（非单调），角距按 2π 取模到 [0, 2π)。
    gaps = np.mod(np.diff(np.concatenate([angles, [angles[0]]])),
                  2.0 * math.pi)
    geom.min_gap = float(gaps.min())
    effective = [dict(
        tooth=k,
        radius_mm=float(radii[k]),
        angle_deg=float(angles[k] * R2D),
        pitch_deviation_deg=float(delta[k] * R2D),
        radial_deviation_mm=float(dr[k]),
        tip_half_angle_deg=float(half[k] * R2D),
    ) for k in range(n)]
    return geom, effective


# ----------------------------------------------------------------------
# 多周期接触仿真
# ----------------------------------------------------------------------
class WheelSimulation(Simulation):
    """整轮接触仿真：多锚周期扫描、非循环配对、按瓦角区检查穿透。"""

    def __init__(self, inp, wheel_geom: ToothedWheelGeom,
                 n_segments: int):
        super().__init__(inp)
        self.wheel = wheel_geom
        self.n_segments = n_segments
        self.cyclic = False
        # 落瓦窗口：须小于最小齿间角距（名义轮为 0.999 齿距）
        self.landing_window = min(wheel_geom.pitch * 0.999,
                                  wheel_geom.min_gap * 0.98)
        self.relock_window = wheel_geom.pitch * 1.08
        # 每瓦角 ±1 齿足以覆盖啮合齿，降低穿透检查开销
        self.pen_span = 1
        # 穿透隔点检查（锁段内穿透缓变），加速整轮扫描
        self.pen_check_every = 3

    def _theta_grid(self) -> np.ndarray:
        """交替 上行/下行 的半摆段序列；每段约对应一次滴答。"""
        segs = [np.arange(self.th_start,
                          self.th_max + 0.5 * self.step, self.step)]
        for i in range(self.n_segments - 1):
            if i % 2 == 0:
                seg = np.arange(self.th_max, self.th_min - 0.5 * self.step,
                                -self.step)
            else:
                seg = np.arange(self.th_min, self.th_max + 0.5 * self.step,
                                self.step)
            segs.append(seg[1:])  # 接缝处去掉重复点
        return np.concatenate(segs)

    def _penetration_zones(self, theta: float, psi: float) -> List[int]:
        """两个瓦角附近的齿都需要检查穿透（逐齿归因）。"""
        zones = []
        for pg in self.pallets:
            c = self.A + rot(theta, pg.lock.b)
            alpha = math.atan2(c[1] - self.O[1], c[0] - self.O[0])
            zones.append(int(round(
                (alpha - self.wheel.s * psi) / self.wheel.pitch)))
        return zones

    # 锁面跟部在摆角极限处可能与齿尖恰好相切，数值上接触会丢失几个
    # 样本再恢复；把被短暂“无接触”缝隙打断的同齿同面段重新合并。
    MAX_GAP_SAMPLES = 12

    def _build_runs(self, samples):
        base = super()._build_runs(samples)
        out = []
        i = 0
        while i < len(base):
            r = base[i]
            if r.face == "none":
                i += 1
                continue
            j = i + 1
            while j < len(base):
                k = j
                while k < len(base) and base[k].face == "none":
                    k += 1
                if k == j:  # 相邻接触段，无缝隙
                    break
                n_gap = sum(len(x.samples) for x in base[j:k])
                merged = False
                if k < len(base) and n_gap <= self.MAX_GAP_SAMPLES:
                    r2 = base[k]
                    if (r2.pallet == r.pallet and r2.face == r.face
                            and r2.samples[0].k == r.samples[-1].k
                            and r2.psi0 is not None and r.psi1 is not None
                            and abs(r2.psi0 - r.psi1)
                            < self.wheel.pitch * 0.05):
                        r.samples.extend(r2.samples)
                        j = k + 1
                        merged = True
                if not merged:
                    break
            out.append(r)
            i = j
        return out


# ----------------------------------------------------------------------
# 统计：峰峰值 / 谐波 / 最差齿
# ----------------------------------------------------------------------
def _clean(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def series_stats(metric: str, values: Sequence[float],
                 n: int) -> dict:
    """单指标逐齿序列的均值/峰峰值/最差齿/周期谐波（NaN 插值后 FFT）。"""
    valid = [(i, float(v)) for i, v in enumerate(values)
             if v is not None and math.isfinite(v)]
    if not valid:
        return dict(metric=metric, mean=float("nan"), min=float("nan"),
                    max=float("nan"), p2p=float("nan"), worst_tooth=-1,
                    harmonics=[])
    idx = np.array([i for i, _ in valid], dtype=float)
    xs = np.array([v for _, v in valid], dtype=float)
    mean = float(xs.mean())
    worst_i = max(valid, key=lambda t: abs(t[1] - mean))[0]
    harmonics = []
    if len(valid) >= 4 and n >= 4:
        grid = np.interp(np.arange(n, dtype=float), idx, xs)
        grid = grid - grid.mean()
        spec = np.fft.rfft(grid)
        for h in range(1, min(MAX_HARMONIC, n // 2) + 1):
            amp = 2.0 * abs(spec[h]) / n
            harmonics.append(dict(
                order=h, amplitude=float(amp),
                phase_deg=float(math.degrees(math.atan2(
                    spec[h].imag, spec[h].real)))))
    return dict(metric=metric, mean=mean, min=float(xs.min()),
                max=float(xs.max()), p2p=float(xs.max() - xs.min()),
                worst_tooth=int(worst_i), harmonics=harmonics)


def _residual_outliers(values: Sequence[float], n: int,
                       keep_harmonics: int = 4,
                       sigma_mult: float = 3.0) -> List[Tuple[int, float]]:
    """去掉低次谐波后的残差离群齿（单齿缺陷的实测特征）。

    用 MAD（中位绝对偏差）估计稳健 σ：孤立尖峰不会像普通标准差
    那样把 σ 抬高到吞没自身。
    """
    valid = [(i, float(v)) for i, v in enumerate(values)
             if v is not None and math.isfinite(v)]
    if len(valid) < max(6, keep_harmonics * 2 + 2):
        return []
    idx = np.array([i for i, _ in valid], dtype=float)
    xs = np.array([v for _, v in valid], dtype=float)
    grid = np.interp(np.arange(n, dtype=float), idx, xs)
    centered = grid - grid.mean()
    spec = np.fft.rfft(centered)
    recon = np.zeros(n)
    for h in range(1, min(keep_harmonics, n // 2) + 1):
        amp = 2.0 * abs(spec[h]) / n
        ph = math.atan2(spec[h].imag, spec[h].real)
        recon += amp * np.cos(
            2.0 * math.pi * h * np.arange(n) / n + ph)
    resid = centered - recon
    med = float(np.median(resid))
    sigma = 1.4826 * float(np.median(np.abs(resid - med)))
    p2p = float(xs.max() - xs.min())
    if sigma < 1e-12:
        return []
    out = []
    for k in range(n):
        dev = abs(resid[k] - med)
        if dev <= sigma_mult * sigma or dev <= 0.1 * p2p:
            continue
        # 须为局部极大：单齿缺陷的尖峰，而非其谐波在邻齿的振铃
        left = abs(resid[(k - 1) % n] - med)
        right = abs(resid[(k + 1) % n] - med)
        if dev >= left and dev >= right:
            out.append((k, float(resid[k] - med)))
    return out


# ----------------------------------------------------------------------
# 误差机理分类
# ----------------------------------------------------------------------
def _classify(table, n: int, series: Dict[str, List[float]],
              stats: Dict[str, dict]) -> dict:
    """区分单齿缺陷 / 轮心偏心 / 累计分度误差。

    机理判定以录入误差表为主（实测序列用于确认与量化）：
    * 轮心偏心：偏心向量非零；实测为每转一次正弦（1 次谐波）；
    * 单齿缺陷：少数孤立齿超差（相对其余齿为离群），或去低次谐波
      后的孤立残差尖峰；
    * 累计分度误差：录入节距偏差的累计漂移（平滑低次谐波）。
    三种机理的实测特征在谐波上可能重叠（累计漂移也含 1 次谐波），
    因此是否“存在”由录入表判定，实测能量只决定主导机理与置信度。
    """
    # ---- 输入侧证据 ----
    ecc = math.hypot(table.eccentricity.x, table.eccentricity.y)
    ecc_phase = math.degrees(math.atan2(
        table.eccentricity.y, table.eccentricity.x)) if ecc > 0 else None
    delta = np.zeros(n)
    mags: List[Tuple[int, float]] = []
    for e in table.teeth:
        delta[e.tooth] = e.pitch_deviation_deg
        mag = max(
            abs(e.tip_radial_deviation_mm) / DEFECT_RADIAL_MM,
            abs(e.pitch_deviation_deg) / DEFECT_PITCH_DEG,
            1.0 if e.tip_half_angle_deg is not None else 0.0)
        mags.append((e.tooth, mag))
    cum = np.concatenate([[0.0], np.cumsum(delta)[:-1]])
    excursion = float(np.abs(cum).max())
    excursion_tooth = int(np.abs(cum).argmax()) if excursion > 0 else -1
    # 孤立超差齿：归一化超差 >1 且数量少（多齿普遍超差属系统性分度）
    big = sorted(t for t, m in mags if m > 1.0)
    isolated = big if 0 < len(big) <= max(2, n // 5) else []

    # ---- 实测侧证据 ----
    drop_st = stats.get("drop_deg", {})
    lock_st = stats.get("lock_deg", {})

    def _harm(st, order):
        for h in st.get("harmonics", []):
            if h["order"] == order:
                return h["amplitude"]
        return 0.0

    h1 = max(_harm(drop_st, 1), _harm(lock_st, 1))
    p2p = max(drop_st.get("p2p") or 0.0, lock_st.get("p2p") or 0.0)
    low_energy = sum(_harm(drop_st, o) ** 2 for o in (1, 2, 3, 4))
    outliers: Dict[int, float] = {}
    for m in ("lock_deg", "drop_deg"):
        for k, r in _residual_outliers(series[m], n):
            if abs(r) > abs(outliers.get(k, 0.0)):
                outliers[k] = r

    # ---- 轮心偏心（录入向量判定，实测 H1 确认）----
    ecc_measured = p2p > 1e-9 and h1 >= 0.35 * p2p
    ecc_present = ecc > 1e-6
    ecc_conf = ("high" if ecc_present and ecc_measured else
                "medium" if ecc_present else "none")
    ecc_evidence = []
    if ecc_present:
        ecc_evidence.append(
            f"录入偏心向量 |e|={ecc:.3f} mm，方位 {ecc_phase:.1f}°")
        if ecc_measured:
            ecc_evidence.append(
                f"逐齿序列 1 次谐波幅值 {h1:.3f}，占峰峰值 "
                f"{(h1 / p2p * 100 if p2p else 0):.0f}%")
    else:
        ecc_evidence.append("未录入偏心向量")

    # ---- 单齿缺陷（孤立超差齿 或 实测残差离群）----
    st_teeth = sorted(set(isolated) | set(outliers))
    st_present = bool(st_teeth)
    agree = sorted(set(isolated) & set(outliers))
    st_conf = ("high" if agree else
               "medium" if st_present else "none")
    st_evidence = []
    if isolated:
        st_evidence.append(
            f"录入孤立超差齿 {isolated}（径向>{DEFECT_RADIAL_MM} mm 或"
            f"节距>{DEFECT_PITCH_DEG}° 或半角实测）")
    if big and not isolated:
        st_evidence.append(
            f"{len(big)} 齿普遍超差，按系统性分度处理而非孤立单齿")
    if outliers:
        st_evidence.append(
            f"残差离群齿 {sorted(outliers)}（去 1-4 次谐波后局部极大）")
    if not st_evidence:
        st_evidence.append("未见孤立单齿超差")
    st_amp = max((abs(r) for r in outliers.values()), default=None)

    # ---- 累计分度误差（录入节距累计漂移判定）----
    ci_present = excursion > table.closure_tol_deg
    ci_measured = low_energy > 0 and p2p > 1e-9 \
        and math.sqrt(low_energy) >= 0.25 * p2p
    ci_conf = ("high" if ci_present and ci_measured else
               "medium" if ci_present else "none")
    ci_evidence = []
    if ci_present:
        ci_evidence.append(
            f"录入节距偏差累计漂移 {excursion:.3f}°"
            f"（齿 {excursion_tooth} 处最大）")
        if ci_measured:
            ci_evidence.append(
                f"低次谐波能量占主导（1-4 次幅值和 "
                f"{math.sqrt(low_energy):.3f}）")
    else:
        ci_evidence.append("未见累计分度漂移")

    # ---- 主导机理：存在的机理间比较实测能量 ----
    e_ecc = h1
    e_cum = math.sqrt(low_energy)
    e_st = max((abs(r) for r in outliers.values()), default=0.0)
    scores = {"eccentricity": e_ecc if ecc_present else -1.0,
              "cumulative_index": e_cum if ci_present else -1.0,
              "single_tooth": e_st if st_present else -1.0}
    dominant = max(scores, key=scores.get)
    if scores[dominant] <= 0.0:
        dominant = "none"

    return dict(
        single_tooth_defect=dict(
            present=st_present, confidence=st_conf,
            evidence="；".join(st_evidence), teeth=st_teeth,
            amplitude=st_amp, phase_deg=None),
        wheel_eccentricity=dict(
            present=ecc_present, confidence=ecc_conf,
            evidence="；".join(ecc_evidence),
            teeth=[], amplitude=ecc if ecc > 1e-6 else None,
            phase_deg=ecc_phase),
        cumulative_index_error=dict(
            present=ci_present, confidence=ci_conf,
            evidence="；".join(ci_evidence),
            teeth=[excursion_tooth] if ci_present else [],
            amplitude=excursion if ci_present else None, phase_deg=None),
        dominant=dominant)


# ----------------------------------------------------------------------
# 驱动：整轮分析
# ----------------------------------------------------------------------
def _table_is_empty(table) -> bool:
    """误差表是否完全为名义值（无录齿、无偏心）。"""
    if abs(table.eccentricity.x) > 1e-12 \
            or abs(table.eccentricity.y) > 1e-12:
        return False
    for e in table.teeth:
        if abs(e.pitch_deviation_deg) > 1e-12 \
                or abs(e.tip_radial_deviation_mm) > 1e-12 \
                or e.tip_half_angle_deg is not None:
            return False
    return True


def _run_once(inp, table, n: int):
    """单趟整轮仿真，返回 (raw, 逐齿拍列表, 指标序列, 逐齿穿透)。"""
    geom, effective = build_toothed_wheel(inp.wheel, table, inp.rotation)
    # 半齿距结构下每齿每转被两瓦各锁一次（约 2n 拍），段数需覆盖
    # 整圈并留出首拍截断与末拍落角闭合的余量
    sim = WheelSimulation(inp, geom, n_segments=2 * n + 4)
    raw = sim.run()

    beats = raw["metrics"]["beats"]
    # 首拍锁面段在扫描开始前已锁住，锁量被截断，弃用
    by_tooth: Dict[int, dict] = {}
    for b in beats[1:]:
        by_tooth.setdefault(b["_k"] % n, b)

    pen = [0.0] * n
    for s in raw["_samples"]:
        if s.pen_tooth is not None \
                and s.penetration > pen[s.pen_tooth % n]:
            pen[s.pen_tooth % n] = s.penetration

    series: Dict[str, List[float]] = {
        m: [float("nan")] * n for m in METRICS}
    for t, b in by_tooth.items():
        for m in METRICS:
            v = _clean(b.get(m))
            if v is not None:
                series[m][t] = v
    return raw, beats, by_tooth, series, pen, effective


def analyze_wheel(inp, table) -> dict:
    """运行整轮逐齿分析，返回原始 dict（不修改 ``inp``/``table``）。

    逐齿行为该误差表下的原始指标；峰峰值/谐波/最差齿/机理分类基于
    与名义轮（空误差表）同流程基线的差分序列，以消除设计本身的
    不对称（如进/出瓦名义落角不同）。
    """
    from .models import WheelErrorTable  # 避免模块级循环引用

    n = inp.wheel.teeth
    raw, beats, by_tooth, series, pen, effective = _run_once(
        inp, table, n)

    # 名义基线差分
    empty = _table_is_empty(table)
    if empty:
        base_series = {m: [0.0] * n for m in METRICS}
        delta = {m: [0.0] * n for m in METRICS}
    else:
        _, _, _, base_series, _, _ = _run_once(inp, WheelErrorTable(), n)
        delta = {m: [float("nan")] * n for m in METRICS}
        for m in METRICS:
            for t in range(n):
                a, b = series[m][t], base_series[m][t]
                if math.isfinite(a) and math.isfinite(b):
                    delta[m][t] = a - b

    per_tooth: List[dict] = []
    for t in range(n):
        b = by_tooth.get(t)
        row_delta = {m: (round(delta[m][t], 9)
                         if math.isfinite(delta[m][t]) else None)
                     for m in METRICS}
        if b is None:
            per_tooth.append(dict(
                tooth=t, pallet=None, pallet_name="",
                lock_deg=None, lift_deg=None, drop_deg=None,
                recoil_deg=None, dead_clearance_mm=None,
                max_penetration_mm=pen[t],
                land_theta_deg=None, unlock_theta_deg=None,
                release_theta_deg=None, land_psi_deg=None,
                unlock_psi_deg=None, release_psi_deg=None,
                flags=["missing_beat"], delta=row_delta))
            continue
        flags = []
        if pen[t] > EPS_MM:
            flags.append("tip_penetration")
        recoil = _clean(b.get("recoil_deg")) or 0.0
        if recoil > RECOIL_FLAG_DEG:
            flags.append("recoil")
        per_tooth.append(dict(
            tooth=t, pallet=b["pallet"], pallet_name=b["pallet_name"],
            lock_deg=_clean(b.get("lock_deg")),
            lift_deg=_clean(b.get("lift_deg")),
            drop_deg=_clean(b.get("drop_deg")),
            recoil_deg=_clean(b.get("recoil_deg")),
            dead_clearance_mm=_clean(b.get("dead_clearance_mm")),
            max_penetration_mm=pen[t],
            land_theta_deg=(_clean(b["_th_land"] * R2D)
                            if b.get("_th_land") is not None else None),
            unlock_theta_deg=(_clean(b["_th_unlock"] * R2D)
                              if b.get("_th_unlock") is not None else None),
            release_theta_deg=(_clean(b["_th_release"] * R2D)
                               if b.get("_th_release") is not None
                               else None),
            land_psi_deg=(_clean(b["_psi_land"] * R2D)
                          if b.get("_psi_land") is not None else None),
            unlock_psi_deg=(_clean(b["_psi_unlock"] * R2D)
                            if b.get("_psi_unlock") is not None else None),
            release_psi_deg=(_clean(b["_psi_release"] * R2D)
                             if b.get("_psi_release") is not None else None),
            flags=flags, delta=row_delta))

    stats_list = [series_stats(m, delta[m], n) for m in METRICS]
    stats = {s["metric"]: s for s in stats_list}

    # 最差齿：各指标差分归一化偏移的最大者
    worst_tooth, worst_score = -1, -1.0
    for t in range(n):
        score = 0.0
        for m in METRICS:
            st = stats[m]
            v = delta[m][t]
            if not math.isfinite(v) or not math.isfinite(
                    st["mean"]) or st["p2p"] <= 1e-12:
                continue
            score = max(score, abs(v - st["mean"]) / (0.5 * st["p2p"]))
        if score > worst_score:
            worst_tooth, worst_score = t, score
    if 0 <= worst_tooth < n and worst_score > 1e-9 and "outlier" not in \
            per_tooth[worst_tooth]["flags"]:
        per_tooth[worst_tooth]["flags"].append("outlier")

    classification = _classify(table, n, delta, stats)

    return dict(
        teeth=n,
        beats_observed=len(beats),
        per_tooth=per_tooth,
        stats=stats_list,
        worst_tooth=worst_tooth,
        classification=classification,
        effective_teeth=effective,
        baseline="nominal-subtracted" if not empty else "none",
        _raw=raw,
    )


# ----------------------------------------------------------------------
# 两组误差表对比
# ----------------------------------------------------------------------
COMPARE_EPS = {"lock_deg": 1e-6, "lift_deg": 1e-6, "drop_deg": 1e-6,
               "recoil_deg": 1e-6, "dead_clearance_mm": 1e-6,
               "max_penetration_mm": 1e-6}
COMPARE_METRICS = METRICS + ["max_penetration_mm"]


def compare_tables(inp, table_a, table_b) -> dict:
    """对比两组误差表：逐齿定位指标变化（不修改入参）。"""
    ra = analyze_wheel(inp, table_a)
    rb = analyze_wheel(inp, table_b)
    n = ra["teeth"]

    def _metric_map(analysis):
        return {row["tooth"]: row for row in analysis["per_tooth"]}

    ma, mb = _metric_map(ra), _metric_map(rb)
    tooth_deltas = []
    changed = []
    for t in range(n):
        metrics = {}
        max_abs = 0.0
        for m in COMPARE_METRICS:
            va = _clean(ma.get(t, {}).get(m))
            vb = _clean(mb.get(t, {}).get(m))
            d = (vb - va) if (va is not None and vb is not None) else None
            metrics[m] = dict(a=va, b=vb, delta=d)
            if d is not None:
                max_abs = max(max_abs, abs(d))
        is_changed = any(
            metrics[m]["delta"] is not None
            and abs(metrics[m]["delta"]) > COMPARE_EPS[m]
            for m in COMPARE_METRICS)
        if is_changed:
            changed.append(t)
        tooth_deltas.append(dict(
            tooth=t, metrics=metrics, changed=is_changed,
            max_abs_delta=max_abs))

    def _stat(analysis, metric):
        for s in analysis["stats"]:
            if s["metric"] == metric:
                return s
        return {}

    metric_summaries = []
    for m in METRICS:
        sa, sb = _stat(ra, m), _stat(rb, m)
        deltas = [(t, abs(tooth_deltas[t]["metrics"][m]["delta"]))
                  for t in range(n)
                  if tooth_deltas[t]["metrics"][m]["delta"] is not None]
        wt, wd = (max(deltas, key=lambda x: x[1]) if deltas
                  else (-1, 0.0))
        metric_summaries.append(dict(
            metric=m,
            p2p_a=sa.get("p2p", float("nan")),
            p2p_b=sb.get("p2p", float("nan")),
            worst_tooth_a=sa.get("worst_tooth", -1),
            worst_tooth_b=sb.get("worst_tooth", -1),
            max_abs_delta=float(wd),
            worst_delta_tooth=int(wt)))

    return dict(analysis_a=ra, analysis_b=rb,
                tooth_deltas=tooth_deltas, changed_teeth=changed,
                metric_summaries=metric_summaries)


# ----------------------------------------------------------------------
# 端点级校验（需要轮齿数上下文）：返回字段化错误列表
# ----------------------------------------------------------------------
def table_field_errors(table, n_teeth: int,
                       prefix: Sequence[str] = ("body", "errors")
                       ) -> List[dict]:
    """齿号范围 / 声明齿数校验；重复齿号与闭合由 Pydantic 负责。"""
    errs = []
    if table.declared_teeth is not None \
            and table.declared_teeth != n_teeth:
        errs.append(dict(
            loc=tuple(prefix) + ("declared_teeth",),
            msg=f"齿数不符: 误差表声明 {table.declared_teeth} 齿，"
                f"擒纵轮为 {n_teeth} 齿",
            type="value_error"))
    for i, e in enumerate(table.teeth):
        if e.tooth >= n_teeth:
            errs.append(dict(
                loc=tuple(prefix) + ("teeth", i, "tooth"),
                msg=f"齿号 {e.tooth} 超出范围 [0, {n_teeth - 1}]"
                    f"（齿数不符: 擒纵轮为 {n_teeth} 齿）",
                type="value_error"))
    return errs
