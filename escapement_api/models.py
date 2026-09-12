# -*- coding: utf-8 -*-
"""Pydantic 输入 / 输出模型。

约定
----
* 平面坐标，单位任意（取 mm），角度全部用度；API 内部转弧度。
* 擒纵轮逆时针旋转 ``rotation="ccw"`` 为 +1，顺时针 ``"cw"`` 为 -1。
* 锚 (anchor) 以锚轴为转动中心，锚角 ``theta=0`` 取输入几何的名义位置，
  正方向按右手系（逆时针）。
* 所有瓦面端点都给在 *锚系* 坐标（随锚一起转动），原点即锚轴。
"""
from __future__ import annotations

import math
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ----------------------------------------------------------------------
# 输入模型
# ----------------------------------------------------------------------
class Vec2(BaseModel):
    x: float
    y: float


class Face(BaseModel):
    """一个瓦面（锁面 lock 或冲面 impulse）。

    ``a -> b`` 给锚系坐标的两个端点。锁面约定 ``a`` 为瓦跟（远离角尖
    的一端），``b`` 为瓦角 (corner)；冲面约定 ``b`` 为瓦角、``c`` 为冲尾。

    死节式（静止式）锁面通常是以锚轴为圆心的同心圆弧：给定
    ``arc_center``（锚系，一般即原点 (0,0)）时该面按圆弧处理；缺省为
    平面（弦/直线）。
    """

    a: Vec2
    b: Vec2
    arc_center: Optional[Vec2] = None


class Pallet(BaseModel):
    name: str = "pallet"
    lock: Face
    impulse: Face


class WheelSpec(BaseModel):
    """擒纵轮：齿数、节圆（齿尖分布）半径、齿根半径、齿面斜度。"""

    teeth: int = Field(..., ge=6, le=120)
    pitch_radius: float = Field(..., gt=0, description="齿尖所在圆半径 Rp")
    root_radius: float = Field(..., gt=0, description="齿根圆半径 Rf < Rp")
    tip_half_angle: float = Field(
        10.0, ge=0.0, le=45.0,
        description="齿尖半宽角（度）：齿尖两侧面相对径线的倾角之和",
    )

    @model_validator(mode="after")
    def _check_radii(self) -> "WheelSpec":
        if self.root_radius >= self.pitch_radius:
            raise ValueError("root_radius 必须小于 pitch_radius")
        return self


class Swing(BaseModel):
    """锚摆角范围（度）与初始相位。

    一个完整锚周期：``theta_min -> theta_max -> theta_min``（去程、回程）。
    一齿周期对应半次摆动（一次滴答）。
    """

    theta_min: float = Field(..., description="锚角下限（度）")
    theta_max: float = Field(..., description="锚角上限（度）")
    theta_start: Optional[float] = Field(
        None, description="起始锚角，缺省=theta_min")
    initial_wheel_phase: float = Field(
        0.0, description="起始轮转角 psi（度，绝对相位）")

    @model_validator(mode="after")
    def _check(self) -> "Swing":
        if self.theta_max <= self.theta_min:
            raise ValueError("theta_max 必须大于 theta_min")
        return self


class Tolerances(BaseModel):
    """制造公差（对称 ± 值）。None 表示该项不参与公差校核。"""

    anchor_xy: Optional[float] = Field(
        None, gt=0, description="锚轴中心位置 ± 公差（mm），x/y 独立")
    pallet_angle: Optional[float] = Field(
        None, gt=0, description="瓦面角度修磨/安装 ± 公差（度）")
    depth: Optional[float] = Field(
        None, gt=0, description="入深（轮心-锚轴距） ± 公差（mm）")
    wheel_radius: Optional[float] = Field(
        None, gt=0, description="节圆半径制造 ± 公差（mm）")


class LockedParam(BaseModel):
    """调整时锁定的参数。"""

    anchor_xy: bool = False
    pallet_angle: bool = False
    depth: bool = False


class AdjustRange(BaseModel):
    """单个调整手段的允许范围（相对名义值）。"""

    max_anchor_move: float = Field(
        0.5, ge=0, description="锚轴最多平移量（mm）")
    lock_face_grind: float = Field(
        4.0, ge=0, description="锁面角度最多改动（度）")
    impulse_face_grind: float = Field(
        4.0, ge=0, description="冲面角度最多改动（度）")
    depth_change: float = Field(
        0.5, ge=0, description="入深最多改变（mm），正值=加深")


class Targets(BaseModel):
    """调整 / 校核的指标窗口（角度均为度，长度 mm）。"""

    drop_min: float = 1.0
    drop_max: float = 6.0
    lock_min: float = 0.5
    lock_max: float = 4.0
    lift_min: float = 3.0
    lift_max: float = 8.0
    recoil_max: float = 0.4
    clearance_min: float = 0.02
    max_penetration: float = 0.0


class EscapementInput(BaseModel):
    name: str = "escapement"
    wheel: WheelSpec
    wheel_center: Vec2 = Field(..., description="擒纵轮中心 O（世界坐标）")
    anchor_center: Vec2 = Field(..., description="锚轴 A（世界坐标）")
    entry_pallet: Pallet = Field(..., alias="entry_pallet")
    exit_pallet: Pallet = Field(..., alias="exit_pallet")
    rotation: Literal["ccw", "cw"]
    swing: Swing
    tolerances: Tolerances = Field(default_factory=Tolerances)
    step_deg: float = Field(
        0.05, gt=0, le=1.0,
        description="离散锚角步长（度），事件边界再用 brentq 细化")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _check_faces(self) -> "EscapementInput":
        for p in (self.entry_pallet, self.exit_pallet):
            for f in (p.lock, p.impulse):
                la = math.hypot(f.a.x, f.a.y)
                lb = math.hypot(f.b.x, f.b.y)
                if la < 1e-9 or lb < 1e-9:
                    raise ValueError(
                        f"瓦面 {p.name} 端点不能落在锚轴上（退化）")
                d = math.hypot(f.a.x - f.b.x, f.a.y - f.b.y)
                if d < 1e-9:
                    raise ValueError(f"瓦面 {p.name} 端点重合")
        return self


# ----------------------------------------------------------------------
# 输出模型
# ----------------------------------------------------------------------
class ContactInfo(BaseModel):
    pallet: int = Field(description="0=进瓦 1=出瓦")
    face: Literal["lock", "impulse"]
    theta_deg: float
    wheel_psi_deg: float
    point: Vec2
    penetration_mm: float = 0.0


class EventOut(BaseModel):
    kind: Literal[
        "initial_lock", "unlock", "release", "landing", "drop",
        "anomaly",
    ]
    theta_deg: float
    wheel_psi_deg: Optional[float] = None
    pallet: Optional[int] = None
    face: Optional[Literal["lock", "impulse"]] = None
    point: Optional[Vec2] = None
    note: str = ""


class BeatMetrics(BaseModel):
    pallet: int
    pallet_name: str
    lock_deg: float = Field(description="锁量（锚角上的锁住幅度，度）")
    lock_at_rest_mm: float = Field(description="静止锁住时齿尖越进瓦角的深度(mm)")
    lift_deg: float = Field(description="升角：冲面作用的锚角跨度（度）")
    wheel_impulse_deg: float = Field(description="冲面期间轮转角（度）")
    drop_deg: float = Field(description="落角：释放到次瓦锁住之间的轮空转角（度）")
    recoil_deg: float = Field(description="回退量（轮角最大后移，度，正=有回退）")
    advance_deg: float = Field(description="本次滴答轮净推进（度）")
    dead_clearance_mm: float = Field(
        description="静止段间隙：锁住期对面瓦与最近齿尖的最小间隙(mm)")
    lock_arc_eccentricity_mm: float = Field(
        description="锁面接触半径变化（相对同心锁面=0，mm）")


class AnomalyOut(BaseModel):
    code: Literal[
        "tip_penetration",
        "recoil",
        "double_contact",
        "not_locked",
        "order_jump",
        "bad_geometry",
    ]
    theta_deg: float
    wheel_psi_deg: Optional[float] = None
    pallet: Optional[int] = None
    face: Optional[str] = None
    point: Optional[Vec2] = None
    detail: str
    value: Optional[float] = None


class SimMetrics(BaseModel):
    beats: List[BeatMetrics] = Field(default_factory=list)
    impulse_share: List[float] = Field(
        default_factory=list,
        description="各瓦冲量分配（按轮冲量角归一化，和≈1）")
    mean_drop_deg: float = 0.0
    max_recoil_deg: float = 0.0
    min_dead_clearance_mm: float = float("inf")
    total_advance_deg: float = 0.0
    max_penetration_mm: float = 0.0


class SimResult(BaseModel):
    status: Literal["ok", "warning", "fail"]
    nominal: bool = True
    metrics: SimMetrics
    events: List[EventOut] = Field(default_factory=list)
    anomalies: List[AnomalyOut] = Field(default_factory=list)
    trace: List[ContactInfo] = Field(
        default_factory=list,
        description="离散接触轨迹（稀疏采样，供 SVG/JSON 导出）")
    message: str = ""


class ToleranceReport(BaseModel):
    samples: int
    fail_rate: float
    worst_margin: float = Field(description="最紧余量（正=全部满足）")
    limiting_target: str = ""
    per_metric_worst: dict = Field(default_factory=dict)
    worst_case: Optional[SimResult] = None
    pass_nominal: bool


class SimResponse(BaseModel):
    input_snapshot: dict
    result: SimResult
    tolerance: Optional[ToleranceReport] = None
    saved_id: Optional[int] = None


class Candidate(BaseModel):
    rank: int
    changes: List[str] = Field(description="可读的参数改动列表")
    anchor_center: Vec2
    depth_change_mm: float
    pallet_face_rotations_deg: List[float] = Field(
        description="[进瓦锁面, 进瓦冲面, 出瓦锁面, 出瓦冲面] 相对名义的转角")
    min_margin: float
    residual_conflicts: List[str]
    result: SimResult
    score: float
    multiplicity_group: int = Field(description="近似多解分组号")


class AdjustResponse(BaseModel):
    nominal: SimResult
    targets: Targets
    candidates: List[Candidate]
    locked: LockedParam


class CompareEntry(BaseModel):
    id: int
    kind: str
    name: str
    created_at: str
    status: str
    metrics_summary: dict


class DiffReport(BaseModel):
    a: CompareEntry
    b: CompareEntry
    metric_deltas: dict
    events_only_in_a: List[dict]
    events_only_in_b: List[dict]
    anomaly_codes_a: List[str]
    anomaly_codes_b: List[str]


# ----------------------------------------------------------------------
# 全轮逐齿误差分析
# ----------------------------------------------------------------------
class ToothError(BaseModel):
    """单个齿的制造误差记录（未录齿沿用名义值）。

    * ``pitch_deviation_deg``: 节距偏差（度）= 该齿到下一齿的实际角距
      减去名义齿距；全轮之和必须为 0（否则整轮无法闭合）；
    * ``tip_radial_deviation_mm``: 齿尖径向偏差（mm）= 实际齿尖半径
      减去名义节圆半径；
    * ``tip_half_angle_deg``: 齿尖半角实测值（度），缺省沿用名义值。
    """

    tooth: int = Field(..., ge=0, description="齿号 0..N-1")
    pitch_deviation_deg: float = 0.0
    tip_radial_deviation_mm: float = 0.0
    tip_half_angle_deg: Optional[float] = Field(None, ge=0.0, le=45.0)


class WheelErrorTable(BaseModel):
    """按齿号记录的整轮误差表 + 轮心偏心向量。"""

    name: str = "error-table"
    teeth: List[ToothError] = Field(default_factory=list)
    eccentricity: Vec2 = Field(
        default_factory=lambda: Vec2(x=0.0, y=0.0),
        description="轮心偏心向量（mm，世界系，随轮旋转）")
    declared_teeth: Optional[int] = Field(
        None, description="声明的轮齿数；与 EscapementInput 不符时报错")
    closure_tol_deg: float = Field(
        0.01, gt=0, le=1.0, description="累计节距闭合公差（度）")

    @model_validator(mode="after")
    def _check_table(self) -> "WheelErrorTable":
        seen = set()
        dups = set()
        for e in self.teeth:
            if e.tooth in seen:
                dups.add(e.tooth)
            seen.add(e.tooth)
        if dups:
            raise ValueError(f"重复齿号: {sorted(dups)}")
        total = sum(e.pitch_deviation_deg for e in self.teeth)
        if abs(total) > self.closure_tol_deg:
            raise ValueError(
                f"累计节距无法闭合: 节距偏差总和 {total:+.4f}° 超出"
                f"闭合公差 ±{self.closure_tol_deg}°（未录齿按 0 计）")
        return self


class WheelAnalyzeRequest(BaseModel):
    input: EscapementInput
    errors: WheelErrorTable = Field(default_factory=WheelErrorTable)


class WheelCompareRequest(BaseModel):
    input: EscapementInput
    errors_a: WheelErrorTable
    errors_b: WheelErrorTable


class HarmonicOut(BaseModel):
    order: int = Field(description="谐波次数（每转周期数）")
    amplitude: float = Field(description="幅值（与所属指标同单位）")
    phase_deg: float


class MetricSeriesStats(BaseModel):
    metric: str
    mean: float
    min: float
    max: float
    p2p: float = Field(description="峰峰值")
    worst_tooth: int = Field(description="偏离均值最远的齿号")
    harmonics: List[HarmonicOut] = Field(default_factory=list)


class ToothMetrics(BaseModel):
    """逐齿（每拍）指标：锁住/解锁/冲面/释放/落瓦的相位与诊断。"""

    tooth: int
    pallet: Optional[int] = None
    pallet_name: str = ""
    lock_deg: Optional[float] = None
    lift_deg: Optional[float] = None
    drop_deg: Optional[float] = None
    recoil_deg: Optional[float] = None
    dead_clearance_mm: Optional[float] = None
    max_penetration_mm: float = 0.0
    land_theta_deg: Optional[float] = None
    unlock_theta_deg: Optional[float] = None
    release_theta_deg: Optional[float] = None
    land_psi_deg: Optional[float] = None
    unlock_psi_deg: Optional[float] = None
    release_psi_deg: Optional[float] = None
    flags: List[str] = Field(default_factory=list)
    delta: dict = Field(
        default_factory=dict,
        description="相对名义轮（空误差表同流程基线）的逐指标差分")


class ErrorMechanism(BaseModel):
    """一类误差机理的识别结果。"""

    present: bool
    confidence: Literal["high", "medium", "low", "none"] = "none"
    evidence: str = ""
    teeth: List[int] = Field(default_factory=list)
    amplitude: Optional[float] = None
    phase_deg: Optional[float] = None


class ClassificationOut(BaseModel):
    single_tooth_defect: ErrorMechanism
    wheel_eccentricity: ErrorMechanism
    cumulative_index_error: ErrorMechanism
    dominant: Literal[
        "single_tooth", "eccentricity", "cumulative_index", "none"]


class EffectiveTooth(BaseModel):
    """折算后的实际齿尖几何（含偏心向量折算）。"""

    tooth: int
    radius_mm: float
    angle_deg: float
    pitch_deviation_deg: float
    radial_deviation_mm: float
    tip_half_angle_deg: float


class WheelAnalysis(BaseModel):
    teeth: int
    beats_observed: int
    per_tooth: List[ToothMetrics]
    stats: List[MetricSeriesStats] = Field(
        description="逐指标统计（基于相对名义轮的差分序列）")
    worst_tooth: int
    classification: ClassificationOut
    effective_teeth: List[EffectiveTooth]
    baseline: str = Field(
        "none", description="统计基线：nominal-subtracted=已减名义轮")


class WheelAnalyzeResponse(BaseModel):
    input_snapshot: dict
    error_table_snapshot: dict
    result: SimResult
    analysis: WheelAnalysis


class MetricDelta(BaseModel):
    a: Optional[float] = None
    b: Optional[float] = None
    delta: Optional[float] = None


class ToothDelta(BaseModel):
    tooth: int
    metrics: dict = Field(description="指标名 -> {a, b, delta}")
    changed: bool
    max_abs_delta: float = 0.0


class MetricCompareSummary(BaseModel):
    metric: str
    p2p_a: float
    p2p_b: float
    worst_tooth_a: int
    worst_tooth_b: int
    max_abs_delta: float
    worst_delta_tooth: int


class WheelCompareResponse(BaseModel):
    input_snapshot: dict
    analysis_a: WheelAnalysis
    analysis_b: WheelAnalysis
    result_a: SimResult
    result_b: SimResult
    tooth_deltas: List[ToothDelta]
    changed_teeth: List[int] = Field(description="任一指标变化超差的齿号")
    metric_summaries: List[MetricCompareSummary]
