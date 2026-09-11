# -*- coding: utf-8 -*-
"""设计参数化与参数修改原语。

:func:`build_symmetric_design` 按经典 Graham 静止式（死节）锚形擒纵的
几何关系构造一组对称设计：

* 锁面取以锚轴为圆心的同心圆弧的弦（静止段不推动轮，无回退）；
* 两瓦锁角（瓦角）在轮心处的张角为整数个齿距——每滴答轮恰好推进
  一个齿；
* 冲面为瓦角到冲尾的直线，冲尾在释放角与齿尖圆相切；
* 轮冲量角 β 与落角 δ 满足 β + δ = 齿距角。

:func:`mutate_design` 实现三种调整手段（移动锚轴 / 修磨瓦面 / 改变入深），
公差抽样也复用它。
"""
from __future__ import annotations

import copy
import math
from typing import List

import numpy as np

from .geometry import D2R, R2D, rot

# 面旋转量在 4 维向量中的下标
FACE_DELTAS = ["entry_lock", "entry_impulse", "exit_lock", "exit_impulse"]


def build_symmetric_design(
    *,
    name: str = "graham-deadbeat-15t",
    teeth: int = 15,
    Rp: float = 15.0,
    Rf: float = 12.5,
    center_distance: float = 18.0,
    corner_half_angle_deg: float = 30.0,
    swing_half_deg: float = 6.0,
    unlock_deg: float = 2.0,
    release_deg: float = 4.0,
    wheel_impulse_deg: float = 20.5,
    tip_half_angle_deg: float = 9.0,
    rotation: str = "ccw",
    step_deg: float = 0.05,
):
    """构造对称死节擒纵输入 dict（与 EscapementInput 兼容）。"""
    pitch = 360.0 / teeth
    # 两锁角张角 = 整数齿距；对称布置时 180 + 2γ = m*pitch
    m = int(round((180.0 + 2 * corner_half_angle_deg) / pitch))
    gamma = (m * pitch - 180.0) / 2.0

    O = np.array([0.0, 0.0])
    A = np.array([0.0, center_distance])
    g = gamma * D2R
    th_u = unlock_deg * D2R
    th_r = release_deg * D2R
    th_min = -swing_half_deg * D2R
    th_max = swing_half_deg * D2R
    beta = wheel_impulse_deg * D2R

    s = 1 if rotation == "ccw" else -1
    # 初始相位（度）：让一个齿的齿尖停在进瓦瓦角（极角 -gamma）
    psi0_deg = -s * gamma

    def corner(side):  # side=+1 右(进瓦), -1 左(出瓦)
        ang = (-gamma if side > 0 else 180.0 + gamma) * D2R
        return O + Rp * np.array([math.cos(ang), math.sin(ang)])

    def release_tip(side):
        # 释放瞬间冲尾处齿尖位置
        if side > 0:
            ang = (-gamma + s * wheel_impulse_deg) * D2R
        else:
            ang = (180.0 + gamma - s * wheel_impulse_deg) * D2R
        return O + Rp * np.array([math.cos(ang), math.sin(ang)])

    pallets = []
    for side, pname in ((1, "entry"), (-1, "exit")):
        C = corner(side)
        # 瓦角锚系向量（在解锁角 θ_u 处接触 C；出瓦解锁角为 -θ_u）
        th_unlock = th_u if side > 0 else -th_u
        th_release = th_r if side > 0 else -th_r
        corner_anchor = rot(-th_unlock, C - A)
        # 锁面瓦跟：摆到对应极限角时同心弧仍经过 C
        th_extreme = th_min if side > 0 else th_max
        heel_anchor = rot(-th_extreme, C - A)
        # 冲尾锚系向量
        T = release_tip(side)
        tail_anchor = rot(-th_release, T - A)

        lock = {"a": _v(heel_anchor), "b": _v(corner_anchor),
                "arc_center": {"x": 0.0, "y": 0.0}}
        impulse = {"a": _v(corner_anchor), "b": _v(tail_anchor)}
        pallets.append({"name": pname, "lock": lock, "impulse": impulse})

    return {
        "name": name,
        "wheel": {
            "teeth": teeth,
            "pitch_radius": Rp,
            "root_radius": Rf,
            "tip_half_angle": tip_half_angle_deg,
        },
        "wheel_center": _v(O),
        "anchor_center": _v(A),
        "entry_pallet": pallets[0],
        "exit_pallet": pallets[1],
        "rotation": rotation,
        "swing": {
            "theta_min": -swing_half_deg,
            "theta_max": swing_half_deg,
            "initial_wheel_phase": psi0_deg,
        },
        "tolerances": {},
        "step_deg": step_deg,
    }


def _v(p: np.ndarray) -> dict:
    return {"x": float(p[0]), "y": float(p[1])}


def mutate_design(
    inp: dict,
    *,
    move_xy=(0.0, 0.0),
    depth_mm: float = 0.0,
    face_rotations_deg: List[float] = (0.0, 0.0, 0.0, 0.0),
    wheel_radius_delta: float = 0.0,
) -> dict:
    """返回修改后的设计 dict（深拷贝）。

    * ``move_xy``: 锚轴世界坐标平移 (dx, dy) mm；
    * ``depth_mm``: 改变入深，正值=入深加大（锚轴沿轮心→锚轴连线
      向轮心靠近）；
    * ``face_rotations_deg``: [进瓦锁面, 进瓦冲面, 出瓦锁面, 出瓦冲面]
      的修磨转角（度，正=逆时针），锁面绕瓦角转、冲面绕瓦角转；
    """
    out = copy.deepcopy(inp)
    O = np.array([out["wheel_center"]["x"], out["wheel_center"]["y"]])
    A = np.array([out["anchor_center"]["x"], out["anchor_center"]["y"]])
    A = A + np.asarray(move_xy, dtype=float)
    if depth_mm:
        d = A - O
        n = d / np.linalg.norm(d)
        A = A - n * depth_mm  # 靠近轮心 = 加深
    out["anchor_center"] = _v(A)

    if wheel_radius_delta:
        out["wheel"]["pitch_radius"] += wheel_radius_delta

    for pallet_key, (dl, di) in zip(
            ("entry_pallet", "exit_pallet"),
            (face_rotations_deg[0:2], face_rotations_deg[2:4])):
        p = out[pallet_key]
        if dl:
            # 锁面绕瓦角 b 转动瓦跟 a
            a = _asv(p["lock"]["a"])
            b = _asv(p["lock"]["b"])
            p["lock"]["a"] = _v(b + rot(dl * D2R, a - b))
        if di:
            # 冲面绕瓦角 a 转动冲尾 b
            a = _asv(p["impulse"]["a"])
            b = _asv(p["impulse"]["b"])
            p["impulse"]["b"] = _v(a + rot(di * D2R, b - a))
    return out


def _asv(v: dict) -> np.ndarray:
    return np.array([v["x"], v["y"]], dtype=float)
