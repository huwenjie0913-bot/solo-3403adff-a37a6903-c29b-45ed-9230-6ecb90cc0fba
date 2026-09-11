# -*- coding: utf-8 -*-
"""FastAPI 应用入口。

本机启动::

    python3 -m escapement_api.main
    # 或
    uvicorn escapement_api.main:app --host 127.0.0.1 --port 8000

路由：
* ``GET  /health``            存活检查；
* ``POST /simulate``          接收 :class:`EscapementInput`，运行离散转角
                              + 根求解的接触追踪，返回结构化校核结果；
* ``GET  /simulate/example``  内置对称死节样例（只读，便于快速联调）。

本模块只做 HTTP 编排与结果序列化，几何/接触逻辑在 :mod:`simulate`。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .design import build_symmetric_design
from .models import (
    AnomalyOut, BeatMetrics, ContactInfo, EscapementInput, EventOut,
    SimMetrics, SimResponse, SimResult, Vec2,
)
from .simulate import simulate as run_simulation

app = FastAPI(
    title="静止式锚形擒纵机构校核 API",
    version="0.1.0",
    description="钟表维修师用的锚形擒纵一齿周期接触顺序追踪与诊断接口",
)


# ----------------------------------------------------------------------
# 仿真原始 dict -> Pydantic 输出模型
# ----------------------------------------------------------------------
def _f(x: Any) -> Optional[float]:
    """NumPy 标量/NaN/Inf 清洗（非有限值序列化为 null）。"""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _vec(p: Any) -> Optional[Vec2]:
    if p is None:
        return None
    try:
        return Vec2(x=float(p[0]), y=float(p[1]))
    except (TypeError, IndexError):
        return None


def _adapt_result(raw: Dict[str, Any]) -> SimResult:
    beats: List[BeatMetrics] = []
    for b in raw.get("metrics", {}).get("beats", []):
        beats.append(BeatMetrics(
            pallet=int(b["pallet"]),
            pallet_name=str(b["pallet_name"]),
            lock_deg=_f(b["lock_deg"]) or 0.0,
            lock_at_rest_mm=_f(b["lock_at_rest_mm"]) or 0.0,
            lift_deg=_f(b["lift_deg"]) or 0.0,
            wheel_impulse_deg=_f(b["wheel_impulse_deg"]) or 0.0,
            drop_deg=_f(b["drop_deg"]) or 0.0,
            recoil_deg=_f(b["recoil_deg"]) or 0.0,
            advance_deg=_f(b["advance_deg"]) or 0.0,
            dead_clearance_mm=_f(b["dead_clearance_mm"]) or 0.0,
            lock_arc_eccentricity_mm=
                _f(b["lock_arc_eccentricity_mm"]) or 0.0,
        ))

    events = [
        EventOut(
            kind=e["kind"],
            theta_deg=_f(e["theta"] * 180.0 / math.pi) or 0.0,
            wheel_psi_deg=(_f(e["psi"] * 180.0 / math.pi)
                           if e.get("psi") is not None else None),
            pallet=e.get("pallet"),
            face=e.get("face"),
            point=_vec(e.get("point")),
            note=e.get("note", ""),
        )
        for e in raw.get("events", [])
    ]

    anomalies = [
        AnomalyOut(
            code=a["code"],
            theta_deg=_f(a["theta"] * 180.0 / math.pi) or 0.0,
            wheel_psi_deg=(_f(a["psi"] * 180.0 / math.pi)
                           if a.get("psi") is not None else None),
            pallet=a.get("pallet"),
            face=a.get("face"),
            point=_vec(a.get("point")),
            detail=a.get("detail", ""),
            value=_f(a.get("value")),
        )
        for a in raw.get("anomalies", [])
    ]

    trace = [
        ContactInfo(
            pallet=int(t["pallet"]),
            face=t["face"],
            theta_deg=float(t["theta_deg"]),
            wheel_psi_deg=float(t["wheel_psi_deg"]),
            point=Vec2(**t["point"]),
            penetration_mm=float(t["penetration_mm"]),
        )
        for t in raw.get("trace", [])
    ]

    m = raw.get("metrics", {})
    metrics = SimMetrics(
        beats=beats,
        impulse_share=[float(x) for x in m.get("impulse_share", [])],
        mean_drop_deg=_f(m.get("mean_drop_deg")) or 0.0,
        max_recoil_deg=_f(m.get("max_recoil_deg")) or 0.0,
        min_dead_clearance_mm=_f(m.get("min_dead_clearance_mm")) or 0.0,
        total_advance_deg=_f(m.get("total_advance_deg")) or 0.0,
        max_penetration_mm=_f(m.get("max_penetration_mm")) or 0.0,
    )

    return SimResult(
        status=raw["status"],
        nominal=True,
        metrics=metrics,
        events=events,
        anomalies=anomalies,
        trace=trace,
        message=raw.get("message", ""),
    )


def _build_response(inp: EscapementInput) -> SimResponse:
    raw = run_simulation(inp)
    result = _adapt_result(raw)
    snapshot = inp.model_dump(mode="json", by_alias=True, exclude_none=True)
    return SimResponse(input_snapshot=snapshot, result=result)


# ----------------------------------------------------------------------
# 路由
# ----------------------------------------------------------------------
@app.get("/health", tags=["system"])
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "escapement-api"}


@app.get("/", tags=["system"])
def root() -> Dict[str, Any]:
    return {
        "service": "静止式锚形擒纵机构校核 API",
        "docs": "/docs",
        "endpoints": [
            "GET  /health",
            "POST /simulate",
            "GET  /simulate/example",
        ],
    }


@app.post("/simulate", response_model=SimResponse, tags=["check"])
def simulate(inp: EscapementInput) -> SimResponse:
    """校核一份擒纵设计，返回事件序列、指标、诊断与接触轨迹。"""
    return _build_response(inp)


@app.get("/simulate/example", response_model=SimResponse, tags=["check"])
def simulate_example() -> SimResponse:
    """内置 15 齿对称死节样例，便于无请求体快速联调。"""
    design = build_symmetric_design()
    inp = EscapementInput.model_validate(design)
    return _build_response(inp)


# 让 pydantic 的校验错误仍走 FastAPI 默认 422（含中文 value 提示）。
@app.exception_handler(ValueError)
def _value_error_handler(_request, exc: ValueError) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "escapement_api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
