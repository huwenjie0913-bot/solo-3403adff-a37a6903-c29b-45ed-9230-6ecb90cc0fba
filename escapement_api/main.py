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

from fastapi import FastAPI, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .design import build_symmetric_design, mutate_design
from .models import (
    AnomalyOut, BeatMetrics, ClassificationOut, ContactInfo,
    EffectiveTooth, ErrorMechanism, EscapementInput, EventOut,
    HarmonicOut, MetricCompareSummary, MetricSeriesStats, SimMetrics,
    SimResponse, SimResult, ToothDelta, ToothError, ToothMetrics, Vec2,
    WheelAnalysis, WheelAnalyzeRequest, WheelAnalyzeResponse,
    WheelCompareRequest, WheelCompareResponse, WheelErrorTable,
)
from .simulate import simulate as run_simulation
from .svg import wheel_svg
from .wheel import analyze_wheel, compare_tables, table_field_errors

app = FastAPI(
    title="静止式锚形擒纵机构校核 API",
    version="0.2.0",
    description="钟表维修师用的锚形擒纵一齿周期接触顺序追踪与诊断接口，"
                "含全轮逐齿误差分析、误差表对比与 SVG 导出",
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


def _adapt_result(raw: Dict[str, Any], nominal: bool = True) -> SimResult:
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
        nominal=nominal,
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
# 全轮逐齿分析：原始 dict -> Pydantic 输出模型
# ----------------------------------------------------------------------
def _adapt_analysis(a: Dict[str, Any]) -> WheelAnalysis:
    stats = [
        MetricSeriesStats(
            metric=s["metric"],
            mean=_f(s["mean"]) or 0.0,
            min=_f(s["min"]) or 0.0,
            max=_f(s["max"]) or 0.0,
            p2p=_f(s["p2p"]) or 0.0,
            worst_tooth=int(s["worst_tooth"]),
            harmonics=[HarmonicOut(**h) for h in s["harmonics"]],
        )
        for s in a["stats"]
    ]
    cls = a["classification"]
    classification = ClassificationOut(
        single_tooth_defect=ErrorMechanism(**cls["single_tooth_defect"]),
        wheel_eccentricity=ErrorMechanism(**cls["wheel_eccentricity"]),
        cumulative_index_error=ErrorMechanism(
            **cls["cumulative_index_error"]),
        dominant=cls["dominant"],
    )
    return WheelAnalysis(
        teeth=int(a["teeth"]),
        beats_observed=int(a["beats_observed"]),
        per_tooth=[ToothMetrics(**row) for row in a["per_tooth"]],
        stats=stats,
        worst_tooth=int(a["worst_tooth"]),
        classification=classification,
        effective_teeth=[EffectiveTooth(**e)
                         for e in a["effective_teeth"]],
        baseline=a.get("baseline", "none"),
    )


def _check_table(req_errors: WheelErrorTable, n_teeth: int,
                 prefix=("body", "errors")) -> None:
    """端点级误差表校验：齿号范围 / 声明齿数（字段化 422）。"""
    errs = table_field_errors(req_errors, n_teeth, prefix)
    if errs:
        raise RequestValidationError(errs)


def _build_wheel_response(inp: EscapementInput,
                          table: WheelErrorTable) -> WheelAnalyzeResponse:
    analysis = analyze_wheel(inp, table)
    return WheelAnalyzeResponse(
        input_snapshot=inp.model_dump(mode="json", by_alias=True,
                                      exclude_none=True),
        error_table_snapshot=table.model_dump(mode="json",
                                              exclude_none=True),
        result=_adapt_result(analysis["_raw"], nominal=False),
        analysis=_adapt_analysis(analysis),
    )


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
            "POST /wheel/analyze",
            "POST /wheel/compare",
            "POST /wheel/svg",
            "GET  /wheel/example",
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


# ----------------------------------------------------------------------
# 全轮逐齿误差分析路由
# ----------------------------------------------------------------------
@app.post("/wheel/analyze", response_model=WheelAnalyzeResponse,
          tags=["wheel"])
def wheel_analyze(req: WheelAnalyzeRequest, response: Response,
                  download: bool = False) -> WheelAnalyzeResponse:
    """整轮逐齿误差分析（JSON 导出；``?download=true`` 加下载头）。

    接收 EscapementInput + 逐齿误差表（节距偏差/齿尖径向偏差/齿尖半角/
    轮心偏心向量），按实际齿序展开整轮接触，返回逐齿锁量、升角、落角、
    回退量、静止段间隙与穿透诊断，以及峰峰值、周期谐波、最差齿和
    误差机理分类。不修改输入。
    """
    _check_table(req.errors, req.input.wheel.teeth)
    if download:
        response.headers["Content-Disposition"] = \
            "attachment; filename=wheel_analysis.json"
    return _build_wheel_response(req.input, req.errors)


@app.post("/wheel/compare", response_model=WheelCompareResponse,
          tags=["wheel"])
def wheel_compare(req: WheelCompareRequest) -> WheelCompareResponse:
    """对比两组误差表：同一输入下各跑一遍整轮分析，按齿号定位指标变化。"""
    n = req.input.wheel.teeth
    _check_table(req.errors_a, n, ("body", "errors_a"))
    _check_table(req.errors_b, n, ("body", "errors_b"))
    cmp = compare_tables(req.input, req.errors_a, req.errors_b)
    return WheelCompareResponse(
        input_snapshot=req.input.model_dump(mode="json", by_alias=True,
                                            exclude_none=True),
        analysis_a=_adapt_analysis(cmp["analysis_a"]),
        analysis_b=_adapt_analysis(cmp["analysis_b"]),
        result_a=_adapt_result(cmp["analysis_a"]["_raw"], nominal=False),
        result_b=_adapt_result(cmp["analysis_b"]["_raw"], nominal=False),
        tooth_deltas=[ToothDelta(**d) for d in cmp["tooth_deltas"]],
        changed_teeth=cmp["changed_teeth"],
        metric_summaries=[MetricCompareSummary(**s)
                          for s in cmp["metric_summaries"]],
    )


@app.post("/wheel/svg", tags=["wheel"])
def wheel_svg_export(req: WheelAnalyzeRequest,
                     download: bool = False) -> Response:
    """导出带齿号、误差极坐标和异常标记的 SVG 图。"""
    _check_table(req.errors, req.input.wheel.teeth)
    analysis = analyze_wheel(req.input, req.errors)
    svg = wheel_svg(req.input, req.errors, analysis)
    headers = {"Content-Disposition":
               "attachment; filename=wheel_analysis.svg"} \
        if download else None
    return Response(content=svg, media_type="image/svg+xml",
                    headers=headers)


@app.get("/wheel/example", response_model=WheelAnalyzeResponse,
         tags=["wheel"])
def wheel_example() -> WheelAnalyzeResponse:
    """内置样例：15 齿死节轮 + 演示误差表（单齿缺陷+节距误差+偏心）。

    设计在名义对称样例基础上把锚轴平移 0.15 mm，使两瓦锁角错开
    整数齿距的临界对齐，整轮 15 齿均能完整啮合。
    """
    design = mutate_design(build_symmetric_design(),
                           move_xy=(0.15, 0.0))
    inp = EscapementInput.model_validate(design)
    table = WheelErrorTable(
        name="demo-errors",
        teeth=[
            ToothError(tooth=3, tip_radial_deviation_mm=0.12),
            ToothError(tooth=7, pitch_deviation_deg=0.15),
            ToothError(tooth=8, pitch_deviation_deg=-0.15),
        ],
        eccentricity=Vec2(x=0.05, y=0.0),
    )
    return _build_wheel_response(inp, table)


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
