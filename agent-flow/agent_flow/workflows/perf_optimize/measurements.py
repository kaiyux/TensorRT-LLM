"""Deterministic measurement checks shared by candidate and integration gates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} number")
    return result


def normalized_gain(reference: float, measured: float, metric: str) -> float:
    direction = -1 if metric.endswith("_ms") else 1
    return direction * (measured / reference - 1) * 100


def _curve_values(curve: Any, points: Sequence[int], name: str) -> dict[int, float]:
    if not isinstance(curve, list):
        raise ValueError("curve verdict is missing a curve")
    values: dict[int, float] = {}
    for row in curve:
        if not isinstance(row, dict):
            raise ValueError(f"{name} curve rows must be objects")
        point = row.get("concurrency")
        if isinstance(point, bool) or not isinstance(point, int) or point in values:
            raise ValueError(f"{name} curve requires unique integer concurrency points")
        values[point] = finite_number(row.get("value"), f"{name}[{point}].value", positive=True)
        for field in ("tok_s_user", "tok_s_gpu"):
            finite_number(row.get(field), f"{name}[{point}].{field}", positive=True)
    if set(values) != set(points):
        raise ValueError("curve verdict does not cover the configured concurrency points")
    if list(values) != sorted(values):
        raise ValueError(f"{name} curve concurrency points must be ascending")
    return values


def validate_acceptance(
    verdict: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    metric: str,
    required_gain: float,
    points: Sequence[int] | None = None,
    focus: Sequence[int] | None = None,
    allowed_regression: float = 0,
) -> float:
    """Validate evidence and return the gain computed from the measurements.

    Reported summaries may round to two decimal places. Acceptance always uses
    the computed gain; rounding cannot lift a result over the threshold.
    """
    measured = finite_number(verdict.get("measured_value"), "measured_value", positive=True)
    baseline = finite_number(reference.get("value"), "reference.value", positive=True)
    reported_gain = finite_number(verdict.get("measured_gain_pct"), "measured_gain_pct")
    threshold = finite_number(required_gain, "required_gain")
    if points is not None:
        reference_values = _curve_values(reference.get("curve"), points, "reference")
        measured_values = _curve_values(verdict.get("curve"), points, "measured")
        gains = {
            point: normalized_gain(reference_values[point], measured_values[point], metric)
            for point in points
        }
        for point, gain in gains.items():
            if gain < -allowed_regression - 1e-9:
                raise ValueError(
                    f"curve regresses concurrency {point} beyond the {allowed_regression}% budget"
                )
        scored = list(focus) if focus is not None else list(points)
        if not scored or not set(scored) <= set(points):
            raise ValueError("focus points must be a nonempty subset of configured points")
        computed_gain = sum(gains[point] for point in scored) / len(scored)
        computed_value = sum(measured_values[point] for point in scored) / len(scored)
        if not math.isclose(measured, computed_value, rel_tol=1e-6, abs_tol=0.01):
            raise ValueError(
                f"measured_value mismatch: reported {measured}, scored mean {computed_value}"
            )
    else:
        computed_gain = normalized_gain(baseline, measured, metric)
    finite_number(computed_gain, "computed gain")
    if computed_gain < threshold - 1e-9:
        raise ValueError(
            f"gain {computed_gain} (reported {reported_gain}) is below required {threshold}"
        )
    if not math.isclose(reported_gain, computed_gain, rel_tol=1e-6, abs_tol=0.01):
        raise ValueError(
            f"measured_gain_pct mismatch: reported {reported_gain}, computed {computed_gain}"
        )
    return computed_gain
