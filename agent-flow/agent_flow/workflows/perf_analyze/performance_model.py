# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate the shared theoretical-best model and derive convergence conservatively.

``performance_model.yaml`` identifies its version, model_id, target_metric and
direction (higher/lower). Its points cover each benchmark concurrency, including
``null`` for a scalar benchmark. Every point records its operating_point (build,
hardware, workload, timing_basis), nullable measured_value and
theoretical_best_value, derivation, assumptions, evidence, measurement_evidence,
components, status, unexplained, and next_test. Evidence for a known bound and a
known measurement must be nonempty. Unknown values stay null.

Optional point measured_ms/theoretical_best_ms fields are paired and require a
timing_derivation explaining overlap removal and conversion to the target metric.
Component times describe disjoint critical-path contributions, never kernel sums
with overlap. Each component has id, nullable measured_ms/theoretical_best_ms,
gap_kind, evidence, and next_test for unresolved or restricted gaps.

Convergence requires numeric timing totals reconciled with complete component
accounting, and an explicit, physically justified convergence_tolerance_pct
(0..5) and convergence_evidence. The residual percentage
is the absolute metric distance divided by theoretical_best_value; it is not the
potential throughput uplift. Numerical comparisons allow 0.1% rounding error.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

MODEL_FILENAME = "performance_model.yaml"
MODEL_VERSION = 1
_REL_TOL = 0.001
_ABS_TOL = 1e-6
_STATUSES = {"open", "converged", "measurement_limited", "scope_limited", "model_invalid"}
_GAP_KINDS = {
    "actionable",
    "physical_limit",
    "scope_limited",
    "measurement_limited",
    "unexplained",
    "model_error",
}
_UNRESOLVED_KINDS = {"scope_limited", "measurement_limited", "unexplained", "model_error"}


class ModelError(ValueError):
    """The theoretical-best model is unreadable or internally inconsistent."""


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any, *, zero: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and (value >= 0 if zero else value > 0)
    except OverflowError:
        return False


def _strings(value: Any, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(_text(item) for item in value)
    )


def _concurrency(value: Any) -> bool:
    return value is None or (type(value) is int and value > 0)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _validate_point(point: dict, prefix: str, direction: str) -> list[str]:
    errors: list[str] = []

    def error(message: str) -> None:
        errors.append(f"{prefix}: {message}")

    operating = point.get("operating_point")
    if not isinstance(operating, dict) or not operating:
        error("operating_point must identify build, hardware, workload, and timing_basis")
    else:
        for name in ("build", "hardware", "workload", "timing_basis"):
            value = operating.get(name)
            if not (_text(value) or isinstance(value, dict) and bool(value)):
                error(f"operating_point.{name} must be a nonempty string or mapping")

    for name in ("measured_value", "theoretical_best_value"):
        if name not in point or point[name] is not None and not _number(point[name]):
            error(f"{name} must be a positive finite number or null")
    if not _text(point.get("derivation")):
        error("derivation must explain the bound or why it is unknown")
    for name in ("assumptions", "evidence", "measurement_evidence"):
        required = (
            name == "evidence"
            and point.get("theoretical_best_value") is not None
            or name == "measurement_evidence"
            and point.get("measured_value") is not None
        )
        if not _strings(point.get(name), nonempty=required):
            error(
                f"{name} must be a list of nonempty strings"
                + (" with evidence" if required else "")
            )
    for name in ("unexplained", "next_test"):
        if not isinstance(point.get(name), str):
            error(f"{name} must be a string (empty only when nothing remains unresolved)")
    status = point.get("status")
    if not isinstance(status, str) or status not in _STATUSES:
        error(f"status must be one of {', '.join(sorted(_STATUSES))}")

    measured, best = point.get("measured_value"), point.get("theoretical_best_value")
    known_metrics = _number(measured) and _number(best)
    violation = (
        known_metrics
        and (measured > best if direction == "higher" else measured < best)
        and not _close(measured, best)
    )
    if violation and status != "model_invalid":
        error("measured_value beats theoretical_best_value; retain it and set status=model_invalid")

    components = point.get("components")
    incomplete = not isinstance(components, list) or not components
    if not isinstance(components, list):
        error("components must be a list")
        components = []
    ids: set[str] = set()
    convergable_components = bool(components)
    for index, component in enumerate(components):
        name = f"components[{index}]"
        if not isinstance(component, dict):
            error(f"{name} must be a mapping")
            incomplete, convergable_components = True, False
            continue
        ident = component.get("id")
        if not _text(ident) or ident in ids:
            error(f"{name}.id must be nonempty and unique")
        else:
            ids.add(ident)
        for field in ("measured_ms", "theoretical_best_ms"):
            value = component.get(field)
            if field not in component or value is not None and not _number(value, zero=True):
                error(f"{name}.{field} must be a nonnegative finite number or null")
            if value is None:
                incomplete = True
        kind = component.get("gap_kind")
        if not isinstance(kind, str) or kind not in _GAP_KINDS:
            error(f"{name}.gap_kind must be one of {', '.join(sorted(_GAP_KINDS))}")
        if not _strings(component.get("evidence"), nonempty=True):
            error(f"{name}.evidence must be a nonempty list of strings")
        unresolved = isinstance(kind, str) and kind in _UNRESOLVED_KINDS
        if kind == "model_error" and status != "model_invalid":
            error(f"{name}: model_error requires status=model_invalid")
        if (
            unresolved
            or component.get("measured_ms") is None
            or component.get("theoretical_best_ms") is None
        ):
            if not _text(component.get("next_test")):
                error(f"{name}.next_test is required for unknown or restricted gaps")
        actual, floor = component.get("measured_ms"), component.get("theoretical_best_ms")
        known = _number(actual, zero=True) and _number(floor, zero=True)
        if known and actual < floor and not _close(actual, floor):
            violation = True
            if status != "model_invalid":
                error(f"{name}: measured time beats its lower bound; set status=model_invalid")
        if (
            not known
            or unresolved
            or kind != "physical_limit"
            and not (known and _close(actual, floor))
        ):
            convergable_components = False

    uses_timing = "measured_ms" in point or "theoretical_best_ms" in point
    if uses_timing:
        if not _text(point.get("timing_derivation")):
            error("timing_derivation must explain overlap removal and target-metric conversion")
        for field in ("measured_ms", "theoretical_best_ms"):
            total = point.get(field)
            if field not in point or total is not None and not _number(total, zero=True):
                error(f"{field} must be a nonnegative finite number or null")
            values = [item.get(field) for item in components if isinstance(item, dict)]
            complete = (
                bool(values)
                and len(values) == len(components)
                and all(_number(value, zero=True) for value in values)
            )
            if complete and _number(total, zero=True) and not _close(sum(values), total):
                error(f"{field} does not equal the sum of disjoint critical-path components")
            if field == "theoretical_best_ms" and not complete and total is not None:
                error("theoretical_best_ms must be null while component bounds are incomplete")
        actual, floor = point.get("measured_ms"), point.get("theoretical_best_ms")
        if _number(actual, zero=True) and _number(floor, zero=True):
            if actual < floor and not _close(actual, floor):
                violation = True
                if status != "model_invalid":
                    error("measured_ms beats theoretical_best_ms; set status=model_invalid")
        else:
            incomplete = True

    restricted = isinstance(status, str) and status in {
        "model_invalid",
        "measurement_limited",
        "scope_limited",
    }
    if incomplete or not known_metrics or restricted:
        for field in ("unexplained", "next_test"):
            if not _text(point.get(field)):
                error(f"{field} is required for incomplete, unknown, or restricted models")
    elif _text(point.get("unexplained")) and not _text(point.get("next_test")):
        error("next_test is required when unexplained is nonempty")

    if status == "converged":
        if (
            not _number(point.get("measured_ms"), zero=True)
            or not _number(point.get("theoretical_best_ms"), zero=True)
            or not _text(point.get("timing_derivation"))
        ):
            error(
                "converged requires numeric measured_ms and theoretical_best_ms totals "
                "with timing_derivation"
            )
        tolerance = point.get("convergence_tolerance_pct")
        if not _number(tolerance, zero=True) or tolerance > 5:
            error("converged requires convergence_tolerance_pct between 0 and 5")
        if not _strings(point.get("convergence_evidence"), nonempty=True):
            error("converged requires convergence_evidence for a physically justified tolerance")
        if (
            not known_metrics
            or violation
            or incomplete
            or not convergable_components
            or _text(point.get("unexplained"))
        ):
            error(
                "converged requires known, nonviolated bounds and complete physically explained gaps"
            )
        if known_metrics and _number(tolerance, zero=True):
            gap_pct = abs(measured - best) / best * 100
            if gap_pct > tolerance + 1e-9:
                error(f"remaining gap {gap_pct:.6g}% exceeds convergence tolerance {tolerance}%")
    return errors


def _validate_model(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return ["performance model must be a mapping"]
    errors: list[str] = []
    if type(data.get("version")) is not int or data["version"] != MODEL_VERSION:
        errors.append(f"version must be {MODEL_VERSION}")
    for name in ("model_id", "target_metric"):
        if not _text(data.get(name)):
            errors.append(f"{name} must be a nonempty string")
    direction = data.get("direction")
    if direction not in ("higher", "lower"):
        errors.append("direction must be higher or lower")
    points = data.get("points")
    if not isinstance(points, list) or not points:
        return errors + ["points must be a nonempty list"]
    seen: set[int | None] = set()
    for index, point in enumerate(points):
        prefix = f"points[{index}]"
        if not isinstance(point, dict):
            errors.append(f"{prefix} must be a mapping")
            continue
        concurrency = point.get("concurrency")
        if "concurrency" not in point or not _concurrency(concurrency):
            errors.append(f"{prefix}.concurrency must be a positive integer or null")
        elif concurrency in seen:
            errors.append(f"{prefix}.concurrency duplicates {concurrency}")
        else:
            seen.add(concurrency)
        errors.extend(_validate_point(point, prefix, direction))
    return errors


def load_model(path: str | Path) -> dict[str, Any]:
    """Read and validate a model without altering its measurements or unknowns."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ModelError(f"Cannot read {path}: {exc}") from exc
    errors = _validate_model(data)
    if errors:
        raise ModelError("; ".join(errors))
    return data


def validate_task_model(data: Any, *, metric: str, concurrencies: list[int | None]) -> list[str]:
    """Return schema errors and mismatches against all scored benchmark points."""
    errors = _validate_model(data)
    if not isinstance(data, dict):
        return errors
    if data.get("target_metric") != metric:
        errors.append(f"target_metric must match task metric {metric!r}")
    expected_direction = "lower" if metric.endswith("_ms") else "higher"
    if data.get("direction") != expected_direction:
        errors.append(f"direction must be {expected_direction!r} for task metric {metric!r}")
    points = data.get("points")
    if isinstance(points, list):
        actual = {
            point["concurrency"]
            for point in points
            if isinstance(point, dict)
            and "concurrency" in point
            and _concurrency(point["concurrency"])
        }
        expected = set(concurrencies)
        if actual != expected:
            errors.append(
                f"points must cover exactly task concurrencies {concurrencies!r}; "
                f"missing={sorted(expected - actual, key=str)!r}, "
                f"unexpected={sorted(actual - expected, key=str)!r}"
            )
    return errors


def convergence_status(data: Any, focus_concurrencies: list[int | None] | None = None) -> str:
    """Derive status across scored points; invalid or absent evidence never converges."""
    if _validate_model(data):
        return "model_invalid"
    points = data["points"]
    if focus_concurrencies is not None:
        selected = set(focus_concurrencies)
        points = [point for point in points if point["concurrency"] in selected]
        if not points or {point["concurrency"] for point in points} != selected:
            return "model_invalid"
    statuses = {point["status"] for point in points}
    if "model_invalid" in statuses:
        return "model_invalid"
    if statuses == {"converged"}:
        return "converged"
    if "open" in statuses:
        return "open"
    if "measurement_limited" in statuses:
        return "measurement_limited"
    if "scope_limited" in statuses:
        return "scope_limited"
    return "open"
