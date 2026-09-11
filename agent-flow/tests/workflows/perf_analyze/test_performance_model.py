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

"""Checks for conservative theoretical-best accounting and convergence gates."""

from copy import deepcopy

import pytest
import yaml

from agent_flow.workflows.perf_analyze.performance_model import (
    MODEL_FILENAME,
    MODEL_VERSION,
    ModelError,
    convergence_status,
    load_model,
    validate_task_model,
)


def _model(*concurrencies):
    return {
        "version": MODEL_VERSION,
        "model_id": "round-4-final",
        "target_metric": "output_throughput",
        "direction": "higher",
        "points": [
            {
                "concurrency": concurrency,
                "operating_point": {
                    "build": "commit-123/cuda-13",
                    "hardware": "8 GPUs, NVLink",
                    "workload": "1024 input, 128 output tokens",
                    "timing_basis": "full benchmark wall time",
                },
                "measured_value": 100.0,
                "theoretical_best_value": 200.0,
                "derivation": "Output tokens / disjoint modeled critical-path time.",
                "assumptions": ["Weights stay resident."],
                "evidence": ["bounds.json"],
                "measurement_evidence": ["benchmark.json"],
                "components": [
                    {
                        "id": "decode",
                        "measured_ms": 10.0,
                        "theoretical_best_ms": 5.0,
                        "gap_kind": "actionable",
                        "evidence": ["trace.sqlite:decode"],
                    }
                ],
                "status": "open",
                "unexplained": "",
                "next_test": "Fuse the decode kernels and remeasure.",
            }
            for concurrency in (concurrencies or (64,))
        ],
    }


def _errors(model, concurrencies=None, metric="output_throughput"):
    return validate_task_model(model, metric=metric, concurrencies=concurrencies or [64])


def _converge(point):
    point.update(
        measured_value=199.0,
        status="converged",
        convergence_tolerance_pct=1.0,
        convergence_evidence=["Packet granularity accounts for a 0.5% physical floor."],
    )
    point["components"][0].update(
        measured_ms=5.025125628,
        gap_kind="physical_limit",
    )
    point.update(
        measured_ms=5.025125628,
        theoretical_best_ms=5.0,
        timing_derivation="Disjoint wall-time regions; throughput = 1000 tokens / time.",
    )


def test_load_preserves_model_and_checks_task_coverage(tmp_path):
    model = _model(64, 128)
    path = tmp_path / MODEL_FILENAME
    path.write_text(yaml.safe_dump(model))
    assert load_model(path) == model
    assert _errors(model, [128, 64]) == []
    assert convergence_status(model) == "open"
    assert "exactly task concurrencies" in ";".join(_errors(model, [64, 256]))
    assert "target_metric" in ";".join(
        validate_task_model(model, metric="latency", concurrencies=[64, 128])
    )


def test_scalar_null_concurrency():
    model = _model(None)
    assert _errors(model, [None]) == []
    assert convergence_status(model, [None]) == "open"


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("output_throughput", "higher"),
        ("request_throughput", "higher"),
        ("p99_latency_ms", "lower"),
    ],
)
def test_direction_matches_task_metric(metric, expected):
    model = _model()
    model.update(target_metric=metric, direction=expected)
    if expected == "lower":
        model["points"][0]["measured_value"] = 400
    assert _errors(model, metric=metric) == []
    model["direction"] = "higher" if expected == "lower" else "lower"
    assert "direction must be" in ";".join(_errors(model, metric=metric))


@pytest.mark.parametrize("value", [None, [], "model", 0])
def test_non_mapping_is_reported(value):
    assert _errors(value)
    assert convergence_status(value) == "model_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("version", 2),
        ("model_id", " "),
        ("direction", []),
        ("direction", "larger"),
        ("points", []),
        ("points", [None]),
        ("points", "invalid"),
    ],
)
def test_invalid_top_level_shapes(field, value):
    model = _model()
    model[field] = value
    assert _errors(model)
    assert convergence_status(model) == "model_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("concurrency", True),
        ("concurrency", 64.0),
        ("concurrency", []),
        ("measured_value", 0),
        ("measured_value", True),
        ("theoretical_best_value", float("nan")),
        ("theoretical_best_value", float("inf")),
        ("operating_point", {"build": "x"}),
        ("derivation", ""),
        ("assumptions", "resident"),
        ("measurement_evidence", []),
        ("evidence", []),
        ("status", []),
        ("components", [None]),
        ("unexplained", None),
    ],
)
def test_invalid_point_shapes(field, value):
    model = _model()
    model["points"][0][field] = value
    assert _errors(model)
    assert convergence_status(model) == "model_invalid"


def test_duplicate_points_and_component_ids():
    model = _model(64, 64)
    assert "duplicates" in ";".join(_errors(model))
    model = _model()
    components = model["points"][0]["components"]
    components.append(deepcopy(components[0]))
    assert "unique" in ";".join(_errors(model))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", []),
        ("measured_ms", -1),
        ("theoretical_best_ms", float("inf")),
        ("gap_kind", []),
        ("evidence", []),
    ],
)
def test_invalid_component_shapes(field, value):
    model = _model()
    model["points"][0]["components"][0][field] = value
    assert _errors(model)
    assert convergence_status(model) == "model_invalid"


def test_unknown_model_remains_open_and_does_not_invent_zero_bounds(tmp_path):
    model = _model()
    point = model["points"][0]
    point.update(
        theoretical_best_value=None,
        evidence=[],
        components=[],
        unexplained="No production profile yet.",
        next_test="Capture the full serving workload.",
    )
    assert _errors(model) == []
    path = tmp_path / MODEL_FILENAME
    path.write_text(yaml.safe_dump(model))
    assert load_model(path)["points"][0]["theoretical_best_value"] is None
    assert convergence_status(model) == "open"
    point["theoretical_best_value"] = 0
    assert _errors(model)
    point["theoretical_best_value"] = None
    point["unexplained"] = ""
    assert "unexplained is required" in ";".join(_errors(model))


def test_unmeasured_point_requires_explanation():
    model = _model()
    point = model["points"][0]
    point.update(measured_value=None, measurement_evidence=[])
    assert "unexplained is required" in ";".join(_errors(model))
    point.update(unexplained="Point has not been benchmarked.", next_test="Benchmark c=64.")
    assert _errors(model) == []


def _timing(point):
    point.update(
        measured_ms=10.0,
        theoretical_best_ms=5.0,
        timing_derivation="Disjoint wall-time regions; throughput = 1000 tokens / time.",
    )


def test_disjoint_timing_must_reconcile_with_components():
    model = _model()
    point = model["points"][0]
    _timing(point)
    assert _errors(model) == []
    point["measured_ms"] = 12
    assert "sum of disjoint" in ";".join(_errors(model))
    point["measured_ms"] = 10.001
    assert _errors(model) == []
    del point["timing_derivation"]
    assert "timing_derivation" in ";".join(_errors(model))


def test_partial_decomposition_cannot_claim_complete_theoretical_time():
    model = _model()
    point = model["points"][0]
    _timing(point)
    point.update(unexplained="Missing prefill bound.", next_test="Profile prefill.")
    point["components"][0].update(
        theoretical_best_ms=None,
        gap_kind="unexplained",
        next_test="Profile prefill.",
    )
    assert "must be null while component bounds" in ";".join(_errors(model))
    point["theoretical_best_ms"] = None
    assert _errors(model) == []
    assert convergence_status(model) == "open"


def test_component_bound_violations_and_model_errors_invalidate_point():
    model = _model()
    point = model["points"][0]
    point["components"][0]["theoretical_best_ms"] = 11
    assert "measured time beats its lower bound" in ";".join(_errors(model))
    point.update(status="model_invalid", unexplained="Incorrect component bound.")
    assert _errors(model) == []
    point["components"][0].update(
        theoretical_best_ms=5,
        gap_kind="model_error",
        next_test="Verify model operation counts.",
    )
    point["status"] = "open"
    assert "model_error requires status=model_invalid" in ";".join(_errors(model))


def test_rounding_tolerance_does_not_hide_material_bound_violation():
    model = _model()
    point = model["points"][0]
    point["measured_value"] = 200.1
    assert _errors(model) == []
    point["measured_value"] = 201
    assert "beats theoretical_best_value" in ";".join(_errors(model))


@pytest.mark.parametrize("direction,measured", [("higher", 220), ("lower", 180)])
def test_bound_violations_are_preserved_as_model_invalid(tmp_path, direction, measured):
    model = _model()
    model["direction"] = direction
    metric = "output_throughput" if direction == "higher" else "latency_ms"
    model["target_metric"] = metric
    point = model["points"][0]
    point["measured_value"] = measured
    assert "set status=model_invalid" in ";".join(_errors(model, metric=metric))
    point.update(
        status="model_invalid",
        unexplained="Measurement exceeds the assumed bound.",
        next_test="Reconcile operation count and hardware peaks.",
    )
    assert _errors(model, metric=metric) == []
    path = tmp_path / MODEL_FILENAME
    path.write_text(yaml.safe_dump(model))
    assert load_model(path)["points"][0]["measured_value"] == measured
    assert convergence_status(model) == "model_invalid"


def test_converged_requires_physical_evidence_and_small_gap():
    model = _model()
    point = model["points"][0]
    _converge(point)
    assert _errors(model) == []
    assert convergence_status(model) == "converged"
    point["measured_value"] = 180
    assert "exceeds convergence tolerance" in ";".join(_errors(model))
    assert convergence_status(model) == "model_invalid"


@pytest.mark.parametrize("field", ["measured_ms", "theoretical_best_ms", "timing_derivation"])
@pytest.mark.parametrize("missing", [True, False])
def test_convergence_requires_reconciled_numeric_timing(field, missing):
    model = _model()
    point = model["points"][0]
    _converge(point)
    if missing:
        del point[field]
    else:
        point[field] = None
    assert "converged requires numeric" in ";".join(_errors(model))
    assert convergence_status(model) == "model_invalid"


def test_convergence_requires_totals_equal_to_component_times():
    model = _model()
    point = model["points"][0]
    _converge(point)
    point["measured_ms"] = 7
    assert "sum of disjoint" in ";".join(_errors(model))
    assert convergence_status(model) == "model_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("convergence_tolerance_pct", 10),
        ("convergence_tolerance_pct", True),
        ("convergence_tolerance_pct", float("nan")),
        ("convergence_evidence", []),
        ("unexplained", "Unexplained CPU time."),
        ("components", []),
        ("theoretical_best_value", None),
    ],
)
def test_false_convergence_is_rejected(field, value):
    model = _model()
    point = model["points"][0]
    _converge(point)
    point[field] = value
    assert _errors(model)
    assert convergence_status(model) == "model_invalid"


@pytest.mark.parametrize(
    "kind", ["actionable", "scope_limited", "measurement_limited", "unexplained", "model_error"]
)
def test_remaining_nonphysical_gaps_cannot_converge(kind):
    model = _model()
    point = model["points"][0]
    _converge(point)
    point["components"][0].update(gap_kind=kind, next_test="Measure the unresolved gap.")
    assert _errors(model)
    assert convergence_status(model) == "model_invalid"


def test_aggregate_status_and_focus_do_not_hide_open_scored_points():
    model = _model(64, 128)
    _converge(model["points"][0])
    assert convergence_status(model) == "open"
    assert convergence_status(model, [64]) == "converged"
    assert convergence_status(model, [512]) == "model_invalid"
    assert convergence_status(model, []) == "model_invalid"
    point = model["points"][1]
    point.update(status="scope_limited", unexplained="Requires unsupported batch shape.")
    assert convergence_status(model) == "scope_limited"
    point["status"] = "measurement_limited"
    assert convergence_status(model) == "measurement_limited"
    point["status"] = "model_invalid"
    assert convergence_status(model) == "model_invalid"
    _converge(point)
    point["unexplained"] = ""
    assert convergence_status(model) == "converged"


@pytest.mark.parametrize("contents", ["[", "[]", "version: 8", ""])
def test_load_errors_are_actionable(tmp_path, contents):
    path = tmp_path / MODEL_FILENAME
    path.write_text(contents)
    with pytest.raises(ModelError):
        load_model(path)


def test_missing_model_raises_model_error(tmp_path):
    with pytest.raises(ModelError, match="Cannot read"):
        load_model(tmp_path / MODEL_FILENAME)
