# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Report contracts keep one current model and avoid duplicate analyses."""

from __future__ import annotations

import re

import pytest

from agent_flow.workflows.perf_analyze.prompts import build_perf_analyze_prompts
from agent_flow.workflows.perf_optimize.prompts import build_perf_optimize_prompts


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _report_sections(prompt: str) -> list[str]:
    template = prompt.split("```\n# ", 1)[1].split("```", 1)[0]
    return re.findall(r"^## (.+)$", template, re.MULTILINE)


@pytest.mark.parametrize("include_sol", [False, True])
@pytest.mark.parametrize("coverage", [None, {"min_share_pct": 0.5, "coverage_target_pct": 95.0}])
def test_optimization_report_has_one_model_and_four_sections(include_sol, coverage):
    prompt = build_perf_optimize_prompts(include_sol=include_sol, kernel_coverage=coverage).reporter
    assert _report_sections(prompt) == [
        "Result",
        "Theoretical performance model",
        "Gap analysis",
        "Changes and next actions",
    ]
    assert "performance_model.yaml" in prompt
    assert "analysis.md" in prompt
    assert "profile_findings.md" not in prompt
    for retired_section in (
        "Executive Summary",
        "Optimization Trajectory",
        "Kernel-Level Comparison",
        "Projection vs Measured",
        "Kernel Coverage",
        "Remaining Roadmap",
        "Durable facts for the next campaign",
    ):
        assert not re.search(rf"^## {re.escape(retired_section)}(?:$|\s*\()", prompt, re.MULTILINE)


@pytest.mark.parametrize("include_sol", [False, True])
def test_analysis_report_uses_current_model_without_duplicate_tables(include_sol):
    prompt = build_perf_analyze_prompts(include_sol=include_sol).reporter
    assert _report_sections(prompt) == [
        "Result",
        "Theoretical performance model",
        "Gap analysis",
        "Next actions",
    ]
    assert "performance_model.yaml" in prompt
    assert "analysis.md" in prompt
    assert "profile_findings.md" not in prompt
    assert "## Projection vs Measured" not in prompt
    assert "## Profiling Findings" not in prompt


@pytest.mark.parametrize("workflow", [build_perf_analyze_prompts, build_perf_optimize_prompts])
def test_report_gap_math_is_metric_direction_aware_and_uncertainty_preserving(workflow):
    prompt = _norm(workflow().reporter)
    for formula in (
        "attainment = measured / best",
        "attainment = best / measured",
        "improvement = best / measured - 1",
        "improvement = 1 - best / measured",
    ):
        assert formula in prompt
    for status in ("open", "converged", "measurement_limited", "scope_limited", "model_invalid"):
        assert f"`{status}`" in prompt
    assert "disjoint critical-path cost" in prompt
    assert "unresolved residual" in prompt


def test_optimization_report_reconciles_final_verification_and_candidate_references():
    prompt = _norm(build_perf_optimize_prompts().reporter)
    assert "final_verification/analysis/{analysis.md,performance_model.yaml}" in prompt
    assert "final_verification/verification_report.md" in prompt
    assert "Evaluator/integrator measurements are not independent verification" in prompt
    assert "A mismatched final build, workload or scope cannot inherit convergence" in prompt
    assert "Recompute gap and convergence from compatible independent final verification" in prompt
    assert "convergence_tolerance_pct" in prompt
    assert "measurement beyond the predicted best requires `model_invalid`" in prompt
    assert "never compute gain from the ratio of two curve means" in prompt
    assert "never add standalone gains" in prompt
    assert "gain below measurement resolution" in prompt
    assert "a short decode-window median" in prompt


def test_profiler_report_preserves_capture_only_ownership_and_requested_coverage():
    prompt = _norm(build_perf_optimize_prompts(include_sol=True).profiler)
    assert "## Capture report (`profiler_report.md`)" in prompt
    assert '"artifacts": ["profiler_report.md",' in prompt
    assert "**Capture**" in prompt
    assert "**Coverage**" in prompt
    assert "**Timing provenance**" in prompt
    assert "every requested point/phase/rank/kernel target" in prompt
    assert "reasons for missing evidence" in prompt
    assert "explicitly requested operating points and phases" in prompt
    assert "With no explicit request, replay the **largest**" in prompt
    assert "Separate prefill/mixed iterations from steady-state decode" in prompt
    assert "expose missing phases/concurrencies" in prompt
    assert "The Analyzer owns bottleneck rankings, optimization opportunities" in prompt
    assert "never apply optimizations" in prompt.lower()
    assert "profile_findings.md" not in prompt


def test_canonical_model_prompt_example_passes_runtime_validation(tmp_path):
    """The authoring example must satisfy the actual analyzer output gate."""
    from agent_flow.workflows.perf_analyze.performance_model import (
        convergence_status,
        load_model,
        validate_task_model,
    )
    from agent_flow.workflows.perf_analyze.prompts._common import PROFILE_FINDINGS_CONTRACT

    example = PROFILE_FINDINGS_CONTRACT.split("```yaml\n", 1)[1].split("```", 1)[0]
    path = tmp_path / "performance_model.yaml"
    path.write_text(example, encoding="utf-8")
    model = load_model(path)
    assert validate_task_model(model, metric="output_throughput", concurrencies=[64]) == []
    assert convergence_status(model) == "measurement_limited"
    assert model["points"][0]["theoretical_best_value"] is None
