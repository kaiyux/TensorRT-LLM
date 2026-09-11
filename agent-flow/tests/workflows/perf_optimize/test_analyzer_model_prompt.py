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

"""Keep model and report ownership consistent across Analyzer configurations."""

import re

import pytest

from agent_flow.workflows.perf_analyze.prompts.analyzer import SYSTEM_PROMPT as ANALYZE_ANALYZER
from agent_flow.workflows.perf_analyze.prompts.projector import SYSTEM_PROMPT as ANALYZE_PROJECTOR
from agent_flow.workflows.perf_analyze.prompts.reporter import SYSTEM_PROMPT as ANALYZE_REPORTER
from agent_flow.workflows.perf_optimize.prompts.analyzer import build_analyzer_prompt
from agent_flow.workflows.perf_optimize.prompts.profiler import build_profiler_prompt
from agent_flow.workflows.perf_optimize.prompts.projector import SYSTEM_PROMPT as OPTIMIZE_PROJECTOR
from agent_flow.workflows.perf_optimize.prompts.reporter import SYSTEM_PROMPT as OPTIMIZE_REPORTER


@pytest.mark.parametrize("include_per_layer_model", [False, True])
def test_central_model_and_compact_analysis_are_always_required(include_per_layer_model):
    prompt = build_analyzer_prompt(include_per_layer_model=include_per_layer_model)
    template = prompt.split("# Analysis: <model>", 1)[1].split("```", 1)[0]

    assert re.findall(r"^## (.+)$", template, flags=re.MULTILINE) == [
        "Result",
        "Theoretical performance model",
        "Gap analysis",
        "Next actions",
    ]
    assert "Machine-readable contract (`performance_model.yaml`, version 1)" in prompt
    assert "## Per-layer theoretical performance model" not in prompt
    assert "## SOL correlation (measured vs ceiling)" not in prompt


def test_replan_updates_current_model_without_copying_imported_report():
    prompt = " ".join(build_analyzer_prompt().split())
    replan = prompt.split("**Replan-only round**", 1)[1].split("**Final reconciliation mode**", 1)[
        0
    ]

    assert "update the current model" in replan
    assert "standing analysis and evaluator verdicts" in replan
    assert "preserving measurement provenance" in replan
    assert "same four-section analysis" in replan
    assert "link imported reports rather than copying or appending them" in replan
    assert "Do not regenerate measured artifacts" in replan


def test_combined_analyzer_records_capture_separately_from_model_analysis():
    capture = " ".join(ANALYZE_ANALYZER.split("## Capture record", 1)[1].split())

    for heading in ("Capture", "Coverage", "Timing provenance"):
        assert f"`## {heading}`" in capture
    assert "Keep model tables, rankings and recommendations in `analysis.md`" in capture
    for artifact in ("analysis.md", "performance_model.yaml", "profiler_report.md"):
        assert artifact in ANALYZE_ANALYZER


@pytest.mark.parametrize(
    "prompt", [ANALYZE_PROJECTOR, OPTIMIZE_PROJECTOR], ids=["analyze", "optimize"]
)
def test_initial_projection_has_one_conditional_model_table(prompt):
    template = prompt.split("# SOL Projection: <model name>", 1)[1].split("```", 1)[0]

    assert re.findall(r"^## (.+)$", template, flags=re.MULTILINE) == [
        "Result",
        "Projection setup",
        "Initial theoretical performance model",
        "Open questions",
    ]
    assert sum(line.startswith("| --- |") for line in template.splitlines()) == 1
    assert "current model for gap accounting and convergence" in " ".join(prompt.split())
    assert "coverage versus a kernel/phase proxy" in template
    assert "unavailable bounds remain unavailable" in template


@pytest.mark.parametrize(
    "prompt", [ANALYZE_PROJECTOR, OPTIMIZE_PROJECTOR], ids=["analyze", "optimize"]
)
def test_projector_methodology_keeps_partial_predictions_and_derivations_scoped(prompt):
    methodology = " ".join(
        prompt.split("## The methodology:", 1)[1].split("## Report", 1)[0].split()
    )

    assert "current model for gap accounting and convergence" in " ".join(prompt.split())
    assert "initial provenance" in prompt
    assert "linked `sol_work/` derivations with actual inputs and units" in methodology
    assert "kernel execution plus per-launch latency only" in methodology
    assert "before claiming an end-to-end ceiling" in methodology
    assert "Unmatched residuals remain unexplained" in methodology
    assert "ratio above 100% signals a model/measurement mismatch" in methodology
    assert "*Caveats*" not in methodology
    assert "therefore points at host/scheduling costs" not in methodology


def test_requested_capture_windows_override_defaults_with_observed_phase_provenance():
    profiler = " ".join(build_profiler_prompt().split())

    assert "Default to the configured `profile.nsys_iter_range`" in profiler
    assert "When the turn explicitly requests a different operating point or phase" in profiler
    assert "may select a different `TLLM_PROFILE_START_STOP` window" in profiler
    assert "configured and effective windows" in profiler
    assert "verify its trace/phase markers" in profiler
    assert "With no explicit phase request, require a usable steady-state decode window" in profiler
    assert "judge coverage against that requested phase and operating point" in profiler
    assert "Keep task and serving config read-only" in profiler
    assert "may select a different `TLLM_PROFILE_START_STOP` window" not in " ".join(
        build_analyzer_prompt().split()
    )


@pytest.mark.parametrize(
    "prompt", [ANALYZE_REPORTER, OPTIMIZE_REPORTER], ids=["analyze", "optimize"]
)
def test_reporter_reassesses_convergence_against_current_verification(prompt):
    rigor = " ".join(prompt.split())

    assert "compatible independent final verification" in rigor
    assert "`convergence_tolerance_pct`" in rigor
    assert "`convergence_evidence`" in rigor
    assert "A gap above tolerance requires `open`" in rigor
    assert "cannot inherit convergence" in rigor
    assert "requires `model_invalid`" in rigor
    assert "every scored point must satisfy" in rigor
    assert "without editing performance_model.yaml or changing its bound" in rigor


def test_final_reconciliation_updates_only_model_and_analysis_from_final_evidence():
    prompt = " ".join(build_analyzer_prompt().split())
    final_mode = prompt.split("**Final reconciliation mode**", 1)[1].split(
        "In optimization rounds", 1
    )[0]

    assert "latest round's model/analysis" in final_mode
    assert "`final_verification/verification_report.md` and result JSONs" in final_mode
    assert "accepted changes and evaluation/integration evidence" in final_mode
    assert "Write only the model and analysis" in final_mode
    assert "`final_verification/analysis/`" in prompt
    assert "Keep `roadmap.yaml` and kernel ledgers read-only" in final_mode
    assert (
        "overrides roadmap authoring, the dormant-capability sweep and fresh-ledger duties"
        in final_mode
    )
    assert "preserve supported structural bounds" in final_mode
    assert "leave final component timings unknown when unprofiled" in final_mode
    assert "Acceptance does not prove a mechanism or make an old profile current" in final_mode
    assert "Apply the shared model's mismatch and convergence rules" in final_mode
    assert "Never apply optimizations or launch servers" in prompt


def test_reporter_prefers_final_reconciled_model_over_latest_round():
    inputs = " ".join(OPTIMIZE_REPORTER.split("## Inputs", 1)[1].split())

    assert "Prefer `final_verification/analysis/{analysis.md,performance_model.yaml}`" in inputs
    assert "final reconciliation is authoritative" in inputs
    assert "Otherwise use the latest completed `rounds/round_<n>/analysis/` model" in inputs
    assert "disclose missing final reconciliation" in inputs
