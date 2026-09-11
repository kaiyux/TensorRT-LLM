"""Performance-optimize workflow built on ``agent_flow.AgentLayer``.

The applying counterpart to perf-analyze. A baseline benchmark and optional
one-shot SOL projection initialize the campaign. A conditional profiler writes
``profiler_report.md`` and ``profile_manifest.json``; an offline analyzer writes
``analysis.md``, updates the current ``performance_model.yaml``, and ranks
``roadmap.yaml`` items. The model is updated every turn, including reuse and
replan-only analysis, and connects measured performance to the current
theoretical best on the same workload and timing basis.

Isolated optimizer/evaluator pairs implement and measure selected items.
Serial mode promotes approved candidates directly; parallel mode combines them
through an integrator. Acceptance gates remain measured-versus-measured.
The orchestrator stops at the configured budget, an exhausted roadmap with no
outstanding measurement request, or an optional measured improvement target. A stop condition does not establish
model convergence. Stateless QA verifies the final accepted state, then the same
analyzer reconciles the current bounds and QA measurements in a checkpointed
``final_analyzer`` stage. It writes nonempty ``analysis.md`` and a validated
``performance_model.yaml`` under ``final_verification/analysis/``, using saved
evidence without GPU work or roadmap changes. A reporter reads that final model
and writes concise ``optimization_report.md`` / ``.html`` deliverables
covering result, current model, remaining gap, and next actions. All nine roles
run on the Claude Code backend.

Public surface:

- :class:`PerfOptimizeWorkflow` — the orchestrator for the
  benchmarker -> (projector) -> [(profiler) -> analyzer -> serial/parallel
  (optimizer <-> evaluator) items -> optional integrator] x rounds -> qa ->
  final_analyzer -> reporter loop.
- :class:`PromptBundle`, :data:`DEFAULT_PROMPTS`, and
  :func:`build_perf_optimize_prompts` — prompt bundle and helpers for
  the workflow's nine agents.
- ``STAGE_*`` constants — stage identifiers used by the checkpoint schema
  (``<workspace>/.perf_optimize_state.json``).
"""

from typing import Any

from .prompts import DEFAULT_PROMPTS, PromptBundle, build_perf_optimize_prompts
from .state import (
    STAGE_ANALYZER,
    STAGE_BENCHMARKER,
    STAGE_EVALUATOR,
    STAGE_FINAL_ANALYZER,
    STAGE_INTEGRATOR,
    STAGE_OPTIMIZER,
    STAGE_OPTIMIZER_EVALUATOR,
    STAGE_PROFILER,
    STAGE_PROJECTOR,
    STAGE_QA,
    STAGE_REPORTER,
)

__all__ = [
    "DEFAULT_PROMPTS",
    "PerfOptimizeWorkflow",
    "PromptBundle",
    "STAGE_ANALYZER",
    "STAGE_BENCHMARKER",
    "STAGE_EVALUATOR",
    "STAGE_FINAL_ANALYZER",
    "STAGE_INTEGRATOR",
    "STAGE_OPTIMIZER",
    "STAGE_OPTIMIZER_EVALUATOR",
    "STAGE_PROFILER",
    "STAGE_PROJECTOR",
    "STAGE_QA",
    "STAGE_REPORTER",
    "build_perf_optimize_prompts",
]


def __getattr__(name: str) -> Any:
    if name == "PerfOptimizeWorkflow":
        from .workflow import PerfOptimizeWorkflow

        return PerfOptimizeWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
