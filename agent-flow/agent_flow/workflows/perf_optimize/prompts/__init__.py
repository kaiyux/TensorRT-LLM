import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agent_flow.workflows.perf_analyze.prompts import (
    PROMPTS_DIRNAME,
    build_remote_execution_context,
    dump_prompt_bundle,
)

from ._common import (
    DISAGG_ANALYSIS_CONTEXT,
    DISAGG_CAMPAIGN,
    EXECUTION_SLURM_BOOTSTRAP,
    KERNEL_COVERAGE_REPORTER_GUIDANCE,
    REMOTE_SLURM_EXECUTION,
    SOL_ANALYZER_CONTEXT,
    SOL_OPTIMIZE_REPORTER_GUIDANCE,
    SOL_OPTIMIZER_CONTEXT,
    SOL_PROFILER_CONTEXT,
    approach_restriction_note,
    kernel_coverage_analyzer_note,
    kernel_coverage_ncu_targeting,
)
from .analyzer import SYSTEM_PROMPT as ANALYZER_SYSTEM_PROMPT
from .analyzer import build_analyzer_prompt
from .benchmarker import SYSTEM_PROMPT as BENCHMARKER_SYSTEM_PROMPT
from .evaluator import SYSTEM_PROMPT as EVALUATOR_SYSTEM_PROMPT
from .integrator import SYSTEM_PROMPT as INTEGRATOR_SYSTEM_PROMPT
from .optimizer import SYSTEM_PROMPT as OPTIMIZER_SYSTEM_PROMPT
from .profiler import SYSTEM_PROMPT as PROFILER_SYSTEM_PROMPT
from .profiler import build_profiler_prompt
from .projector import SYSTEM_PROMPT as PROJECTOR_SYSTEM_PROMPT
from .projector import build_projector_prompt
from .qa import SYSTEM_PROMPT as QA_SYSTEM_PROMPT
from .reporter import SYSTEM_PROMPT as REPORTER_SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptBundle:
    """System prompts for the nine agents in ``PerfOptimizeWorkflow``.

    Pass a custom bundle to ``PerfOptimizeWorkflow(..., prompts=...)``
    to swap or extend the default prompts; use ``with_extensions`` to
    derive a bundle that appends domain-specific guidance to the defaults.
    """

    benchmarker: str
    projector: str
    analyzer: str
    optimizer: str
    evaluator: str
    integrator: str
    qa: str
    reporter: str
    profiler: str = PROFILER_SYSTEM_PROMPT

    def with_extensions(
        self,
        *,
        benchmarker: str = "",
        projector: str = "",
        profiler: str = "",
        analyzer: str = "",
        optimizer: str = "",
        evaluator: str = "",
        integrator: str = "",
        qa: str = "",
        reporter: str = "",
    ) -> "PromptBundle":
        """Return a new bundle with each non-empty extension appended.

        Empty / whitespace-only extensions leave the corresponding base
        prompt unchanged. Non-empty extensions are joined to the base with
        a single blank line separator.
        """

        def _append(base: str, extra: str) -> str:
            if not extra.strip():
                return base
            return base.rstrip() + "\n\n" + extra

        return PromptBundle(
            benchmarker=_append(self.benchmarker, benchmarker),
            projector=_append(self.projector, projector),
            profiler=_append(self.profiler, profiler),
            analyzer=_append(self.analyzer, analyzer),
            optimizer=_append(self.optimizer, optimizer),
            evaluator=_append(self.evaluator, evaluator),
            integrator=_append(self.integrator, integrator),
            qa=_append(self.qa, qa),
            reporter=_append(self.reporter, reporter),
        )


DEFAULT_PROMPTS = PromptBundle(
    benchmarker=BENCHMARKER_SYSTEM_PROMPT,
    projector=PROJECTOR_SYSTEM_PROMPT,
    profiler=PROFILER_SYSTEM_PROMPT,
    analyzer=ANALYZER_SYSTEM_PROMPT,
    optimizer=OPTIMIZER_SYSTEM_PROMPT,
    evaluator=EVALUATOR_SYSTEM_PROMPT,
    integrator=INTEGRATOR_SYSTEM_PROMPT,
    qa=QA_SYSTEM_PROMPT,
    reporter=REPORTER_SYSTEM_PROMPT,
)


def build_perf_optimize_prompts(
    include_slurm_environment: bool = False,
    remote_execution: Mapping[str, Any] | None = None,
    campaign_name: str = "perf-optimize",
    approaches: Sequence[str] | None = None,
    include_sol: bool = False,
    kernel_coverage: Mapping[str, Any] | None = None,
    sol_methodology: str = "full",
    include_disagg: bool = False,
) -> PromptBundle:
    """Return the workflow's prompt bundle, augmented per the task spec.

    When ``include_slurm_environment`` is True (the task spec carries a
    ``slurm-environment`` block), the Slurm container-bootstrap guidance
    is appended to every role that launches servers, including the profiler.
    The analyzer works offline; the reporter synthesizes existing artifacts;
    the projector launches no servers either (under Slurm it runs on
    the login node and records the latency constants as unmeasured, per
    its own prompt).

    ``remote_execution`` is the resolved task spec. When it names an SSH
    target, a short remote boundary plus its task-specific connection and
    Slurm values is appended to runtime-executing roles. The analyzer
    consumes staged artifacts locally and gets no launch instructions.

    When ``approaches`` (``optimize.approaches`` from the task spec)
    restricts the run to a subset of the roadmap's approach values, the
    restriction note is appended to every role that plans, applies, or
    judges roadmap items — analyzer, optimizer, evaluator. (QA only
    verifies the final state, so the restriction does not concern it;
    the projector never touches roadmap items.)
    ``None`` or the full set leaves the prompts unchanged.

    When ``include_sol`` is True (the projector stage is enabled — the
    default, unless the task spec sets ``sol.enabled: false``), the
    SOL-consumption guidance is
    appended to the analyzer (rank roadmap items against the projected
    headroom and bound mix, and attribute any remaining gap before
    leaving the roadmap exhausted), the optimizer (aim each item's
    realization at the binding ceiling — context, never an expansion of
    the item), and the reporter (the "Projection vs Measured" section
    with its remaining-gap accountability breakdown). The profiler gets
    only missing-constant calibration and snapshot guidance. The projector's
    own SOL prompt is always in the bundle — the stage gate lives in
    the workflow. The evaluator and QA deliberately get no SOL context:
    their gates are measured-vs-measured with deterministic thresholds,
    and an analytical ceiling as context could anchor a fresh-eyes
    verdict on a model instead of the measurements. (The evaluator's
    projection-free contribution to gap accountability is the *Gap
    implication* line its negative verdicts always carry.)

    ``sol_methodology`` is ``"reduced"`` when this session has
    ``perf-analysis`` but not ``internal-perf-sol-analysis`` (resolved
    before the run by perf-analyze's
    ``sol_methodology.resolve_sol_methodology``, which this workflow
    shares); it appends the projector's fallback block and changes
    nothing else.

    When ``kernel_coverage`` is set (the validated
    ``profile.kernel_coverage`` block — the per-kernel coverage
    contract), the profiler gets coverage-driven ncu targeting and the
    analyzer gets the four per-kernel questions (eliminable? faster? fusible?
    overlappable?), and the
    ``kernel_ledger.yaml`` contract with the task's bars interpolated and a
    best theoretical performance model updated from evidence every turn;
    the reporter gets the "Kernel Coverage" accountability section. The
    other roles are unchanged — the ledger is authored by the analyzer
    and consumed by the reporter, with the orchestrator's deterministic
    validation in between.

    When ``include_disagg`` is True (the task spec carries a ``disagg``
    block), the disaggregated-serving section is appended to every role
    that launches or measures a server. It supersedes the single-server
    lifecycle, the tuning-config note and the profiling runs those roles
    otherwise follow, so it is composed last. The analyzer gets only
    topology and interpretation context. The reporter and projector are
    left alone: neither stands up a server.

    Composing it here rather than carrying it unconditionally is what
    keeps the override unambiguous — a role either has the section and it
    applies, or it does not have it at all. The alternative (always
    present, gated on a sentence telling the agent to check
    ``task.yaml``) makes every aggregate campaign pay for it and turns a
    deployment-time fact into a per-turn inference the agent can get
    wrong.
    """
    bundle = DEFAULT_PROMPTS
    if kernel_coverage is not None:
        bundle = dataclasses.replace(
            bundle,
            profiler=build_profiler_prompt(
                ncu_targeting=kernel_coverage_ncu_targeting(
                    float(kernel_coverage["min_share_pct"]),
                    float(kernel_coverage["coverage_target_pct"]),
                )
            ),
        )
    if sol_methodology != "full":
        bundle = dataclasses.replace(bundle, projector=build_projector_prompt(sol_methodology))
    restriction = approach_restriction_note(approaches) if approaches is not None else ""
    if restriction:
        bundle = bundle.with_extensions(
            analyzer=approach_restriction_note(approaches or (), analyzer_only=True),
            optimizer=restriction,
            evaluator=restriction,
        )
    if include_slurm_environment:
        bundle = bundle.with_extensions(
            benchmarker=EXECUTION_SLURM_BOOTSTRAP,
            profiler=EXECUTION_SLURM_BOOTSTRAP,
            optimizer=EXECUTION_SLURM_BOOTSTRAP,
            evaluator=EXECUTION_SLURM_BOOTSTRAP,
            integrator=EXECUTION_SLURM_BOOTSTRAP,
            qa=EXECUTION_SLURM_BOOTSTRAP,
        )
    if include_sol:
        bundle = bundle.with_extensions(
            profiler=SOL_PROFILER_CONTEXT,
            analyzer=SOL_ANALYZER_CONTEXT,
            optimizer=SOL_OPTIMIZER_CONTEXT,
            reporter=SOL_OPTIMIZE_REPORTER_GUIDANCE,
        )
    if kernel_coverage is not None:
        bundle = bundle.with_extensions(
            analyzer=kernel_coverage_analyzer_note(
                float(kernel_coverage["min_share_pct"]),
                float(kernel_coverage["coverage_target_pct"]),
            ),
            reporter=KERNEL_COVERAGE_REPORTER_GUIDANCE,
        )
    if include_disagg:
        bundle = bundle.with_extensions(
            benchmarker=DISAGG_CAMPAIGN,
            profiler=DISAGG_CAMPAIGN,
            analyzer=DISAGG_ANALYSIS_CONTEXT,
            optimizer=DISAGG_CAMPAIGN,
            evaluator=DISAGG_CAMPAIGN,
            integrator=DISAGG_CAMPAIGN,
            qa=DISAGG_CAMPAIGN,
        )
    context = build_remote_execution_context(remote_execution, campaign_name)
    if context:
        bundle = bundle.with_extensions(
            benchmarker=context,
            projector=context,
            profiler=context,
            optimizer=context,
            evaluator=context,
            qa=context,
        )
    return bundle


__all__ = [
    "ANALYZER_SYSTEM_PROMPT",
    "BENCHMARKER_SYSTEM_PROMPT",
    "DEFAULT_PROMPTS",
    "EVALUATOR_SYSTEM_PROMPT",
    "INTEGRATOR_SYSTEM_PROMPT",
    "OPTIMIZER_SYSTEM_PROMPT",
    "PROFILER_SYSTEM_PROMPT",
    "PROJECTOR_SYSTEM_PROMPT",
    "PROMPTS_DIRNAME",
    "REMOTE_SLURM_EXECUTION",
    "PromptBundle",
    "QA_SYSTEM_PROMPT",
    "REPORTER_SYSTEM_PROMPT",
    "build_perf_optimize_prompts",
    "build_analyzer_prompt",
    "build_profiler_prompt",
    "build_projector_prompt",
    "dump_prompt_bundle",
]
