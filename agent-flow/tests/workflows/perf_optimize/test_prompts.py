"""Tests that the perf-optimize role prompts carry their contracts.

The measuring roles must drive ``benchmark_serving.py`` / ``nsys`` from
the same canonical templates as perf-analyze; the mutating roles must
carry the git discipline and the acceptance gate; and every role that
touches ``roadmap.yaml`` must carry the roadmap contract (including the
orchestrator-owns-lifecycle rule).
"""

from __future__ import annotations

import re

import pytest

from agent_flow.workflows.perf_analyze.prompts._common import (
    SOL_METHODOLOGY_FALLBACK as _ANALYZE_METHODOLOGY_FALLBACK,
)
from agent_flow.workflows.perf_analyze.prompts._common import build_server_lifecycle
from agent_flow.workflows.perf_optimize.prompts import (
    ANALYZER_SYSTEM_PROMPT,
    BENCHMARKER_SYSTEM_PROMPT,
    DEFAULT_PROMPTS,
    EVALUATOR_SYSTEM_PROMPT,
    INTEGRATOR_SYSTEM_PROMPT,
    OPTIMIZER_SYSTEM_PROMPT,
    PROFILER_SYSTEM_PROMPT,
    PROJECTOR_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    REPORTER_SYSTEM_PROMPT,
    PromptBundle,
    build_perf_optimize_prompts,
    build_profiler_prompt,
    build_projector_prompt,
    dump_prompt_bundle,
)
from agent_flow.workflows.perf_optimize.prompts._common import (
    CASEBOOK_APPLY,
    EXPECTATION_GATE,
    GIT_DISCIPLINE,
    KERNEL_COVERAGE_REPORTER_GUIDANCE,
    KERNEL_REUSE,
    MEASUREMENT_PROTOCOL,
    OPTIMIZE_HTML_COMPANION,
    PROFILE_FINDINGS_CONTRACT,
    ROADMAP_SPEC,
    RUNTIME_CHECKOUT,
    SOL_ANALYZER_CONTEXT,
    SOL_METHODOLOGY_FALLBACK,
    SOL_OPTIMIZE_REPORTER_GUIDANCE,
    SOL_OPTIMIZER_CONTEXT,
    TUNING_CONFIG_NOTE,
    approach_restriction_note,
    kernel_coverage_analyzer_note,
    kernel_coverage_ncu_targeting,
)

_ALL_PROMPTS = {
    "benchmarker": BENCHMARKER_SYSTEM_PROMPT,
    "projector": PROJECTOR_SYSTEM_PROMPT,
    "profiler": PROFILER_SYSTEM_PROMPT,
    "analyzer": ANALYZER_SYSTEM_PROMPT,
    "optimizer": OPTIMIZER_SYSTEM_PROMPT,
    "evaluator": EVALUATOR_SYSTEM_PROMPT,
    "integrator": INTEGRATOR_SYSTEM_PROMPT,
    "qa": QA_SYSTEM_PROMPT,
    "reporter": REPORTER_SYSTEM_PROMPT,
}

# Roles that run the canonical benchmark_serving.py command themselves.
_MEASURING = ("benchmarker", "profiler", "evaluator", "integrator", "qa")


def _norm(text: str) -> str:
    """Collapse whitespace so substring assertions survive line-wrapping."""
    return re.sub(r"\s+", " ", text)


# Canonical ``benchmark_serving.py`` flags every measuring role must carry.
_BENCHMARK_CANONICAL_FLAGS = (
    "--tokenizer",
    "--trust-remote-code",
    "--random-ids",
    "--tokenize-on-client",
    "--ignore-eos",
    "--no-test-input",
    "--percentile-metrics",
)

# Canonical ``nsys profile`` flags the profiler must carry.
_NSYS_CANONICAL_FLAGS = (
    "-t 'cuda,nvtx,python-gil'",
    "-c cudaProfilerApi",
    "--cuda-graph-trace node",
    "TLLM_NVTX_DEBUG=1",
    "--trace-fork-before-exec=true",
)


def test_measuring_roles_carry_canonical_benchmark_flags():
    for role in _MEASURING:
        for flag in _BENCHMARK_CANONICAL_FLAGS:
            assert flag in _ALL_PROMPTS[role], (role, flag)
        assert "do not improvise" in _ALL_PROMPTS[role], role


def test_profiler_carries_canonical_nsys_flags():
    for flag in _NSYS_CANONICAL_FLAGS:
        assert flag in PROFILER_SYSTEM_PROMPT, flag
    # The safety flag that keeps nsys from SIGTERMing the engine.
    assert "--capture-range-end=stop" in PROFILER_SYSTEM_PROMPT
    # Knob verification (verify before asserting) came along with the
    # profiling reference blocks.
    assert "TLLM_PROFILE_START_STOP" in PROFILER_SYSTEM_PROMPT
    assert "Verify the profiling knobs first" in PROFILER_SYSTEM_PROMPT


def test_additional_profiling_passes_belong_to_the_profiler():
    # Approval-time evidence has a bounded purpose: compare this candidate's
    # timing to the reference. Round-level capture owns utilization/stacks.
    for flag in (
        "--gpu-metrics-devices=all",
        "--gpu-metrics-frequency=100000",
        "--python-backtrace",
        "--python-sampling=true",
        "--cudabacktrace=kernel:5000,sync:10000",
    ):
        assert flag in PROFILER_SYSTEM_PROMPT, flag
        assert flag not in EVALUATOR_SYSTEM_PROMPT, flag
    assert "## Run A2" in PROFILER_SYSTEM_PROMPT
    assert "## Run A2" not in EVALUATOR_SYSTEM_PROMPT
    assert "## Run B" not in EVALUATOR_SYSTEM_PROMPT
    assert "ncu --import" not in EVALUATOR_SYSTEM_PROMPT
    assert "Author `<workspace>/nsys_analysis/items.json`" not in EVALUATOR_SYSTEM_PROMPT
    assert "refine taxonomy" in EVALUATOR_SYSTEM_PROMPT  # explicitly prohibited


def test_run_a2_is_a_separate_capture_and_degrades_gracefully():
    # Metric sampling and backtraces perturb the timeline, so they take
    # their own captures; and when the host withholds the profiling
    # permission the run reports it rather than inventing utilization.
    prompt = _norm(PROFILER_SYSTEM_PROMPT)
    assert "server_nsys_metrics" in prompt
    assert "server_nsys_stacks" in prompt
    assert "additional** captures" in prompt
    assert "ERR_NVGPUCTRPERM" in prompt
    assert "additive, never blocking" in prompt


def test_run_a2_call_stack_pass_states_the_cuda_graph_limit():
    # perf-optimize profiles graph-captured servers almost exclusively, so
    # the prompt has to say that a backtrace on cudaGraphLaunch names the
    # launch site and not the operator inside the graph.
    prompt = _norm(PROFILER_SYSTEM_PROMPT)
    assert "cudaGraphLaunch" in prompt
    assert "graphId IS NOT NULL" in prompt


def test_findings_contract_carries_the_run_a2_evidence():
    contract = _norm(PROFILE_FINDINGS_CONTRACT)
    assert "gpu metrics unavailable" in contract
    assert "call stacks unavailable" in contract
    assert "bounding resource" in contract


def test_analyzer_carries_the_nsys_timeline_decomposition():
    # Run A does not stop at ``nsys stats``: the analyzer exports a
    # ``.sqlite`` and runs the internal-perf-nsight-system-analysis pipeline, whose
    # per-iteration budget is what separates a host-exposure item from a
    # slow-kernel item.
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "internal-perf-nsight-system-analysis" in prompt
    assert "trtllm-agent-toolkit:internal-perf-nsight-system-analysis" in prompt
    assert "nsys export --type sqlite" in prompt
    # Proactive by construction — it re-reads a trace already captured.
    assert "costs no extra server launch" in prompt
    # The roadmap's expected-gain grounding reads the split, not just shares.
    assert "compute-absent split" in prompt
    assert "faster kernels do not recover launch-starved host time" in prompt
    # Degrades to a one-liner rather than blocking or fabricating.
    assert "timeline analysis unavailable" in prompt


def test_evaluator_decomposes_the_accept_evidence_capture():
    # The accept-evidence trace gets the same treatment, so "the launch
    # gaps shrunk" is measured on both sides rather than eyeballed.
    prompt = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "internal-perf-nsight-system-analysis" in prompt
    assert "trtllm-agent-toolkit:internal-perf-nsight-system-analysis" in prompt
    assert "nsys_analysis" in prompt
    assert "the launch-starved share shrunk" in prompt
    # Never blocks the verdict, never states an unmeasured split.
    assert "never block the verdict on it" in prompt


def test_evaluator_diffs_with_the_skills_comparative_mode():
    # The skill differences two variants natively; running it twice
    # single-variant and eyeballing the trees throws that away.
    prompt = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "--variant before" in prompt
    assert "--variant after" in prompt
    assert "difference/rank-0/iteration.json" in prompt
    # Per-call deltas survive a launch-count mismatch between the sides.
    assert "difference/rank-0/module_slice.json" in prompt
    # One taxonomy across both sides — a diff across two is not a diff.
    assert "a diff across two taxonomies is not a diff" in prompt
    # Degrades where the earlier capture kept no sqlite.
    assert "kept no `.sqlite`" in prompt
    # The mechanism claim names a row, not a vibe.
    assert "is not a mechanism" in prompt


def test_run_a2a_capture_feeds_the_timeline_pipeline():
    # The two changes compose rather than sit side by side: A2a's
    # metric-sampling capture is exactly what the skill's per-operator
    # utilization step reads, and folding it in costs no GPU.
    prompt = _norm(PROFILER_SYSTEM_PROMPT)
    assert "--metrics-profile 0=<workspace>/server_nsys_metrics.sqlite" in prompt
    assert "it reads the sampling capture, never the timing one" in prompt
    # A2b is not an input to it — those flags name another tool's exports.
    assert "different tool’s exports, not A2b" in prompt
    # And step 5 never waits on a pass that is allowed to be skipped.
    assert "do not wait for A2" in prompt


def test_profiler_carries_the_ncu_capture():
    # The shared Run B: a bounded per-kernel ncu capture of the top nsys
    # kernels, interpreted with the perf-nsight-compute-analysis skill.
    for flag in (
        "--target-processes all",
        "--profile-from-start off",
        "--section SpeedOfLight",
        "--launch-count",
    ):
        assert flag in PROFILER_SYSTEM_PROMPT, flag
    prompt = _norm(PROFILER_SYSTEM_PROMPT)
    assert "perf-nsight-compute-analysis" in prompt
    assert "trtllm-agent-toolkit:perf-nsight-compute-analysis" in prompt
    # The findings carry the dedicated section, degrading honestly.
    assert "ncu --import" in prompt
    assert "Final kernel interpretation" in prompt


def test_analyzer_grounds_roadmap_items_across_the_analyses() -> None:
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    for evidence in ("nsys timeline", "ncu kernel analysis", "SOL correlation"):
        assert evidence in prompt
    assert "including disagreement or missing analyses" in prompt
    assert "measured bound class" in prompt
    assert "Dormant capabilities use the explicit source/config evidence exception" in prompt


def test_no_prompt_references_removed_builtin_tools():
    # The agents run on the CLI's ``default`` toolset, which no longer
    # includes ``Grep``/``Glob``; instructing them makes the agent call a
    # nonexistent tool. Every role prompt must avoid the tool names.
    for name, prompt in _ALL_PROMPTS.items():
        for tool in ("Grep", "Glob"):
            assert not re.search(rf"\b{tool}\b", prompt), (name, tool)


# ------------------------------------------------------------------- casebook


def test_benchmarker_and_analyzer_load_casebook_read_only():
    for role in ("benchmarker", "analyzer"):
        prompt = _ALL_PROMPTS[role]
        assert "perf-optimization-casebook" in prompt, role
        assert "`Skill` tool" in prompt, role
        assert "Ground your analysis in the optimization casebook" in prompt, role


def test_optimizer_gets_the_actionable_casebook_variant():
    assert "Apply from the optimization casebook" in OPTIMIZER_SYSTEM_PROMPT
    assert "perf-optimization-casebook" in OPTIMIZER_SYSTEM_PROMPT
    # The read-only stance would contradict the optimizer's job.
    assert "Ground your analysis in the optimization casebook" not in OPTIMIZER_SYSTEM_PROMPT
    block = _norm(CASEBOOK_APPLY)
    assert "how-to-apply" in block
    assert "rollback" in block
    # Still no hard dependency on the toolkit.
    assert "not available in this environment" in block


# -------------------------------------------------------------- git discipline


def test_mutating_roles_carry_git_discipline():
    for role in ("optimizer", "evaluator"):
        prompt = _norm(_ALL_PROMPTS[role])
        assert "dedicated optimization branch" in prompt, role
        assert "import tensorrt_llm" in prompt, role
        assert "`git push`" in prompt, role
    block = _norm(GIT_DISCIPLINE)
    # The orchestrator owns commits and reverts; agents never mutate git.
    assert "Never run `git commit`" in block
    assert "orchestrator owns all git state" in block
    assert "only the current roadmap item's changes" in block
    assert "active runtime checkout from the turn instructions" in block
    assert "prepend that exact checkout to `PYTHONPATH`" in block
    assert "may differ from the active runtime checkout" in block
    assert "an editable install" not in block


def test_code_edits_never_reference_run_internals():
    # A committed comment like "See the opt-008 bench in the perf
    # workspace" is meaningless to TRT-LLM readers: the discipline block
    # bans run-internal references in source, and the evaluator's
    # code-quality axis gates on it.
    block = _norm(GIT_DISCIPLINE)
    assert "stand on its own" in block
    assert "roadmap item ids (`opt-008`)" in block
    assert "the provenance story belongs in `optimization_summary.md`" in block
    gate = _norm(EXPECTATION_GATE)
    assert "no comments or names that reference this run's internals" in gate


# ---------------------------------------------------------------- kernel reuse


def test_kernel_work_roles_prefer_existing_kernels():
    # A correct fusion realized as a hand-written kernel that flashinfer
    # (or TRT-LLM itself) already ships is still the wrong change — the
    # whole plan → apply → gate chain must carry the reuse rule.
    for role in ("analyzer", "optimizer", "evaluator"):
        prompt = _ALL_PROMPTS[role]
        assert "Prefer existing kernels over writing new ones" in prompt, role
    block = _norm(KERNEL_REUSE)
    # Search order: the checkout first, then flashinfer, then any other
    # provider already integrated; a new kernel is the last resort.
    assert "TRT-LLM checkout's custom ops" in block
    assert "flashinfer" in block
    assert "Other integrated providers" in block
    # Planning and implementing both record the search that came up empty.
    assert "what you searched" in block
    # The preference is conditional: an empty search makes a new kernel the
    # encouraged realization, never a dropped item — the analyzer still
    # plans it, the optimizer falls back to writing instead of recording a
    # no-change blocker, and the evaluator judges it on the normal axes.
    assert "If none fits, plan a scoped **new kernel**" in block
    assert "fall back to writing the kernel rather than recording a no-change blocker" in block
    assert "the new kernel is a legitimate realization" in block
    # The evaluator enforces reuse on the code-quality axis, gain or not.
    assert "never passes the code-quality axis, whatever gain it measures" in block
    assert "PUSH_BACK with `reason_category: code_quality`" in block
    gate = _norm(EXPECTATION_GATE)
    assert "adds no hand-written kernel" in gate
    # The measuring-only and synthesis roles never touch kernels.
    assert "Prefer existing kernels" not in BENCHMARKER_SYSTEM_PROMPT
    assert "Prefer existing kernels" not in REPORTER_SYSTEM_PROMPT


# ---------------------------------------------------------------- roadmap spec


def test_roadmap_touching_roles_carry_the_contract():
    for role in ("analyzer", "optimizer", "evaluator", "qa", "reporter"):
        prompt = _ALL_PROMPTS[role]
        assert "The roadmap contract (`roadmap.yaml`)" in prompt, role
        assert "List order is priority order" in prompt, role
    # The benchmarker runs before the roadmap exists.
    assert "The roadmap contract" not in BENCHMARKER_SYSTEM_PROMPT


def test_roadmap_contract_pins_ownership():
    for role in ("analyzer", "optimizer"):
        prompt = _norm(_ALL_PROMPTS[role])
        assert "The **orchestrator** owns every lifecycle field" in prompt, role


def test_roadmap_readers_do_not_receive_analyzer_authoring_duties():
    assert "Initialization and ids" in ANALYZER_SYSTEM_PROMPT
    for role in ("optimizer", "evaluator", "integrator", "qa", "reporter"):
        prompt = _ALL_PROMPTS[role]
        assert "Initialization and ids" not in prompt, role
        assert "Do not initialize, reorder" in prompt, role
        assert "candidate-ready" in prompt, role


# ------------------------------------------------------------ acceptance gate


def test_evaluator_carries_the_expectation_gate():
    gate = _norm(EXPECTATION_GATE)
    assert "accept_fraction × expected_gain_pct" in gate
    assert "noise_floor_pct" in gate
    # Gains accumulate: the reference is the last accepted measurement.
    assert "last ACCEPTED measurement" in gate
    assert "never the original baseline" in gate
    prompt = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "accept_fraction" in prompt
    for category in ("code_quality", "functionality", "perf_shortfall"):
        assert category in prompt, category


def test_expectation_gate_is_three_way():
    gate = _norm(EXPECTATION_GATE)
    # PUSH_BACK = winnable with a concrete fix; REJECT = broken premise,
    # terminal (saving the retries' benchmarks); final attempt coerces.
    assert "PUSH_BACK" in gate
    assert "REJECT" in gate
    assert "premise is broken" in gate
    assert "no retry would help" in gate
    assert "the orchestrator treats it as REJECT" in gate
    prompt = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "`APPROVE` | `REJECT` | `PUSH_BACK`" in prompt


def test_evaluator_carries_the_accept_evidence_procedure():
    prompt = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "Accept-evidence capture (APPROVE only)" in prompt
    # The capture is diagnostic and never contaminates the measurement.
    assert "diagnostic, never a measurement" in prompt
    assert "fresh relaunch" in prompt
    assert "never a reason to flip the verdict" in prompt
    # Mechanism verification is the point of the capture.
    assert "claimed mechanism is visible" in prompt
    # The canonical nsys wrap ships with the prompt so the evaluator
    # never improvises profiler flags.
    for flag in _NSYS_CANONICAL_FLAGS:
        assert flag in EVALUATOR_SYSTEM_PROMPT, flag
    # And the report widens beyond the target metric.
    assert "full-metric diff" in prompt
    assert "Kernel evidence" in prompt


def test_expectation_gate_carries_the_pareto_rule():
    gate = _norm(EXPECTATION_GATE)
    # Curve mode: mean over per-point gains vs the same-concurrency
    # current_best.curve entry, plus the no-regress condition.
    assert "Pareto gate" in gate
    assert "mean_gain_pct = arithmetic mean of gain_i" in gate
    assert "every gain_i >= -regression_bar" in gate
    # The bar defaults to the noise floor when no budget is declared.
    assert "else noise_floor_pct" in gate
    assert "current_best.curve" in gate
    # Missing evidence blocks acceptance instead of bypassing regression checks.
    assert "carries no `curve`" in gate
    assert "the performance gate cannot pass" in gate
    assert "never skip the per-point no-regress check" in gate


def test_expectation_gate_carries_focus_scoring():
    gate = _norm(EXPECTATION_GATE)
    # The scored subset narrows the mean, never the no-regress veto.
    assert "optimize.focus_concurrencies" in gate
    assert "scored points" in gate
    assert "no-regress condition covers **every** point" in gate
    # The ledger fields follow the scored mean.
    assert "the **scored** mean" in gate
    # Roadmap-touching roles learn the ledger semantics from the contract.
    spec = _norm(ROADMAP_SPEC)
    assert "Focus scoring" in spec
    assert "mean over **only those points**" in spec
    # The measurement protocol's aggregation rule names the subset too.
    assert "optimize.focus_concurrencies" in _norm(MEASUREMENT_PROTOCOL)


def test_expectation_gate_carries_the_regression_budget():
    gate = _norm(EXPECTATION_GATE)
    # The budget is owner-declared, never assumed, and defaults strict.
    assert "optimize.max_regression_pct" in gate
    assert "regression_bar" in gate
    assert "never yours to assume" in gate
    # Used budgets must be surfaced, not buried in the mean.
    assert "name that point" in gate
    reporter = _norm(REPORTER_SYSTEM_PROMPT)
    assert "regression budget is headline material" in reporter


def test_analyzer_carries_the_dormant_capability_sweep():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # Profiling is blind to levers that never run; round 1 must sweep
    # for them in the checkpoint config, serving config, and gated code.
    assert "Dormant-capability sweep" in prompt
    assert "mtp_num_hidden_layers" in prompt
    assert "speculative_config" in prompt
    assert 'grep -n "environ"' in prompt
    assert "## Dormant capabilities" in prompt
    # Dormant levers cannot have trace evidence — dismissing them for
    # lacking it is exactly the failure the sweep exists to prevent.
    assert 'Never dismiss for "no trace evidence"' in prompt


def test_reporter_carries_the_durable_facts_section():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Durable facts for the next campaign" in prompt
    # The three tags, each demanding a citation.
    assert "`[dead]`" in prompt
    assert "`[alive]`" in prompt
    assert "`[env]`" in prompt
    assert "Durable facts are evidence, not opinion" in prompt


def test_measuring_roles_carry_the_measurement_protocol():
    protocol = _norm(MEASUREMENT_PROTOCOL)
    assert "positive = improvement" in protocol
    assert "output_throughput" in protocol
    # Curve mode: one run per point over one server launch, per-point
    # result dirs, and the worked Pareto example.
    assert "one run per `benchmark.concurrency` point" in protocol
    assert "concurrency_<c>" in protocol
    assert "Curve worked example" in protocol
    assert "mean = +3.24%" in protocol
    for role in ("benchmarker", "evaluator", "integrator", "qa"):
        assert "Measurement protocol" in _ALL_PROMPTS[role], role
        assert "one run per `benchmark.concurrency` point" in _norm(_ALL_PROMPTS[role]), role


def test_measuring_roles_carry_the_derived_metrics_reference():
    for role in ("benchmarker", "evaluator", "integrator", "qa"):
        prompt = _norm(_ALL_PROMPTS[role])
        assert "1000 / mean_tpot_ms" in prompt, role
        assert "output_throughput / num_gpus" in prompt, role
        assert "curve summary table" in prompt, role


# --------------------------------------------------------------- tuning config


def test_server_roles_carry_the_tuning_config_supersede_note():
    note = _norm(TUNING_CONFIG_NOTE)
    assert "supersedes" in note
    assert "**always** passes" in note
    assert "turn instructions name the exact **active tuning config**" in note
    assert "supersedes shorthand references" in note
    assert "<workspace>/tuning/extra_llm_api_options.yaml" not in note
    for role in ("benchmarker", "profiler", "optimizer", "evaluator", "integrator", "qa"):
        assert "The active tuning config" in _ALL_PROMPTS[role], role
    # Item and integration changes have separate writable configs; the
    # accepted snapshot is always orchestrator-managed.
    assert "isolated integration config" in note
    assert "accepted config snapshot" in note
    assert "Never edit" in note


def test_measurement_roles_cannot_repair_startup_by_changing_tuning():
    for role in ("benchmarker", "profiler", "evaluator", "qa"):
        prompt = _norm(_ALL_PROMPTS[role])
        assert "Do not change the read-only tuning config" in prompt, role
        assert "OOM: lower" not in prompt, role
        assert "Pass `--extra_llm_api_options <path>` **only when**" not in prompt, role
    for role in ("optimizer", "integrator"):
        prompt = _norm(_ALL_PROMPTS[role])
        assert (
            "Correct only defects within your assigned item or candidate-combination scope"
            in prompt
        )
        assert "Do not change the read-only tuning config" not in prompt, role
        assert "never change unrelated performance knobs" in prompt, role


def test_lifecycle_builder_preserves_default_and_exposes_scoped_edit_policy():
    default = _norm(build_server_lifecycle())
    measurement = _norm(build_server_lifecycle(active_tuning_config=True))
    mutation = _norm(build_server_lifecycle(active_tuning_config=True, allow_config_changes=True))
    assert "top-level `extra_llm_api_options` key; omit the flag otherwise" in default
    assert "OOM: lower" in default
    for lifecycle in (measurement, mutation):
        assert "--extra_llm_api_options <active tuning config>" in lifecycle
        assert "owns_port" in lifecycle
        assert "OOM: lower" not in lifecycle
    assert "read-only tuning config" in measurement
    assert "candidate-combination scope" in mutation


def test_all_launchers_verify_checkout_in_the_actual_runtime_environment():
    for role in ("benchmarker", "profiler", "optimizer", "evaluator", "integrator", "qa"):
        prompt = _ALL_PROMPTS[role]
        assert prompt.count(RUNTIME_CHECKOUT) == 1, role
        assert "tensorrt_llm.__file__" in prompt, role
        assert "do not benchmark" in prompt, role
        assert "allocated container using its staged checkout path" in _norm(prompt), role


def test_integrator_has_a_complete_measured_acceptance_contract():
    prompt = _norm(INTEGRATOR_SYSTEM_PROMPT)
    for contract in (
        "(new − reference) / reference × 100",
        "(reference − new) / reference × 100",
        "finite positive numbers",
        "missing, duplicate or extra points",
        "optimize.focus_concurrencies",
        "including unscored points",
        "gain_i >= -regression_bar",
        "optimize.max_regression_pct",
        "output_throughput / num_gpus",
        "full-metric diff",
        "isolated integration config",
        "candidate config changes relative to the common base",
        "never edit the accepted config snapshot",
    ):
        assert contract in prompt, contract


def test_parallel_candidate_approval_is_distinct_from_campaign_acceptance():
    evaluator = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "parallel mode waits for integration before accepting the item" in evaluator
    assert "it does not describe the later combined accepted state" in evaluator
    reporter = _norm(REPORTER_SYSTEM_PROMPT)
    assert "only the accepted integrator APPROVE or FALLBACK_BEST" in reporter
    assert "must never become successive trajectory steps" in reporter
    assert "`100 → 116`" in reporter
    assert "never compute gain from the ratio of two curve means" in reporter
    html = _norm(OPTIMIZE_HTML_COMPANION)
    assert "accepted parallel integrations" in html
    assert "scored concurrency points" in html


def test_parallel_slurm_and_disagg_prompts_require_distinct_node_allocations():
    slurm = build_perf_optimize_prompts(include_slurm_environment=True)
    disagg = build_perf_optimize_prompts(include_disagg=True)
    for role in ("optimizer", "evaluator", "integrator"):
        prompt = _norm(getattr(slurm, role))
        assert "exclusive Slurm node allocation" in prompt
        assert "--exclusive" in prompt
        assert "never attach to or reuse a sibling item's allocation" in prompt
        prompt = _norm(getattr(disagg, role))
        assert "`--exclusive`" in prompt
        assert "run-local copy's `slurm.extra_args`" in prompt
        assert "keep the original harness config read-only" in prompt
        assert "integrator combines candidates only in its isolated integration config" in prompt


# ------------------------------------------------------------------------- qa


def test_qa_prompt_is_a_decisionless_final_verification():
    prompt = _norm(QA_SYSTEM_PROMPT)
    # QA runs once and verifies; the orchestrator owns the loop.
    assert "final verification" in prompt
    assert "You do not decide whether the campaign continues" in prompt
    assert "CONTINUE" not in prompt
    assert "Final-profile" not in prompt
    # Accuracy runs only when task.yaml configures it.
    assert "only if `task.yaml` has an `accuracy` block" in prompt
    assert "accuracy: not configured" in prompt
    # Fresh-eyes isolation.
    assert "Do **not** read the evaluator's" in prompt


# -------------------------------------------------------------------- reporter


def test_reporter_reports_expected_vs_measured_and_future_work():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Expected vs measured" in prompt
    assert "Failed Attempts" in prompt
    assert "Remaining Roadmap" in prompt
    # The headline is the final verification's independent number, not
    # the evaluator chain's.
    assert "The headline number is the final verification's" in prompt
    assert "Final Verification" in prompt
    assert "verification_report.md" in prompt
    assert "optimization_report.html" in REPORTER_SYSTEM_PROMPT


def test_reporter_never_launches_servers():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Do **not** launch servers" in prompt


def test_reporter_carries_the_trajectory_section():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Optimization Trajectory" in prompt
    # The path is reconstructed from the structured trail in the order
    # applied — the roadmap's listing order is priority, not chronology —
    # and gaps stay gaps.
    assert "measured_value" in prompt
    assert "never interpolated" in prompt
    # The markdown table is the data the HTML line chart plots.
    assert "line chart" in prompt


def test_reporter_carries_the_kernel_comparison():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Kernel-Level Comparison" in prompt
    # Grounded in the analyzer rounds' nsys artifacts. The kernel table
    # still comes from `kern_sum` — it is the only per-kernel artifact
    # carrying call counts — but it is a sum across overlapping streams
    # over the whole capture, so it carries the relative before/after
    # shift and the iteration budget carries the magnitude.
    assert "cuda_gpu_kern_sum" in prompt
    assert "nsys_stats.txt" in prompt
    assert "the only artifact carrying per-kernel call" in prompt
    assert "never as a per-iteration magnitude" in prompt
    assert "the budget is the one describing" in prompt
    # ...with honest provenance: what each profile covers, and no
    # fabricated "after" data when only round 1 was profiled.
    assert "which accepted items were in effect" in prompt
    assert "closing profiler round may supply that evidence" in prompt
    assert (
        "capture directory your driving instructions name as matching the final accepted state"
        in prompt
    )
    assert "no post-optimization profile exists" in prompt


def test_reporter_lists_both_sides_of_the_iteration_budget():
    # The comparison opens on the iteration budget "where both sides
    # carry a nsys_analysis/" — so both sides have to be reachable. The
    # round-level products are the *before* side; the *after* side lives
    # under the accepted attempt's profile/, and listing only the former
    # leaves the reporter looking for a directory it was never given.
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "rounds/round_<n>/analysis/nsys_analysis/" in prompt
    assert "`profile/nsys_analysis/` beside it" in prompt
    assert "parallel candidate capture does not describe the later integrated" in prompt
    # And the budget leads the section, ahead of the kernel table.
    assert "open the section with the iteration budget before the kernel table" in prompt
    # Degrades rather than fabricating a one-sided budget.
    assert "where either side lacks it, say so and compare on kernels alone" in prompt


def test_html_companion_charts_are_self_contained():
    block = _norm(OPTIMIZE_HTML_COMPANION)
    assert "Trajectory line chart" in block
    assert "Kernel before/after bars" in block
    # No chart library: data embedded inline, rendered to inline SVG.
    assert "no chart library" in block
    assert "inline SVG" in block
    # Charts never diverge from the tables they sit above.
    assert "the table is the source of truth" in block
    # New sections slot into the HTML body in markdown order.
    assert "Baseline, Optimization Trajectory, Pareto Improvement — curve mode only" in block
    assert "Kernel-Level Comparison, Failed Attempts" in block


def test_html_companion_carries_the_pareto_chart():
    block = _norm(OPTIMIZE_HTML_COMPANION)
    assert "Pareto improvement chart" in block
    assert "x = tok/s/user, y = tok/s/gpu" in block
    # Exactly two series — baseline vs final — with labeled points, and
    # the chart disappears rather than plotting invented curves.
    assert "baseline curve vs the final curve" in block
    assert "`c=<n>`" in block
    assert "Omit the chart and the section in scalar mode" in block


def test_reporter_carries_the_pareto_improvement_section():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Pareto Improvement" in prompt
    assert "omit this section entirely in scalar mode" in prompt
    # Provenance: baseline from the roadmap curve, final from QA's curve.
    assert "`baseline.curve`" in prompt
    # Curve-mode headline aggregates as the mean across points.
    assert "mean across concurrency points" in prompt
    # Rigor: per-point values trace to recorded curves.
    assert "Pareto values trace to recorded curves" in prompt


# -------------------------------------------------------------- SOL projector
# The projection methodology is the internal-perf-sol-analysis skill
# (peaks from its calculator, latency constants measured when a GPU is
# reachable, the α-β-u ceiling arithmetic shown in the report); the
# model architecture comes from the checkpoint's config.json. It runs
# once per campaign, against the perf-optimize baseline artifacts, and
# its guidance addresses the Analyzer and Reporter.


def test_projector_prompt_carries_no_dlsim_traces():
    prompt = PROJECTOR_SYSTEM_PROMPT
    # dlsim is gone entirely — no checkout cross-check, no paths, no
    # MCP tools, no execution-path names.
    assert "dlsim" not in prompt.lower()
    assert "python/lwdlm" not in prompt
    # The structural quantities come from the checkpoint's config.json.
    assert "config.json" in prompt


def test_projector_prompt_builds_on_sol_skill():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    # The methodology is the SOL skill, loaded via the Skill tool — with
    # the fully-qualified name so a plugin-namespaced install resolves,
    # and graceful degradation when the skill is not installed.
    assert "internal-perf-sol-analysis" in prompt
    assert "trtllm-agent-toolkit:internal-perf-sol-analysis" in prompt
    assert "`Skill` tool" in prompt
    assert "not available in this environment" in prompt


# --------------------------------------------------------------------------- #
# The fallback block is single-sourced in perf-analyze's ``_common`` and
# re-exported here, so the two workflows cannot drift.
# --------------------------------------------------------------------------- #


def test_projection_setup_template_states_no_methodology_as_fact():
    """The template is copied verbatim into `sol_projection.md`.

    A hardcoded `Method:` / `Peaks file:` line makes the projector assert
    the full methodology even when it ran the fallback — a false
    provenance claim in the one artifact whose job is to disclose it, and
    one the Analyzer then follows to a peaks file nobody wrote.
    """
    for label, prompt in (
        ("full", PROJECTOR_SYSTEM_PROMPT),
        ("reduced", build_projector_prompt("reduced")),
    ):
        assert "- Method: <" in prompt, label
        assert "- Peaks file: <" in prompt, label
        assert "- Method: internal-perf-sol-analysis" not in prompt, label
        assert "- Peaks file: sol_work/peaks.json" not in prompt, label
        # The environment without a calculator has something to write.
        assert "not written: no peaks" in prompt, label


def test_full_methodology_leaves_the_projector_prompt_untouched():
    assert SOL_METHODOLOGY_FALLBACK is _ANALYZE_METHODOLOGY_FALLBACK
    assert build_projector_prompt() == PROJECTOR_SYSTEM_PROMPT
    assert build_projector_prompt("nonsense") == PROJECTOR_SYSTEM_PROMPT


def test_reduced_methodology_appends_the_fallback_block_and_nothing_else():
    bundle = build_perf_optimize_prompts(include_sol=True, sol_methodology="reduced")
    assert bundle.projector == PROJECTOR_SYSTEM_PROMPT + SOL_METHODOLOGY_FALLBACK
    full = build_perf_optimize_prompts(include_sol=True)
    assert bundle.analyzer == full.analyzer
    assert bundle.optimizer == full.optimizer
    assert bundle.reporter == full.reporter


def test_projector_prompt_resolves_peaks_and_latencies_via_skill():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    # Peaks come from the skill's calculator. The "resolve, never
    # recall" rule is the skill's own — the loaded skill states it, so
    # the prompt carries only what the skill cannot know: which part
    # name to resolve.
    assert "sol_calc.py peaks --part" in prompt
    assert "part-name hint" in prompt
    # Latency constants: measured here when a GPU is reachable,
    # recorded as unmeasured (never guessed) when one is not — this
    # stage may run on a login node, which the skill cannot know.
    assert "measure_channels.py" in prompt
    assert "do **not** guess" in prompt
    assert "unmeasured" in prompt


def test_projector_prompt_never_fabricates_measured_inputs():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    # ``sol_calc.py analyze`` correlates measured per-op times; no
    # profiling stage has run yet, so there are none — and script inputs
    # are never invented to force a run.
    assert "never fabricate an input" in prompt
    assert "measured_ms" in prompt


def test_projector_prompt_speaks_skill_vocabulary():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    for term in ("% of SOL", "MFU", "MBU", "gap-to-SOL", "α-β-u"):
        assert term in prompt, term
    assert "compute / memory / launch" in prompt
    # The ceiling models kernel execution + per-launch latency only — a
    # gap beyond it points at host/scheduling costs it does not price.
    assert "kernel execution plus per-launch latency only" in prompt
    assert "request queueing" in prompt


def test_projector_prompt_names_internal_knowledge_and_keeps_it_consultative():
    prompt = PROJECTOR_SYSTEM_PROMPT
    assert "internal-glean-search" in prompt
    assert "internal-glean-specialist" in prompt
    # No site-specific URL is baked into the prompt.
    assert "http://" not in prompt
    assert "https://" not in prompt
    normed = _norm(prompt)
    assert "if that skill/subagent exists" in normed
    assert "consultative" in normed
    assert "reproducible from the arithmetic" in normed


def test_projector_prompt_degrades_honestly():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    assert "Projection unavailable" in prompt
    assert "never fabricate" in prompt


def test_projector_prompt_template_sections():
    for header in (
        "## Projection setup",
        "## Projected SOL ceiling",
        "## Measured vs SOL",
        "## Headroom & bound mix",
        "## Guidance for optimization",
        "## Caveats",
    ):
        assert header in PROJECTOR_SYSTEM_PROMPT, header


def test_projector_prompt_targets_the_optimize_pipeline():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    # Once per campaign, against perf-optimize's artifact layout: the
    # baseline lives under baseline/ and the parallel mapping comes from
    # the live tuning config, not the task-level extra_llm_api_options.
    assert "once per campaign" in prompt
    assert "baseline/benchmark_results.md" in prompt
    assert "tuning/extra_llm_api_options.yaml" in prompt
    # Guidance addresses this workflow's consumers — the Analyzer owns
    # the roadmap, while capture details belong to the profiler.
    assert "Analyzer" in prompt
    assert "the Analyzer owns `roadmap.yaml`" in prompt
    assert "expected_gain_pct" in prompt
    assert "Profiler" not in prompt
    # Later stages' files are off-limits.
    assert "do not touch them" in prompt
    # Curve mode: the ceiling is derived per configured point.
    assert "once per concurrency point" in prompt
    assert "point by point" in prompt


def test_sol_analyzer_context_is_context_not_evidence() -> None:
    block = _norm(SOL_ANALYZER_CONTEXT)
    assert "its measured column is the baseline snapshot" in block
    assert "bound `expected_gain_pct` by recoverable headroom" in block
    assert "Fresh measured evidence outranks the projection" in block
    assert "Projected numbers must remain labeled as projections" in block
    assert "Treat the projection as the initial theoretical model, not a frozen answer" in block
    assert "`kernel_ledger.yaml`'s `model_revisions`" in block
    assert "never justifies weakening a bound" in block
    assert "projection is missing or unavailable, skip correlation" in block


def test_sol_analyzer_context_forbids_silent_exhaustion() -> None:
    block = _norm(SOL_ANALYZER_CONTEXT)
    assert "no actionable item remains despite meaningful projected headroom" in block
    assert "## Remaining-gap attribution" in block
    assert (
        "each kernel or logical region, its roadmap item or evidence-backed campaign constraint"
        in block
    )
    assert "any `unexplained` remainder" in block
    assert "failed items' `evaluation.md`" in block
    assert "Gap implication" in block
    assert "preserve the ledger's unexplained discrepancies" in block
    assert "campaign-end accounting brief" in block
    assert "link to the existing gap explanations" in block


def test_sol_analyzer_context_correlates_per_round_with_the_skill_calculator():
    block = _norm(SOL_ANALYZER_CONTEXT)
    # The correlation is the skill's calculator over structural facts,
    # joined against the projector's persisted peaks file.
    assert "sol_calc.py analyze" in block
    assert "regions.json" in block
    assert "sol_work/peaks.json" in block
    assert "never invent params or `measured_ms` rows" in block
    # The assembled findings contract selects one destination for the
    # comparison; the calculator recipe retains honest degradation.
    assert "## SOL correlation (measured vs ceiling)" not in block
    assert "report's comparison section" in block
    assert "Correlation unavailable" in block
    # Optimize-specific placement and cadence: per-round artifacts, one
    # campaign-level peaks file, a fresh join every profiling round.
    assert "this round's `analysis/` directory" in block
    assert "Re-run correlation in full analyses, including re-analysis" in block
    # Replan preserves measurements while facts may revise the prediction.
    assert "Replan-only rounds preserve standing measurements" in block
    assert "may revise analytical predictions from new facts" in block


def test_projector_prompt_persists_peaks_for_the_analyzer():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    assert "Persist the machine-readable peaks file" in prompt
    assert "sol_work/peaks.json" in prompt
    # And the required-output template records the path — as the
    # placeholder it is, so a run without a calculator does not assert a
    # file it never wrote (see the template guard above).
    assert "Peaks file: <sol_work/peaks.json" in prompt


def test_analyzer_composes_the_shared_findings_contract():
    # perf-optimize's analyzer is perf-analyze's analyzer plus the
    # roadmap machinery: the findings report follows the same shared
    # contract (including the reserved SOL correlation section).
    assert PROFILE_FINDINGS_CONTRACT in ANALYZER_SYSTEM_PROMPT
    assert "## SOL correlation (measured vs ceiling)" in _norm(PROFILE_FINDINGS_CONTRACT)


def test_sol_optimizer_context_aims_at_the_binding_ceiling_without_scope_creep():
    block = _norm(SOL_OPTIMIZER_CONTEXT)
    # Context, not spec: the item outranks the projection, and the
    # projection never grows the change.
    assert "context, not spec" in block
    assert "Aim the implementation at the binding ceiling" in block
    assert "The projection never expands the item" in block
    assert "not yours to chase" in block
    assert "outrank the projection" in block
    # The claimed mechanism becomes checkable downstream.
    assert "SOL alignment:" in block
    assert "Mapping to the roadmap item" in block
    # Honest degradation when the projection is absent.
    assert "missing or declares itself unavailable" in block


def test_sol_reporter_guidance_carries_remaining_gap_accountability():
    block = _norm(SOL_OPTIMIZE_REPORTER_GUIDANCE)
    assert "Remaining-gap accountability" in block
    # The four exhaustive verdicts.
    for verdict in ("`closed`", "`infeasible: <constraint>`", "`untried`", "`unexplained`"):
        assert verdict in block, verdict
    # Verdicts trace to artifacts; fabricated justifications are worse
    # than an honest unexplained bucket. The analyzer's per-op
    # correlation table is a named evidence source for the gap parts.
    assert "Every accountability verdict traces to an artifact" in block
    assert "Gap implication" in block
    assert "comparison in *Per-layer theoretical performance model*" in block
    assert "*SOL correlation* when kernel coverage is disabled" in block
    assert "worse than reporting it unexplained" in block
    # A zero-accept campaign still owes the breakdown.
    assert "accepted nothing must still fill the accountability" in block


def test_evaluator_negative_verdicts_carry_gap_implication():
    # PUSH_BACK/REJECT evidence feeds the analyzer's re-planning and the
    # report's remaining-gap attribution — without any SOL exposure.
    prompt = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "Gap implication:" in prompt
    for tag in (
        "mechanism-already-present",
        "mechanism-inapplicable",
        "applied-but-no-gain",
        "blocked-by-constraint",
    ):
        assert tag in prompt, tag
    # Judged from the evaluator's own evidence, and recorded in the
    # progress entry too.
    assert "judged from your own evidence" in prompt
    assert "include the `Gap implication` line" in prompt


def test_sol_reporter_guidance_carries_the_headroom_story():
    block = _norm(SOL_OPTIMIZE_REPORTER_GUIDANCE)
    assert "## Projection vs Measured" in block
    # Placement inside the optimize report.
    assert 'between "Final Verification" and "Config & Code Diff Summary"' in block
    # The optimize-flavored table: baseline vs final % of SOL.
    assert "Baseline % of SOL" in block
    assert "Final % of SOL" in block
    assert "final % of SOL − baseline % of SOL" in block
    # The final side falls back to the ledger, and a no-accept campaign
    # captured no headroom.
    assert "`current_best`" in block
    assert "captured none of the projected headroom" in block
    # Honesty rules.
    assert "Projection unavailable" in block
    assert "never fabricate" in block


def test_sol_bundle_extends_profiler_analyzer_optimizer_and_reporter():
    base = build_perf_optimize_prompts(include_sol=False)
    sol = build_perf_optimize_prompts(include_sol=True)
    assert "SOL projection as context" in sol.analyzer
    assert "SOL projection as context" in sol.optimizer
    assert "Projection vs Measured" in sol.reporter
    assert "SOL projection as context" not in base.analyzer
    assert "SOL projection as context" not in base.optimizer
    assert "Projection vs Measured (this task has a `sol` block)" not in base.reporter
    # Everything else — including the projector's own prompt, which is
    # always in the bundle (the stage gate lives in the workflow) — is
    # unchanged.
    assert "Capture measured SOL constants when needed" in sol.profiler
    for role in ("benchmarker", "projector", "evaluator", "qa"):
        assert getattr(sol, role) == getattr(base, role), role


def test_sol_bundle_composes_with_slurm_and_restriction():
    bundle = build_perf_optimize_prompts(
        include_slurm_environment=True, approaches=["config"], include_sol=True
    )
    assert "SOL projection as context" in bundle.analyzer
    assert "SOL projection as context" in bundle.optimizer
    assert "Approach restriction (`optimize.approaches`)" in bundle.analyzer
    assert "slurm-environment" in bundle.profiler
    assert "slurm-environment" not in bundle.analyzer
    assert "Projection vs Measured" in bundle.reporter
    # The evaluator and QA judge on measurements alone — no SOL context.
    assert "SOL projection" not in bundle.evaluator
    assert "SOL projection" not in bundle.qa


def test_html_companion_overlays_the_sol_projected_curve():
    block = _norm(OPTIMIZE_HTML_COMPANION)
    assert "SOL-projected" in block
    assert "third polyline" in block
    # The overlay is honest: omitted, never approximated, without data.
    assert "omit the overlay, never approximate it" in block


# ----------------------------------------------------------------------- slurm


def test_slurm_bundle_augments_all_server_roles_but_not_reporter():
    base = build_perf_optimize_prompts(include_slurm_environment=False)
    slurm = build_perf_optimize_prompts(include_slurm_environment=True)
    for role in ("benchmarker", "profiler", "optimizer", "evaluator", "integrator", "qa"):
        assert "slurm-environment" in getattr(slurm, role), role
        assert "slurm-environment" not in getattr(base, role), role
    # The reporter never launches a server, so it is unchanged — and so
    # is the projector (no server work; under Slurm it runs on the login
    # node and records the latency constants as unmeasured).
    assert slurm.reporter == base.reporter
    assert slurm.projector == base.projector
    assert "slurm-environment" not in slurm.projector


def test_slurm_bundle_preserves_canonical_templates():
    slurm = build_perf_optimize_prompts(include_slurm_environment=True)
    for role in _MEASURING:
        for flag in _BENCHMARK_CANONICAL_FLAGS:
            assert flag in getattr(slurm, role), (role, flag)
    for flag in _NSYS_CANONICAL_FLAGS:
        assert flag in slurm.profiler, flag


def test_remote_execution_prompt_is_short_and_task_specific():
    task = {
        "checkpoint_path": "/models/gemma",
        "slurm-environment": {
            "slurm_partition": "batch",
            "docker_image": "/images/trtllm.sqsh",
            "cluster_ssh": "user@login",
            "remote_run_root": "/scratch/runs/gemma-serial",
            "account": "acct",
            "qos": "short",
        },
    }
    bundle = build_perf_optimize_prompts(
        include_slurm_environment=True,
        remote_execution=task,
        campaign_name="gemma-serial",
    )
    for role in (
        "benchmarker",
        "projector",
        "profiler",
        "optimizer",
        "evaluator",
        "integrator",
        "qa",
    ):
        prompt = getattr(bundle, role)
        compact = _norm(prompt)
        assert "SSH target: user@login" in prompt
        assert "Remote run root: /scratch/runs/gemma-serial" in prompt
        assert "Container image: /images/trtllm.sqsh" in prompt
        assert "Model checkpoint: /models/gemma" in prompt
        assert "partition=batch, account=acct, qos=short" in prompt
        assert "configured SSH target" in compact
        assert "For each remote Slurm job" in compact
        assert "isolated directory under the remote run root" in compact
        assert "excluding `.git`, builds, and caches" in compact
        assert "changed source" in compact
        assert "wait for Slurm to finish" in compact
        assert "role's required outputs and failure logs" in compact
        assert "local inspection" in compact
        assert "remove that remote job directory" in compact
        assert "Prefer one allocation for related work" in compact
        assert "retry only after a concrete correction" in compact
        assert "or when another measurement is needed" in compact
        for removed_detail in (
            "Treat command output as noisy",
            "probe, control, confirmation",
            "harness failure",
            "only Slurm submission",
        ):
            assert removed_detail not in compact
    assert "SSH target:" not in bundle.reporter


# --------------------------------------------------------- approach restriction


def test_approach_restriction_note_only_built_for_real_restrictions():
    assert approach_restriction_note(("config", "code")) == ""
    assert approach_restriction_note(()) == ""  # nothing allowed = nonsense, no note
    code_only = _norm(approach_restriction_note(("code",)))
    # The disallowed side's deterministic guard is spelled out...
    assert "tuning/extra_llm_api_options.yaml" in code_only
    assert "auto-rejects the attempt without any evaluation" in code_only
    # ... including the defaults-in-source loophole.
    assert "the same violation in disguise" in code_only
    config_only = _norm(approach_restriction_note(("config",)))
    assert "git status --porcelain" in config_only
    assert "read-only for every role" in config_only


def test_restricted_bundle_augments_planning_and_gating_roles_only():
    base = build_perf_optimize_prompts()
    restricted = build_perf_optimize_prompts(approaches=["code"])
    marker = "Approach restriction (`optimize.approaches`)"
    for role in ("analyzer", "optimizer", "evaluator"):
        assert marker in getattr(restricted, role), role
        assert marker not in getattr(base, role), role
    # The benchmarker measures, qa verifies the final state, and the
    # reporter synthesizes — none plans, applies, or judges items.
    assert restricted.benchmarker == base.benchmarker
    assert restricted.qa == base.qa
    assert restricted.reporter == base.reporter


def test_full_approaches_list_leaves_bundle_unchanged():
    assert build_perf_optimize_prompts(approaches=["config", "code"]) == DEFAULT_PROMPTS
    assert build_perf_optimize_prompts(approaches=None) == DEFAULT_PROMPTS


def test_restriction_composes_with_slurm_augmentation():
    bundle = build_perf_optimize_prompts(include_slurm_environment=True, approaches=["code"])
    assert "Approach restriction (`optimize.approaches`)" in bundle.optimizer
    assert "slurm-environment" in bundle.optimizer
    for flag in _BENCHMARK_CANONICAL_FLAGS:
        assert flag in bundle.evaluator, flag


# ------------------------------------------------------ per-kernel coverage


def _coverage_bundle():
    return build_perf_optimize_prompts(
        kernel_coverage={"min_share_pct": 0.5, "coverage_target_pct": 95.0}
    )


def test_kernel_coverage_note_interpolates_the_task_bars():
    for text in (
        kernel_coverage_analyzer_note(0.75, 92.0),
        kernel_coverage_ncu_targeting(0.75, 92.0),
    ):
        assert "0.75%" in text
        assert "92.0%" in text
    assert "min_share_pct: 0.75" in kernel_coverage_analyzer_note(0.75, 92.0)


def test_kernel_coverage_note_poses_all_four_questions_per_kernel():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    for question in (
        "can it be eliminated?",
        "can it be made faster?",
        "can it be fused with its neighbors?",
        "can it be overlapped with independent work?",
    ):
        assert question in block
    assert "all four questions for every row" in block
    assert "even when elimination is an item" in block
    assert "kernel_ledger.yaml" in block
    assert "aborts the stage" in block
    for field in (
        "enumerated_share_pct:",
        "other_share_pct:",
        "min_share_pct:",
        "gpu_busy_pct:",
        "full_name:",
        "share_pct:",
        "elimination:",
        "faster:",
        "fusion:",
        "overlap:",
        "why_it_runs:",
        "ref:",
    ):
        assert field in block, field
    assert "disposition: item" in block
    assert "disposition: dismissed" in block
    assert "compute | memory | latency | balanced | comm" in block


def test_kernel_coverage_note_orders_questions_by_what_they_presuppose():
    """Elimination leads, while all alternative assessments remain required."""
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    positions = [block.index(f"### Question {number} per kernel") for number in range(1, 5)]
    assert positions == sorted(positions)
    assert "even when elimination is an item" in block
    assert "Prioritize elimination over that row's alternative implementations" in block
    assert "do not add their expected gains together" in block


def test_kernel_coverage_note_grounds_elimination_in_why_the_kernel_runs():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "source and the NVTX timeline" in block
    assert "why_it_runs" in block
    for shape in ("Redundant", "Wasted", "Hoistable", "Accidental slow path"):
        assert shape in block, shape
    assert "measure the fraction that cannot affect the output" in block
    assert "faster implementation of necessary work belongs to question 2" in block
    for tag in (
        "mandatory-math",
        "padding-minimal",
        "already-hoisted",
        "fast-path-active",
        "fast-path-blocked",
        "approach-restricted",
        "accuracy-scope",
    ):
        assert tag in block, tag


def test_kernel_coverage_note_justifies_the_overlap_question():
    """A faster-execution dismissal must not also dismiss overlap."""
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "A faster-execution dismissal does not settle overlap" in block
    assert "inter-launch gaps (graph/launch amortization)" in block
    assert "underfilling the device inside a graph replay (question 4)" in block


def test_kernel_coverage_note_grounds_overlap_in_an_independent_partner():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "overlap.concurrent_with" in block
    assert "neither reads what the other writes" in block
    assert "disjoint outputs and step state" in block
    assert "evidence that they are serialized today" in block
    assert "Sum demand on the binding resource" in block
    assert "under ~100%" in block
    assert "maybe_execute_in_parallel" in block
    assert "AuxStreamType" in block
    assert "with_multi_stream(True)" in block
    assert "Plan graph enablement only if the task permits it" in block
    for tag in (
        "graph-disabled",
        "no-independent-partner",
        "resource-saturated",
        "already-concurrent",
        "below-materiality",
        "phase-boundary",
    ):
        assert tag in block, tag
    assert "identify them as alternatives" in block
    assert "count the saving once" in block


def test_kernel_coverage_note_fixes_the_materiality_unit():
    """Percent shares convert to wall time before gain and materiality checks."""
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "coverage.gpu_busy_pct" in block
    assert "percentages (0–100)" in block
    assert "wall_clock_share_pct = share_pct x gpu_busy_pct / 100" in block
    assert "time_saved_pct = wall_clock_share_pct x recovery_fraction" in block
    assert "latency_gain_pct = time_saved_pct" in block
    assert "throughput_gain_pct = 100 x time_saved_pct / (100 - time_saved_pct)" in block
    assert "50% less elapsed time implies 100% more throughput" in block
    assert "0 <= time_saved_pct < 100" in block
    assert "expected_gain_rationale" in block
    assert "optimize.noise_floor_pct" in block
    assert "Whole affected chain x best-case saving fraction" in block
    assert "min(wall_clock_share_pct_A, wall_clock_share_pct_B)" in block
    assert "separate host/launch finding" in block


def test_kernel_coverage_note_grounds_fusion_in_observed_adjacency():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "predecessor/successor launches" in block
    assert "cuda_gpu_trace" in block
    assert "producer/consumer tensors from NVTX plus source" in block
    assert "fusion.neighbors" in block
    for tag in (
        "at-sol-floor",
        "below-materiality",
        "multi-consumer-pinned",
        "already-fused",
        "phase-boundary",
        "needs-rebuild",
        "neighbors-at-bandwidth-floor",
    ):
        assert tag in block, tag
    assert "whole chain's materiality bound" in block


def test_kernel_coverage_needs_rebuild_requires_ruling_out_a_replacement():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "why the artifact cannot be rebuilt" in block
    assert "why a replacement kernel cannot help" in block
    assert "no Python-reachable dispatch to reroute" in block
    assert "no credible headroom over the tuned incumbent" in block
    assert "Otherwise plan the replacement" in block
    assert "a newly written fused kernel" in block


def test_kernel_coverage_note_bounds_the_capture():
    block = _norm(kernel_coverage_ncu_targeting(0.5, 95.0))
    assert block.startswith("2. **Select kernels by coverage")
    assert "3 passes" in block
    assert "still-missing stems" in block
    assert "8 × the pass's stem count (cap ~300)" in block
    for artifact in (
        "server_ncu_pass<k>.ncu-rep",
        "ncu_details_pass<k>.txt",
        "ncu_raw_pass<k>.csv",
    ):
        assert artifact in block
    assert "same iteration gate" in block
    assert "excluding collectives" in block
    assert 'ncu: "unavailable: <reason>"' in block
    ledger = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "Items below `optimize.noise_floor_pct` are not actionable" in ledger


def test_kernel_coverage_template_never_shows_a_note_the_schema_rejects():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    # `has_note` requires a non-empty string, so an empty exemplar would
    # teach a copy-paste that the schema then rejects.
    assert 'note: ""' not in block
    # The template demonstrates the partial-capture shape it is teaching.
    assert "occupancy_pct: null" in block
    assert 'note: "occupancy section empty' in block


def test_kernel_coverage_degrade_string_takes_bound_on_the_row():
    prompt = kernel_coverage_analyzer_note(0.5, 95.0)
    block = _norm(prompt)
    assert re.search(r'    ncu: "unavailable: collective[^\n]+\n    bound: comm', prompt)
    assert "with `bound` on the row" in block
    assert "For captured kernels, `bound` lives inside the `ncu` mapping" in block
    assert "`bound` is required in either shape" in block


def test_kernel_coverage_note_requires_marking_unmeasured_rows():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "## Kernel disposition ledger" in block
    assert "authoritative YAML" in block
    assert "mark every unmeasured bound" in block
    assert "degrade reason or null metric's `note`" in block


def test_kernel_coverage_reporter_discloses_how_much_ncu_measured():
    block = _norm(KERNEL_COVERAGE_REPORTER_GUIDANCE)
    # A ledger of degrade strings / null metrics is valid, so the reporter
    # must say so — otherwise an unmeasured coverage proof renders exactly
    # like a measured one.
    assert "Say how much of the table ncu actually measured" in block
    assert "must never render" in block


def test_kernel_coverage_bundle_extends_profiler_analyzer_and_reporter():
    base = build_perf_optimize_prompts()
    coverage = _coverage_bundle()
    assert "Per-kernel coverage contract" in coverage.analyzer
    assert "## Kernel Coverage" in coverage.reporter
    assert "Per-kernel coverage contract" not in base.analyzer
    assert "## Kernel Coverage" not in base.reporter
    for role in ("benchmarker", "projector", "optimizer", "evaluator", "qa"):
        assert getattr(coverage, role) == getattr(base, role), role


def test_kernel_coverage_selects_one_effective_ncu_policy():
    base = _norm(build_perf_optimize_prompts().profiler)
    active = _norm(_coverage_bundle().profiler)
    default_selection = (
        "2. **Pick the targets from the timeline decomposition, not the kernel sum.**"
    )
    coverage_selection = "2. **Select kernels by coverage and capture in bounded passes.**"
    assert base.count(default_selection) == 1
    assert coverage_selection not in base
    assert active.count(coverage_selection) == 1
    assert default_selection not in active
    assert "This pass targets the top kernels Run A surfaced" not in active
    run_b = active.split("## Run B", 1)[1].split("## ", 1)[0]
    assert "supersedes" not in run_b
    assert "superseded" not in run_b
    assert (
        _norm(kernel_coverage_ncu_targeting(0.5, 95.0))
        .strip()
        .replace("nsys_analysis", "capture_preprocessing")
        in active
    )


def test_kernel_coverage_carries_dismissals_only_with_current_evidence():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    for condition in (
        "~20% relative",
        "bound class is unchanged",
        "no accepted item touched the kernel",
        "carried from round <k>",
        "when the neighbor/partner changed",
        "whenever `gpu_busy_pct` changes",
    ):
        assert condition in block
    assert "Replan-only rounds keep standing measurements" in block


def test_kernel_coverage_off_leaves_bundle_unchanged():
    assert build_perf_optimize_prompts(kernel_coverage=None) == DEFAULT_PROMPTS


def test_kernel_coverage_reporter_section_slots_after_kernel_comparison():
    block = _norm(KERNEL_COVERAGE_REPORTER_GUIDANCE)
    assert 'between "Kernel-Level Comparison" and "Failed Attempts"' in block
    # Dispositions resolve to campaign outcomes and the untried tail is
    # itemized, never buried.
    assert "pending at campaign end" in block
    assert "untried tail" in block
    # All four questions get a column, and the busy share that makes the
    # shares readable as wall clock is stated.
    assert "| eliminate → | faster → | fusion → | overlap → |" in block
    assert "coverage.gpu_busy_pct" in block
    assert "all four questions" in block
    # One item can span two cells (alternative realizations) or two rows
    # (a pair) — resolved consistently, counted once.
    assert "never count its gain twice" in block
    # Honest degrade when the final ledger is missing.
    assert "Kernel coverage ledger unavailable" in block


def test_kernel_coverage_composes_with_sol_slurm_and_restriction():
    bundle = build_perf_optimize_prompts(
        include_slurm_environment=True,
        approaches=["code"],
        include_sol=True,
        kernel_coverage={"min_share_pct": 0.5, "coverage_target_pct": 95.0},
    )
    assert "Per-kernel coverage contract" in bundle.analyzer
    assert "SOL projection as context" in bundle.analyzer
    assert "Approach restriction (`optimize.approaches`)" in bundle.analyzer
    assert "slurm-environment" in bundle.profiler
    assert "slurm-environment" not in bundle.analyzer
    assert "## Kernel Coverage" in bundle.reporter
    assert "Projection vs Measured" in bundle.reporter


def test_measuring_roles_inherit_the_server_identity_checks():
    """Every role that launches a server must carry the stale-server guards.

    perf-optimize is where a stale server does the most damage: each
    `approach: config` item rewrites the *same*
    `tuning/extra_llm_api_options.yaml` against the *same* checkpoint, so a
    survivor from the previous round answers on :8000 under a matching
    model name and the evaluator's measured gain — the accept/reject gate
    — silently scores the wrong config.
    """
    for name, prompt in (
        ("benchmarker", BENCHMARKER_SYSTEM_PROMPT),
        ("optimizer", OPTIMIZER_SYSTEM_PROMPT),
        ("evaluator", EVALUATOR_SYSTEM_PROMPT),
        ("integrator", INTEGRATOR_SYSTEM_PROMPT),
        ("qa", QA_SYSTEM_PROMPT),
    ):
        text = _norm(prompt)
        assert "port 8000 already in use" in text, f"{name} lost the port precheck"
        assert "owns_port" in text, f"{name} lost the listener-ownership check"
        assert "not owned by PID" in text, f"{name} lost the identity failure path"


# --------------------------------------------------------------------------- #
# Consuming the nsys timeline analysis: headroom bounds what an item may
# claim, and the skill's opportunity list must be accounted for rather than
# read once and paraphrased into prose.
# --------------------------------------------------------------------------- #


def test_analyzer_bounds_expected_gain_by_measured_headroom() -> None:
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    for field in ("bounding_resource", "bounding_pct", "headroom_verdict"):
        assert field in prompt
    assert "bound faster-execution claims" in prompt
    assert "elimination, fusion or independent-overlap" in prompt
    assert "utilization cannot support an item alone" in prompt
    assert "If ncu's bound disagrees with utilization" in prompt


def test_analyzer_must_account_for_every_nsys_opportunity() -> None:
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "nsys_analysis/items.json" in prompt
    assert "`nsys_items` accounts for the timeline analysis, one row per id" in prompt
    assert "disposition: item" in prompt
    assert "disposition: dismissed" in prompt
    assert "an unaccounted id does" in prompt
    assert "Never pad the roadmap" in prompt


def test_roadmap_spec_documents_the_nsys_items_block():
    spec = _norm(ROADMAP_SPEC)
    assert "nsys_items:" in spec
    assert "disposition: item" in spec
    assert "disposition: dismissed" in spec
    # Same vocabulary as the kernel ledger, and the same reason for it.
    assert "must name a real roadmap item id, any status" in spec
    # Absent when the pipeline could not run; required when it did.
    assert "omitted entirely when it does not" in spec
    # Ids are local to the analysis that wrote them, so the block is
    # authored fresh each round rather than carried forward — a stale row
    # names an id this round's file does not have.
    assert "author the block fresh from this round's `items.json`" in spec
    assert "replan-only rounds use the standing analysis" in spec
    assert "never carried forward" in spec


def test_analyzer_categorizes_imbalance_by_the_work_not_the_collective() -> None:
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "Categorize imbalance by `imbalance_operator`'s work, not communication" in prompt
    assert "uneven experts are `compute`" in prompt
    assert "uneven KV footprint is `kv-capacity`" in prompt
    assert "`pinned` / `rotating`" in prompt
    assert "A pinned machine issue may be outside this campaign" in prompt
    assert "rotating imbalance calls for work distribution" in prompt
    assert "bound recovery by `pct_of_iter`, not the whole rank spread" in prompt


# ------------------------------------------------------ unified performance model


def test_kernel_ledger_updates_the_model_on_every_analyzer_turn():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "Every analyzer turn, including re-analysis and replan-only rounds" in block
    assert "copy the standing measurements and coverage into a fresh ledger" in block
    assert "preserve measurement provenance" in block.lower()
    assert "append-only `model_revisions`" in block
    assert "record each changed model field with its previous and new value" in block.lower()
    assert "Include the full current section on **replan-only** turns" in block
    assert "name their source round" in block
    assert "preserve the imported report verbatim" in block
    assert "clearly labeled current-campaign section with this heading" in block
    assert "Preserve imported anchors and give the appended section a unique anchor" in block
    assert "`current-campaign-` if its round anchor already exists in imported text" in block


def test_kernel_model_converges_from_facts_and_preserves_unknowns():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "A failed optimization alone cannot justify relaxing the model" in block
    assert "Measured time below a predicted lower bound exposes an invalid model" in block
    assert "never clamp the measurement" in block
    assert "Unknown predictions or measurements are `null`" in block
    assert "unknown never means no opportunity" in block
    assert "facts explain the residual" in block
    assert "unexplained gaps open" in block
    assert "matching units and scope" in block
    assert "Never sum overlapping kernel durations" in block
    assert "logical region shared by several kernels" in block
    report = block.split("### Required analyzer report section:", 1)[1].split(
        "### The kernel ledger contract", 1
    )[0]
    assert "## Per-layer theoretical performance model" in report
    assert "before **## Ranked bottleneck hypotheses**" in report
    assert "HTML anchor `per-layer-theoretical-performance-model-round-N`" in report
    assert "logical layers from config/source" in report
    assert "model IDs and kernel/region evidence" in report
    assert "index range/count" in report
    assert "per-layer, averaged and repeated-total timings" in report
    assert "sharding" in report
    assert "shapes/dtypes, FLOPs, necessary memory traffic" in report
    assert "collectives" in report
    assert "Substitute numbers and units into formulas" in report
    assert "hardware constants and their sources" in report
    assert "Separate theoretical minimum latency from empirical practical estimates" in report
    assert "Label which kind `predicted_ms` represents in `derivation`" in report
    assert "record both calculations there when both are available" in report
    assert "Match the operating point and timing scope" in report
    assert "unexplained residuals and the next discriminating test" in report
    assert "critical-path lower bound" in report
    assert "never double count regions or alternative savings" in report
    assert "Label incomplete totals as partial" in report
    assert "include the `sol.json` comparison here once" in report
    assert "calculator columns and region scopes intact" in report
    assert "add layer totals only with an explicit mapping" in report
    assert "equations, bounds, comparison tables and gap explanations in this section" in report
    assert "nsys/ncu sections own measured diagnostics" in report
    assert "kernel disposition ledger owns decisions" in report
    assert "ranked hypotheses own prioritized experiments" in report
    assert "Both reference the model's gaps instead of repeating them" in report
    assert "Remaining-gap attribution is brief campaign-end accounting with links" in report


@pytest.mark.parametrize("include_sol", [False, True])
def test_per_layer_report_keeps_absolute_and_relative_practical_gaps(include_sol: bool) -> None:
    bundle = build_perf_optimize_prompts(
        include_sol=include_sol,
        kernel_coverage={"min_share_pct": 0.5, "coverage_target_pct": 95.0},
    )
    report = (
        _norm(bundle.analyzer)
        .split("### Required analyzer report section:", 1)[1]
        .split("### The kernel ledger contract", 1)[0]
    )
    assert "**Gap vs practical (ms)** and **Gap vs practical (%)**" in report
    assert "gap_ms = measured_ms - practical_ms" in report
    assert "gap_pct = gap_ms / practical_ms * 100" in report
    assert "Calculate before rounding" in report
    assert "show signed percentages to one decimal place" in report
    assert "Show `—` for percentages with missing or mismatched timings" in report
    assert "practical_ms <= 0" in report
    assert "Label gaps against theoretical estimates separately" in report


@pytest.mark.parametrize("include_sol", [False, True])
def test_unified_ledger_exposes_models_only_to_analyzer_and_reporter(include_sol):
    bundle = build_perf_optimize_prompts(
        include_sol=include_sol,
        kernel_coverage={"min_share_pct": 0.5, "coverage_target_pct": 95.0},
    )
    assert "version: 2" in bundle.analyzer
    assert "models:" in bundle.analyzer
    assert "## Per-layer theoretical performance model" in bundle.analyzer
    analyzer = _norm(bundle.analyzer)
    assert "## SOL correlation (measured vs ceiling)" not in analyzer
    assert ("sol_calc.py analyze" in analyzer) is include_sol
    if include_sol:
        for artifact in ("regions.json", "sol.json", "sol_recipes/", "sol_work/peaks.json"):
            assert artifact in analyzer
        for field in (
            "region",
            "calls",
            "measured ms",
            "SOL ms",
            "% of SOL",
            "MFU %",
            "MBU %",
            "gap ms",
            "bound",
        ):
            assert field in analyzer
        assert "Correlation unavailable" in analyzer
    reporter = _norm(bundle.reporter)
    assert "Theoretical headroom summary" in reporter
    assert "link directly to **Per-layer theoretical performance model**" in reporter
    assert "relative link with its section anchor in both Markdown and HTML" in reporter
    assert "Use the actual current-campaign anchor" in reporter
    assert "an imported section with the same heading is historical evidence" in reporter
    assert "do not reproduce or re-derive them here" in reporter
    assert "Theoretical model vs silicon" not in bundle.reporter
    assert "if the analyzer section is absent, report it as unavailable" in reporter.lower()
    assert "| eliminate → | faster → | fusion → | overlap → |" in bundle.reporter
    assert ("Projection vs Measured" in bundle.reporter) is include_sol
    base = build_perf_optimize_prompts(include_sol=include_sol)
    assert "## Per-layer theoretical performance model" not in base.analyzer
    assert "## SOL correlation (measured vs ceiling)" in _norm(base.analyzer)
    assert ("sol_calc.py analyze" in base.analyzer) is include_sol
    if include_sol:
        assert SOL_ANALYZER_CONTEXT in base.analyzer
    for role in ("optimizer", "evaluator", "qa"):
        assert getattr(bundle, role) == getattr(base, role)
    for role in _ALL_PROMPTS:
        assert "headroom_ledger.yaml" not in getattr(bundle, role)
        assert "## Headroom Accounting" not in getattr(bundle, role)


def test_evaluator_carries_the_change_not_live_implication():
    # It looks identical to applied-but-no-gain in the numbers and means
    # the opposite: one bounds the headroom, the other bounds nothing.
    block = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "change-not-live" in block
    assert "bounds nothing, because the mechanism was never tested" in block


def test_evaluator_is_told_to_pass_the_structured_fields():
    block = _norm(EVALUATOR_SYSTEM_PROMPT)
    for field in ("gap_implication", "gap_implication_note", "lever", "target_blocker"):
        assert field in block, field
    assert "Prose is not a contract" in block
    # The blocker is a fact about the code that it verifies, not authors.
    assert "Forward, never author" in block


# ---------------------------------------------------- the workspace prompt snapshot


_SNAPSHOT_ROLES = (
    "benchmarker",
    "projector",
    "profiler",
    "analyzer",
    "optimizer",
    "evaluator",
    "integrator",
    "qa",
    "reporter",
)


def test_dump_writes_every_role_verbatim(tmp_path):
    """A snapshot a reader can diff against the source module."""
    bundle = build_perf_optimize_prompts(include_sol=True)
    dump_prompt_bundle(bundle, tmp_path / "prompts")

    written = {p.stem: p.read_text(encoding="utf-8") for p in tmp_path.glob("prompts/*.md")}
    assert written == {role: getattr(bundle, role) for role in _SNAPSHOT_ROLES}


def test_dump_clears_a_previous_launch_stale_role(tmp_path):
    """Two versions' prompts in one directory would read as one campaign's."""
    directory = tmp_path / "prompts"
    directory.mkdir()
    (directory / "retired_role.md").write_text("from an older version\n", encoding="utf-8")

    dump_prompt_bundle(build_perf_optimize_prompts(), directory)

    assert not (directory / "retired_role.md").exists()
    assert (directory / "analyzer.md").is_file()


def test_profiler_emits_only_the_effective_config_and_replay_policies() -> None:
    for coverage in (None, {"min_share_pct": 0.5, "coverage_target_pct": 95.0}):
        bundle = build_perf_optimize_prompts(
            approaches=["code"], include_sol=True, kernel_coverage=coverage
        )
        prompt = _norm(bundle.profiler)
        assert prompt.count("Effective profiling point policy") == 1
        assert "--extra_llm_api_options <active tuning config>" in prompt
        assert "Treat it and the accepted config snapshot as read-only" in prompt
        assert "only when** `task.yaml` sets" not in prompt
        assert "Ignore the earlier instruction" not in prompt
        assert "every item must be `approach: config`" not in prompt
        assert "still measure and report every point" not in prompt
        assert "**One run per concurrency point.**" not in prompt
        assert "replay only the **largest**" in prompt
        assert "lowest and highest scored concurrency points" not in prompt


def test_analyzer_omits_other_roles_implementation_and_verdict_instructions() -> None:
    prompt = _norm(build_perf_optimize_prompts(approaches=["code"]).analyzer)
    assert "**Optimizer**" not in prompt
    assert "**Evaluator**" not in prompt
    assert "Allowed roadmap approaches: `code`" in prompt
    assert "`config` is off-limits" in prompt
    assert "what you searched" in prompt
    assert "how_to_apply" in prompt
    assert "Never apply optimizations" in prompt
    assert "Curve worked example" not in prompt


def test_analyzer_replan_preserves_measurements_and_history() -> None:
    prompt = _norm(_coverage_bundle().analyzer)
    assert "launch no server, run no profiler" in prompt
    assert "Do not regenerate measured artifacts" in prompt
    assert "full findings structure applies only to full analysis, including re-analysis" in prompt
    assert "replan/reuse notes must also include" in prompt
    assert "its full current per-layer theoretical performance model section" in prompt
    assert "freeze `baseline`" in prompt
    assert "preserve all accepted / failed / in_progress items" in prompt
    assert "Never renumber or reuse ids" in prompt


def test_profiler_and_analyzer_have_separate_artifact_ownership() -> None:
    profiler = _norm(PROFILER_SYSTEM_PROMPT)
    analyzer = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "rounds/round_<n>/profile/" in profiler
    assert "capture_preprocessing" in profiler
    assert "append_profiler_progress" in profiler
    assert "Write `profile_manifest.json` last" in profiler
    assert "never overwrite the profiler's preliminary decomposition" in analyzer
    assert "`<profile_dir>` means the read-only source capture directory" in analyzer
    assert "rounds/round_<n>/analysis/" in analyzer
    assert "append_analyzer_progress" in analyzer
    assert "## Required findings structure" not in profiler
    assert "## The roadmap contract" not in profiler
    assert "Author `<workspace>/nsys_analysis/items.json`" not in profiler


def test_profiler_manifest_preserves_runtime_and_capture_provenance() -> None:
    prompt = _norm(PROFILER_SYSTEM_PROMPT)
    for field in (
        '"schema_version": 1',
        '"capture_id"',
        '"serve_command"',
        '"benchmark_command"',
        '"config"',
        '"build"',
        '"import_path"',
        '"checkout"',
        '"hardware"',
        '"model"',
        '"workload"',
        '"operating_points"',
        '"profile_ranks"',
        '"methods"',
        '"status": "captured"',
        '"status": "unavailable"',
        '"artifacts"',
        '"limitations"',
    ):
        assert field in prompt, field
    assert "Every configured method needs a final" in prompt
    assert "Every listed artifact is an existing nonempty file" in prompt
    assert "No final `pending` or `failed` status is valid" in prompt
    assert "completed capture must survive an Analyzer failure" in prompt
    assert "mutable active-tuning path alone cannot establish what ran" in prompt
    assert "top-level `artifacts` so they travel with imported captures" in prompt


def test_sol_calibration_is_owned_by_profiler_and_snapshotted() -> None:
    base = build_perf_optimize_prompts(include_sol=False)
    sol = build_perf_optimize_prompts(include_sol=True)
    prompt = _norm(sol.profiler)
    assert "measure_channels.py --launch" in prompt
    assert "servers stopped and the GPU idle" in prompt
    assert "profile directory" in prompt
    assert "as `sol_peaks.json`" in prompt
    assert "manifest's top-level `artifacts`" in prompt
    assert "measure_channels.py" not in base.profiler
    assert "measure_channels.py" not in sol.analyzer


def test_analyzer_can_reanalyze_saved_captures_without_runtime_access() -> None:
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "**re-analysis of existing captures**" in prompt
    assert "A new capture is not required" in prompt
    assert "earlier round or another workspace" in prompt
    assert "nsys export --type sqlite" in prompt
    assert "nsys stats --report" in prompt
    assert "ncu --import <profile_dir>/server_ncu.ncu-rep" in prompt
    assert "--metrics-profile 0=<workspace>/server_nsys_metrics.sqlite" in prompt
    assert "taxonomy before quoting a single category number" in prompt
    assert "Never run a workload to fill a missing file" in prompt
    assert "Additional capture requested:" in prompt
    assert "Do not launch a runtime probe to re-analyze" in prompt
    assert "runtime.import_path" in prompt
    assert "The current source checkout may differ from the profiled build" in prompt


def test_analyzer_has_no_capture_commands_under_any_extensions() -> None:
    for bundle in (
        DEFAULT_PROMPTS,
        build_perf_optimize_prompts(
            include_slurm_environment=True,
            include_disagg=True,
            include_sol=True,
            kernel_coverage={"min_share_pct": 0.5, "coverage_target_pct": 95.0},
        ),
    ):
        prompt = bundle.analyzer
        for command in (
            "nsys profile",
            "ncu --target-processes",
            "setsid env",
            "python submit.py",
            "squeue -j",
            "--container-image",
            "Effective profiling point policy",
            "benchmark_serving.py",
            "measure_channels.py",
            "you just profiled on it",
        ):
            assert command not in prompt, command
        assert "never launch a server" in prompt


def test_disagg_analyzer_gets_only_capture_interpretation_context() -> None:
    bundle = build_perf_optimize_prompts(include_disagg=True)
    assert "Disaggregated capture interpretation" in bundle.analyzer
    assert "Context and generation" in bundle.analyzer
    assert "different clocks" in bundle.analyzer
    assert "first-class `communication` cost" in bundle.analyzer
    assert "python submit.py" in bundle.profiler
    assert "Disaggregated serving (supersedes" not in bundle.analyzer


def test_reanalysis_refreshes_ledgers_and_correlation_without_recapture() -> None:
    bundle = build_perf_optimize_prompts(
        include_sol=True,
        kernel_coverage={"min_share_pct": 0.5, "coverage_target_pct": 95.0},
    )
    prompt = _norm(bundle.analyzer)
    assert "Re-analysis rebuilds the ledger from saved evidence" in prompt
    assert "Re-run correlation in full analyses, including re-analysis" in prompt
    assert "without collecting measurements" in prompt
    assert "Never run a GPU microbenchmark" in prompt
    assert "<campaign_workspace>/sol_work/peaks.json" in prompt
    assert "**Full analyses**: author the block fresh" in prompt
    assert "replan-only rounds use the standing analysis" in prompt
    for artifact in ("regions.json", "sol.json", "sol_recipes/"):
        assert artifact in prompt
    assert "## SOL correlation (measured vs ceiling)" not in prompt
    assert "## Per-layer theoretical performance model" in prompt


def test_prompt_bundle_profiler_extensions_are_independent() -> None:
    extended = DEFAULT_PROMPTS.with_extensions(
        profiler="Custom capture instructions", analyzer="Custom interpretation instructions"
    )
    assert extended.profiler.endswith("\n\nCustom capture instructions")
    assert extended.analyzer.endswith("\n\nCustom interpretation instructions")
    assert "Custom capture instructions" not in extended.analyzer
    assert "Custom interpretation instructions" not in extended.profiler
    assert DEFAULT_PROMPTS.with_extensions(profiler=" \n") == DEFAULT_PROMPTS


def test_custom_bundle_without_profiler_retains_default_capture_prompt() -> None:
    legacy_roles = {role: f"custom {role}" for role in _SNAPSHOT_ROLES if role != "profiler"}
    bundle = PromptBundle(**legacy_roles)
    assert bundle.profiler == PROFILER_SYSTEM_PROMPT
    assert build_profiler_prompt() == DEFAULT_PROMPTS.profiler
    for role, prompt in legacy_roles.items():
        assert getattr(bundle, role) == prompt


def test_reporter_pairs_raw_captures_with_their_completed_analysis() -> None:
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "profile/profile_manifest.json" in prompt
    assert "analysis/analysis_manifest.yaml" in prompt
    assert "Re-analysis alone never makes a capture newer" in prompt
    assert "Match analysis and capture identities" in prompt
    assert "associated `analysis/nsys_analysis/`" in prompt


def test_evaluator_comparative_analysis_preserves_previous_capture() -> None:
    prompt = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "Keep previous captures and analyses read-only" in prompt
    assert "associated round's `analysis/taxonomy.json`" in prompt
    assert "<previous capture or analysis server_nsys.sqlite>" in prompt


def test_kernel_ledger_prompt_example_validates(tmp_path):
    from agent_flow.workflows.perf_optimize.kernel_ledger import load_ledger

    prompt = kernel_coverage_analyzer_note(0.5, 95.0)
    example = prompt.split("```yaml\n", 1)[1].split("```", 1)[0]
    path = tmp_path / "kernel_ledger.yaml"
    path.write_text(example, encoding="utf-8")
    ledger = load_ledger(path)
    assert ledger["version"] == 2
    assert {row["model"] for row in ledger["kernels"]} == {
        model["id"] for model in ledger["models"]
    }
    assert any(model["predicted_ms"] is None for model in ledger["models"])


def test_projector_records_initial_model_without_freezing_future_analysis():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    assert "Later facts may expose a faulty assumption or omitted cost" in prompt
    assert "the analyzer maintains the best current model" in prompt.lower()
    assert "ceiling stays valid for every later round" not in prompt


def test_reporter_distinguishes_superseded_projection_from_current_model():
    block = _norm(SOL_OPTIMIZE_REPORTER_GUIDANCE)
    assert "superseded initial model" in block
    assert "cite the model revision" in block
    assert "Never present a falsified prediction" in block
    assert "Attribute missing time to host/scheduler costs only when traces" in block
    assert (
        "a large discrepancy alone does not distinguish implementation inefficiency from model error"
        in block
    )
