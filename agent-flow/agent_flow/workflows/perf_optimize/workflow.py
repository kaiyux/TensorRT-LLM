from __future__ import annotations

import json
import re
import shlex
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from rich.markup import escape

from agent_flow import (
    CLAUDE_CODE_DEFAULT_MODEL,
    AgentLayer,
    AgentLayerConfig,
    BackendConfig,
    SessionConfig,
    require_tool_call_stop_hook,
)
from agent_flow.console import print_message, print_rule
from agent_flow.logger import get_logger
from agent_flow.workflows.perf_analyze import performance_model
from agent_flow.workflows.perf_analyze.prompts._common import profile_ranks_note
from agent_flow.workflows.perf_analyze.sol_methodology import (
    SolMethodology,
    output_instruction,
    projector_instruction,
)
from agent_flow.workflows.perf_analyze.workflow import clear_stale_benchmark_results

from . import gitops, kernel_ledger, measurements, nsys_items, reuse, roadmap_schema
from .disagg import disagg_config_path, has_disagg, load_disagg_config, worker_config_yaml
from .profile import (
    PROFILE_MANIFEST_NAME,
    PROFILE_REPORT_NAME,
    ProfileError,
    validate_profile_manifest,
)
from .progress import (
    EVALUATOR_DECISIONS,
    GAP_IMPLICATIONS,
    INTEGRATOR_DECISIONS,
    MEASUREMENT_CONFIDENCES,
    OPTIMIZATION_STAGE,
    ProgressContext,
    append_workflow_event,
    build_progress_tools,
    init_progress_file,
    latest_entry,
    read_progress,
)
from .prompts import DEFAULT_PROMPTS, PromptBundle
from .roadmap_schema import RoadmapError
from .state import (
    ROUND_STAGES,
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
    STATE_FILENAME,
    WorkflowState,
    load_state,
    save_state,
)
from .task_schema import (
    OPTIMIZE_DEFAULTS,
    concurrency_points,
    dump_task_yaml,
    focus_concurrencies,
    has_slurm_environment,
    is_curve_mode,
    kernel_coverage,
    load_and_validate_task_yaml,
    max_regression_pct,
    profile_ranks,
    sol_enabled,
)


def _progress_has_entries(path: Path) -> bool:
    """Return True iff ``path`` holds a progress.yaml with real entries.

    Distinguishes the empty ``{optimization: []}`` shell left by
    ``init_progress_file`` (which should not block a retry) from real
    entries written by an earlier run (which should). Missing/empty files
    count as no entries. Malformed YAML counts as "has content" so the
    user is forced to ``--clean`` rather than silently losing the bad
    file.
    """
    if not path.is_file():
        return False
    try:
        data = read_progress(path)
    except (ValueError, yaml.YAMLError):
        return True
    return bool(data[OPTIMIZATION_STAGE])


def _compose_required_tools_hooks(required_tools: list[str]) -> dict | None:
    """Compose stop hooks that require *every* listed tool to be called.

    ``require_tool_call_stop_hook`` enforces "at least one of the listed
    names was called". Stacking one such hook per tool — each independent
    — yields AND semantics: every per-tool hook must allow the stop, so
    all listed tools must have been called this turn.
    """
    if not required_tools:
        return None
    merged: dict[str, list] = {"Stop": []}
    for name in required_tools:
        merged["Stop"].extend(require_tool_call_stop_hook([name])["Stop"])
    return merged


def _make_agent(
    name: str,
    system_prompt: str,
    tools: list | None = None,
    required_tools: list[str] | None = None,
    backend_kind: str = "claude-code",
    model: str = CLAUDE_CODE_DEFAULT_MODEL,
    session_mode: str = "persistent",
    cwd: Path | None = None,
) -> AgentLayer:
    hooks = _compose_required_tools_hooks(required_tools or [])
    return AgentLayer(
        AgentLayerConfig(
            name=name,
            system_prompt=system_prompt,
            backend=BackendConfig(
                kind=backend_kind,
                model=model,
                tools=tools,
                hooks=hooks,
                cwd=cwd,
            ),
            session=SessionConfig(mode=session_mode),
        )
    )


_ROLES = (
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


class PerfOptimizeWorkflow:
    """Iterative optimization loop over a TRT-LLM serving setup.

    ``benchmarker`` measures the baseline once; the one-shot
    ``projector`` stage then runs between the baseline and round 1 —
    unless task.yaml sets ``sol.enabled: false`` — writing
    ``sol_projection.md``, the analytical speed-of-light ceiling per the
    ``internal-perf-sol-analysis`` skill (or ``perf-analysis`` where that
    is not installed), that the analyzer weighs when ranking roadmap
    items (and answers for when leaving the roadmap exhausted with
    headroom remaining — the remaining-gap attribution), the optimizer
    aims each item's realization with, and the reporter turns into a
    headroom-captured story closed by a remaining-gap accountability
    breakdown; disabled, that stage is skipped. Then the loop runs
    up to ``max_rounds`` rounds of ``profiler`` (capture evidence) →
    ``analyzer`` (interpret saved evidence and rank ``roadmap.yaml`` by
    expected benefit) → a batch of up to ``max_items_per_round`` pending
    items running serially or concurrently in isolated worktrees (per item:
    ``optimizer`` ⇄ ``evaluator`` — review code/functionality/perf against
    the acceptance gate; the evaluator's three-way verdict either APPROVEs
    the attempt, REJECTs the item terminally, or PUSH_BACKs it to the
    optimizer with feedback, bounded by ``max_attempts_per_item`` — so
    every item keeps its own measured gain, verdict, and revert). No
    agent decides when to stop, and none decides what a round costs
    either. A round that accepted work is re-profiled. When every candidate
    was rejected, its isolated worktree is removed without changing the
    campaign checkout, so the next round opens **replan-only** — the analyzer
    re-plans from the standing profile and the failed items' evidence without
    launching a server. The loop
    runs the full round budget unless a deterministic break fires — an
    analyzer turn leaves the roadmap with no actionable pending item (a
    roadmap that runs dry *between* items earns one more round first, so
    the campaign always closes on a plan made against its latest
    measurements), or the optional ``optimize.target_improvement_pct``
    is met on the roadmap ledger.
    ``qa`` then runs **once** as the
    campaign's final verification (independent benchmark + optional
    accuracy eval; skipped when nothing was accepted), and ``reporter``
    synthesizes
    ``optimization_report.md`` / ``.html``. All nine roles
    run on the Claude Code backend, with sessions scoped to each role's
    unit of work: the analyzer keeps one session for the whole campaign
    (its roadmap memory), the optimizer's session spans a single item's
    attempts and is reset between items, and the profiler, evaluator and QA are
    stateless — every verdict gets fresh eyes. The evaluator and QA
    deliberately never see the SOL projection: their gates stay
    measured-vs-measured.

    The orchestrator — not the agents — owns the TRT-LLM checkout's git
    state (dedicated branch, one commit per accepted item, hard revert of
    rejected/pushed-back attempts) and every ``roadmap.yaml`` lifecycle
    field, driven by the evaluator's structured progress decisions.

    ``reuse_analysis`` seeds a fresh workspace from a previous
    ``perf-analyze`` run or ``perf-optimize`` campaign (see
    :mod:`.reuse`): the baseline report, the SOL projection, and the
    newest profile findings + traces are copied into this workspace's
    round-1 layout, the benchmarker and projector stages are marked done,
    and round 1's analyzer runs **plan-only** — authoring
    ``roadmap.yaml`` from the imported evidence without launching a
    server or a profiler. With ``reanalyze=True``, the analyzer instead
    regenerates derived evidence from preserved captures. Each completed
    analysis records its source capture in ``analysis_manifest.yaml``.

    Every transition checkpoints before the next agent runs, so a crash /
    Ctrl-C resumes at the same stage with the same round/attempt indices.
    """

    def __init__(
        self,
        workspace: Path,
        clean: bool = False,
        prompts: PromptBundle | None = None,
        max_rounds_override: int | None = None,
        reuse_analysis: str | Path | None = None,
        sol_methodology: SolMethodology | None = None,
        reanalyze: bool = False,
    ) -> None:
        if reanalyze and reuse_analysis is None:
            raise ValueError("reanalyze requires reuse_analysis")
        self.reanalyze = reanalyze
        self.workspace = workspace
        self.prompts = prompts or DEFAULT_PROMPTS
        # Which SOL methodology skill this session has. Resolved by the CLI
        # before the run (so the projector's prompt matches), and defaulted to
        # the full methodology here so a direct constructor call — the test
        # suite included — never pays a live probe.
        self.sol_methodology = sol_methodology or SolMethodology()
        self.max_rounds_override = max_rounds_override
        self.reuse_analysis = Path(reuse_analysis).expanduser() if reuse_analysis else None
        self.task_path = workspace / "task.yaml"
        self.baseline_dir = workspace / "baseline"
        self.baseline_results_path = self.baseline_dir / "benchmark_results.md"
        self.sol_projection_path = workspace / "sol_projection.md"
        self.tuning_dir = workspace / "tuning"
        self.tuning_config_path = self.tuning_dir / "extra_llm_api_options.yaml"
        self.tuning_accepted_path = self.tuning_dir / "extra_llm_api_options.accepted.yaml"
        self.roadmap_path = workspace / "roadmap.yaml"
        self.sol_work_dir = workspace / "sol_work"
        # Where ``--reuse-analysis`` parks its provenance manifest and the
        # source campaign's roadmap (prior art, never the live ledger).
        self.reuse_dir = workspace / reuse.REUSE_DIRNAME
        self.reuse_manifest_path = self.reuse_dir / reuse.MANIFEST_NAME
        self.prior_roadmap_path = self.reuse_dir / reuse.PRIOR_ROADMAP_NAME
        self.rounds_dir = workspace / "rounds"
        self.worktrees_dir = workspace / "worktrees"
        self.final_verification_dir = workspace / "final_verification"
        self.verification_report_path = self.final_verification_dir / "verification_report.md"
        self.final_analysis_dir = self.final_verification_dir / "analysis"
        self.report_path = workspace / "optimization_report.md"
        self.report_html_path = workspace / "optimization_report.html"
        self.progress_path = workspace / "progress.yaml"
        self.state_path = workspace / STATE_FILENAME
        self._checkpoint_lock = Lock()
        self._progress_lock = Lock()

        self.workspace.mkdir(parents=True, exist_ok=True)
        if clean:
            # Remove linked worktrees through git before deleting their
            # directories, otherwise the source repository retains stale
            # worktree registrations.
            if self.state_path.is_file() and self.task_path.is_file():
                try:
                    stale_state = load_state(self.state_path)
                    task_data = yaml.safe_load(self.task_path.read_text(encoding="utf-8"))
                    stale_repo = (
                        task_data.get("trtllm_repo_path") if isinstance(task_data, dict) else None
                    )
                    stale_paths = [
                        str(entry.get("item_worktree_path", "")) for entry in stale_state.item_batch
                    ]
                    stale_paths.append(stale_state.integration_worktree_path)
                    for stale_path in stale_paths:
                        if stale_repo and stale_path and Path(stale_path).exists():
                            gitops.remove_worktree(str(stale_repo), stale_path)
                except (OSError, ValueError, yaml.YAMLError, gitops.GitOpsError):
                    # ``--clean`` must still recover from a corrupt checkpoint;
                    # git's later `worktree prune` can remove an orphan record.
                    pass
            # Wipe the workflow's managed files and directories so the
            # constructor proceeds as a fresh run. The TRT-LLM checkout is
            # NOT touched — abandoned optimization branches are left for
            # the user to inspect/delete.
            for path in (
                self.state_path,
                self.sol_projection_path,
                self.roadmap_path,
                self.report_path,
                self.report_html_path,
                self.progress_path,
            ):
                path.unlink(missing_ok=True)
            for directory in (
                self.baseline_dir,
                self.rounds_dir,
                self.worktrees_dir,
                self.tuning_dir,
                self.final_verification_dir,
                self.reuse_dir,
                self.sol_work_dir,
            ):
                shutil.rmtree(directory, ignore_errors=True)

        # Resume is auto-detected from the checkpoint's presence;
        # ``--clean`` has just wiped it if the user wanted to start over.
        self.resume = self.state_path.is_file()

        if not self.resume:
            # On a fresh run, every managed output must be empty so we
            # don't silently scribble over a prior run the user forgot
            # about. ``task.yaml`` is exempt — it's (re)written from the
            # validated spec in ``_init_state``.
            guarded = [
                self.baseline_results_path,
                self.sol_projection_path,
                self.roadmap_path,
                self.report_path,
                self.report_html_path,
            ]
            existing = [p for p in guarded if p.is_file() and p.read_text(encoding="utf-8").strip()]
            for directory in (self.rounds_dir, self.final_verification_dir):
                if directory.is_dir() and any(directory.iterdir()):
                    existing.append(directory)
            if _progress_has_entries(self.progress_path):
                existing.append(self.progress_path)
            if existing:
                names = ", ".join(p.name for p in existing)
                raise FileExistsError(
                    f"{names} already contains content in {self.workspace} "
                    f"but no checkpoint was found. Pass --clean to "
                    f"overwrite, or delete the file(s) manually to start "
                    f"fresh."
                )

            self.baseline_dir.mkdir(parents=True, exist_ok=True)
            self.baseline_results_path.write_text("", encoding="utf-8")
            # ``sol_projection.md`` is managed unconditionally — the
            # constructor cannot see the task spec (only ``run`` does), so
            # a run without a ``sol`` block simply leaves it blank.
            self.sol_projection_path.write_text("", encoding="utf-8")
            self.roadmap_path.write_text("", encoding="utf-8")
            self.report_path.write_text("", encoding="utf-8")
            # ``report_html_path`` stays absent until the Reporter writes
            # it — keeping it missing rather than blank makes the
            # "Reporter produced HTML" check robust to empty-file edge
            # cases.
            init_progress_file(self.progress_path)
            # task.yaml and the tuning config are materialized from the
            # validated spec in ``_init_state``.

        # The tool handlers close over this context; updating its fields
        # before each agent call stamps every entry with the right loop
        # position without the agent having to pass it.
        self._progress_ctx = ProgressContext(
            path=self.progress_path,
            global_lock=self._progress_lock,
        )
        progress_tools = build_progress_tools(self._progress_ctx)

        for role in _ROLES:
            setattr(
                self,
                role,
                _make_agent(
                    role,
                    getattr(self.prompts, role),
                    progress_tools[role],
                    required_tools=[f"append_{role}_progress"],
                    # Sessions are scoped to each role's unit of work: the
                    # judges (evaluator, qa) are stateless so every verdict
                    # gets fresh eyes, uninfluenced by earlier attempts' /
                    # rounds' conclusions; the analyzer keeps campaign-long
                    # memory of the roadmap it authored.
                    session_mode=(
                        "stateless"
                        if role in ("profiler", "qa", "evaluator", "integrator")
                        else "persistent"
                    ),
                ),
            )
        self._progress_tools = progress_tools

    def __enter__(self) -> "PerfOptimizeWorkflow":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for role in _ROLES:
            layer = getattr(self, role)
            if hasattr(layer, "__exit__"):
                layer.__exit__(None, None, None)

    # ------------------------------------------------------------- orchestration

    def run(self, task: str) -> None:
        log = get_logger().console

        state = self._init_state(task, log)
        if state is None:
            return

        try:
            self._ensure_optimization_branch(state, log)

            # ---- one-shot: baseline ----
            if state.stage == STAGE_BENCHMARKER:
                print_rule("[bold cyan]Benchmarker (baseline)[/bold cyan]", log)
                clear_stale_benchmark_results(self.baseline_dir)
                self._run_benchmarker(state)
                self._require_stage_outputs(STAGE_BENCHMARKER, [self.baseline_results_path])
                self._require_baseline_measurement()
                state.benchmarker_done = True
                state.stage = STAGE_PROJECTOR
                self._checkpoint(state)

            # ---- one-shot: SOL projection (conditional) ----
            if state.stage == STAGE_PROJECTOR:
                # The stage transition is unconditional but execution is
                # gated on the resolved task.yaml — this also covers the
                # resume edge where a checkpoint parked at the projector
                # is re-run after the stage was disabled. A projection
                # imported by ``--reuse-analysis`` arrives already marked
                # done: the ceiling is a property of hardware + model +
                # operating point, so re-deriving it would only restate it.
                if state.projector_done:
                    print_message(
                        f"[dim]projector skipped — reusing the projection "
                        f"imported from {state.reuse_analysis_dir}[/dim]",
                        log,
                    )
                elif self._sol_enabled():
                    print_rule("[bold cyan]Projector[/bold cyan]", log)
                    self._run_projector(state)
                    self._require_stage_outputs(STAGE_PROJECTOR, [self.sol_projection_path])
                    state.projector_done = True
                else:
                    print_message(
                        "[dim]projector skipped — `sol.enabled: false` in task.yaml[/dim]",
                        log,
                    )
                state.stage = STAGE_PROFILER
                self._checkpoint(state)

            # ---- outer round loop (fixed budget; deterministic breaks) ----
            while state.stage in ROUND_STAGES and state.round_index < state.max_rounds:
                round_no = state.round_index + 1
                print_rule(
                    f"[bold cyan]Optimization round {round_no}/{state.max_rounds}[/bold cyan]",
                    log,
                )

                if state.stage == STAGE_PROFILER:
                    if not (
                        state.reuse_pending or state.reanalyze_pending or self._replan_only(state)
                    ):
                        profile_dir = self._profile_dir(state)
                        profile_dir.mkdir(parents=True, exist_ok=True)
                        print_rule("[bold cyan]Profiler[/bold cyan]", log)
                        self._run_profiler(state)
                        self._require_profile_outputs(profile_dir)
                        state.last_profile_dir = str(profile_dir)
                        self._record_nsys_capture(state, profile_dir)
                    # Capture completion is durable before interpretation starts.
                    # Keep profile_required until the full analysis validates so
                    # a retry cannot accidentally replan from older findings.
                    state.stage = STAGE_ANALYZER
                    self._checkpoint(state)

                if state.stage == STAGE_ANALYZER:
                    analysis_dir = self._analysis_dir(state)
                    analysis_dir.mkdir(parents=True, exist_ok=True)
                    replan_only = self._replan_only(state)
                    if replan_only:
                        print_message(
                            f"[dim]round {round_no} opens replan-only — the "
                            f"previous round accepted nothing, so "
                            f"{state.last_profiled_analysis_dir} still describes "
                            f"this build[/dim]",
                            log,
                        )
                    if not (replan_only or state.reuse_pending):
                        self._require_profile_outputs(
                            Path(state.last_profile_dir),
                            imported=state.reanalyze_pending,
                        )
                    self._run_analyzer(state)
                    analyzer_outputs = [
                        self.roadmap_path,
                        analysis_dir / "analysis.md",
                        analysis_dir / performance_model.MODEL_FILENAME,
                    ]
                    enforce_ledger = self._kernel_coverage() is not None
                    if enforce_ledger:
                        analyzer_outputs.append(analysis_dir / kernel_ledger.LEDGER_FILENAME)
                    self._require_stage_outputs(STAGE_ANALYZER, analyzer_outputs)
                    roadmap = self._validate_roadmap()
                    model = self._validate_performance_model(analysis_dir)
                    if enforce_ledger:
                        self._validate_kernel_ledger(roadmap, analysis_dir, round_no=round_no)
                    self._validate_nsys_items(roadmap, analysis_dir)
                    if not state.reuse_pending:
                        self._record_analysis_source(
                            state, analysis_dir, mode="replan" if replan_only else "capture"
                        )
                    if not replan_only and not (state.reuse_pending or state.reanalyze_pending):
                        # This round's evidence now describes the current
                        # build; a replan round produced none and leaves the
                        # pointer on the analysis it planned from. The
                        # ``--reuse-analysis`` import turn is excluded for the
                        # same reason: the artifacts under ``analysis_dir``
                        # were profiled by *another* run against another
                        # checkout, so they are prior art, not evidence about
                        # this build. Leaving the pointer empty is what makes
                        # round 2 of a reuse campaign profile normally instead
                        # of replanning against a stranger's traces.
                        state.last_profiled_analysis_dir = str(analysis_dir)
                        state.profile_required = self._model_needs_measurement(model)
                    elif self._model_needs_measurement(model):
                        # Replans may discover a new capture need even when an
                        # existing optimization item remains actionable.
                        state.profile_required = True
                    state.reuse_pending = False
                    state.reanalyze_pending = False
                    noise_floor = float(self._optimize_block()["noise_floor_pct"])
                    items = roadmap_schema.top_pending_items(
                        roadmap,
                        state.max_items_per_round,
                        noise_floor,
                        self._allowed_approaches(),
                    )
                    if not items:
                        model_status = performance_model.convergence_status(
                            model, focus_concurrencies=self._focus_points()
                        )
                        state.round_index += 1
                        state.item_index = 0
                        if (
                            self._model_needs_measurement(model)
                            and state.round_index < state.max_rounds
                        ):
                            state.profile_required = True
                            state.stage = STAGE_PROFILER
                            print_message(
                                "[cyan]model requires new evidence; scheduling a targeted "
                                "profile from its next tests[/cyan]",
                                log,
                            )
                            self._checkpoint(state)
                            continue
                        budget_note = (
                            "; round budget exhausted"
                            if state.round_index >= state.max_rounds
                            else ""
                        )
                        self._conclude_round_loop(
                            state,
                            "roadmap has no actionable pending items; theoretical model status: "
                            + model_status
                            + budget_note,
                            log,
                        )
                        break
                    state.stage = STAGE_OPTIMIZER_EVALUATOR
                    parallel = state.item_execution == "parallel"
                    self._prepare_item_batch(state, items, eager_runtime=parallel)
                    if parallel:
                        for item in items:
                            roadmap_schema.mark_in_progress(self.roadmap_path, item["id"])
                    print_message(
                        f"[bold cyan]→ {state.item_execution} optimizer/evaluator batch: "
                        f"{', '.join(str(item['id']) for item in items)}[/bold cyan]",
                        log,
                    )

                if state.stage == STAGE_OPTIMIZER_EVALUATOR:
                    if state.item_execution == "serial":
                        self._run_opt_items_serial(state, log)
                        self._finish_item_batch(state, log)
                    else:
                        self._run_opt_items_parallel(state, log)
                        if any(
                            entry.get("status") == "candidate_ready" for entry in state.item_batch
                        ):
                            state.stage = STAGE_INTEGRATOR
                            self._checkpoint(state)
                        else:
                            self._finalize_failed_parallel_batch(state, log)

                if state.stage == STAGE_INTEGRATOR:
                    self._integrate_batch(state, log)

            # Defensive: a checkpoint parked inside the round ladder with
            # the round budget already spent falls through to the final
            # verification instead of dead-ending.
            if state.stage in ROUND_STAGES:
                state.stage = STAGE_QA
                self._checkpoint(state)

            # ---- one-shot: final verification ----
            if state.stage == STAGE_QA:
                if self._any_accepted_items():
                    print_rule("[bold cyan]QA (final verification)[/bold cyan]", log)
                    self.final_verification_dir.mkdir(parents=True, exist_ok=True)
                    clear_stale_benchmark_results(self.final_verification_dir)
                    self._run_qa(state)
                    self._require_stage_outputs(STAGE_QA, [self.verification_report_path])
                else:
                    print_message(
                        "[bold yellow]no accepted items — skipping the final "
                        "verification benchmark[/bold yellow]",
                        log,
                    )
                state.stage = STAGE_FINAL_ANALYZER if self._any_accepted_items() else STAGE_REPORTER
                self._checkpoint(state)

            # A pre-reconciliation reporter checkpoint must also acquire a
            # final model; the existing QA artifact remains the measurement.
            if state.stage in (STAGE_REPORTER, STAGE_FINAL_ANALYZER) and self._any_accepted_items():
                self._require_stage_outputs(STAGE_QA, [self.verification_report_path])
            if (
                state.stage == STAGE_REPORTER
                and self._any_accepted_items()
                and self.verification_report_path.is_file()
                and not self._is_nonempty(
                    self.final_analysis_dir / performance_model.MODEL_FILENAME
                )
            ):
                state.stage = STAGE_FINAL_ANALYZER
                self._checkpoint(state)

            if state.stage == STAGE_FINAL_ANALYZER:
                print_rule("[bold cyan]Analyzer (final model reconciliation)[/bold cyan]", log)
                self.final_analysis_dir.mkdir(parents=True, exist_ok=True)
                self._run_final_analyzer(state)
                self._require_stage_outputs(
                    STAGE_FINAL_ANALYZER,
                    [
                        self.final_analysis_dir / "analysis.md",
                        self.final_analysis_dir / performance_model.MODEL_FILENAME,
                    ],
                )
                self._validate_performance_model(self.final_analysis_dir)
                self._record_analysis_source(
                    state, self.final_analysis_dir, mode="final_reconciliation"
                )
                state.stage = STAGE_REPORTER
                self._checkpoint(state)

            # ---- one-shot: report ----
            if state.stage == STAGE_REPORTER:
                print_rule("[bold cyan]Reporter[/bold cyan]", log)
                model_path = self._report_performance_model()
                if model_path is None:
                    raise RuntimeError("reporter requires a validated performance_model.yaml")
                self._validate_performance_model(model_path.parent)
                self._run_reporter(state)
                self._require_stage_outputs(
                    STAGE_REPORTER, [self.report_path, self.report_html_path]
                )
                state.reporter_done = True
                state.done = True
                self._checkpoint(state)
                print_message(
                    f"[bold green]✔ optimization report written to {self.report_path}[/bold green]",
                    log,
                )
        except KeyboardInterrupt:
            print_message(
                "[bold yellow]⚠ interrupted — run again to continue from "
                "the last checkpoint, or pass --clean to start fresh"
                "[/bold yellow]",
                log,
            )
            raise
        except Exception as exc:
            # Record the abort in the session log — otherwise the log just
            # ends after the last agent's turn and the crash (which only
            # reaches stderr) looks like a silent, deliberate exit.
            print_message(
                f"[bold red]✗ workflow aborted at stage '{state.stage}' "
                f"(round {state.round_index + 1}, attempt "
                f"{state.attempt_index + 1}): "
                f"{escape(f'{type(exc).__name__}: {exc}')}\n"
                f"Run again to continue from the last checkpoint, or pass "
                f"--clean to start fresh.[/bold red]",
                log,
            )
            raise

    # ------------------------------------------------------------ state & setup

    def _init_state(self, task: str, log) -> WorkflowState | None:
        """Load or create the workflow state; return ``None`` to no-op.

        ``task`` is the path to the input ``task.yaml``. On a fresh run it
        is validated and the normalized spec is written verbatim into
        ``workspace/task.yaml``, and the live tuning config is
        materialized; on resume the checkpointed ``workspace/task.yaml``
        is the source of truth.
        """
        if self.resume:
            state = load_state(self.state_path)
            if state.done:
                print_message(
                    "[bold green]✔ workflow already completed; pass --clean to rerun[/bold green]",
                    log,
                )
                return None
            if (
                self.max_rounds_override is not None
                and self.max_rounds_override != state.max_rounds
            ):
                print_message(
                    f"[bold yellow]⚠ --max-rounds {self.max_rounds_override} ignored on "
                    f"resume; the checkpointed budget ({state.max_rounds}) wins. Pass "
                    f"--clean to start fresh with the new budget.[/bold yellow]",
                    log,
                )
            if self.reanalyze:
                print_message(
                    "[bold yellow]⚠ --reanalyze ignored on resume; the checkpointed "
                    "analysis mode wins. Pass --clean with --reuse-analysis to "
                    "start a new re-analysis campaign.[/bold yellow]",
                    log,
                )
            if self.reuse_analysis is not None:
                print_message(
                    f"[bold yellow]⚠ --reuse-analysis {self.reuse_analysis} ignored on "
                    f"resume; the import is a fresh-run seeding step and this "
                    f"workspace already has a checkpoint. Pass --clean to re-seed "
                    f"from it.[/bold yellow]",
                    log,
                )
            return state

        # Fresh run: validate + normalize the spec and materialize it into
        # the workspace so the agents read a fully-resolved task.yaml.
        task_data = load_and_validate_task_yaml(
            task,
            max_rounds_override=self.max_rounds_override,
        )
        self.task_path.write_text(dump_task_yaml(task_data), encoding="utf-8")
        # Materialize the live tuning config (the single
        # --extra_llm_api_options every serve in this workflow uses) and
        # its last-accepted snapshot. In a disagg campaign the same file
        # holds the harness config's ctx / gen worker_config instead, so
        # the optimizer still edits exactly one file and the diff /
        # revert / accepted-snapshot machinery applies unchanged.
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        disagg_config = disagg_config_path(task_data)
        extra = task_data.get("extra_llm_api_options")
        if disagg_config is not None:
            self.tuning_config_path.write_text(
                worker_config_yaml(load_disagg_config(disagg_config)), encoding="utf-8"
            )
        elif extra:
            shutil.copyfile(extra, self.tuning_config_path)
        else:
            self.tuning_config_path.write_text("{}\n", encoding="utf-8")
        shutil.copyfile(self.tuning_config_path, self.tuning_accepted_path)

        optimize = task_data["optimize"]
        state = WorkflowState(
            task_path=str(self.task_path),
            max_rounds=int(optimize["max_rounds"]),
            max_attempts_per_item=int(optimize["max_attempts_per_item"]),
            max_items_per_round=int(optimize["max_items_per_round"]),
            item_execution=str(optimize["item_execution"]),
            stage=STAGE_BENCHMARKER,
        )
        if self.reuse_analysis is not None:
            self._seed_from_reuse(state, log)
        # Checkpoint before running the first stage so a crash mid-stage
        # can be picked up on the next run.
        self._checkpoint(state)
        return state

    def _seed_from_reuse(self, state: WorkflowState, log) -> None:
        """Import a previous run's analysis and skip the stages it covers.

        Copies the ``--reuse-analysis`` source's baseline report (+ result
        JSONs), SOL projection (+ ``sol_work/``) and newest profile
        findings (+ traces, kernel ledger) into this workspace's canonical
        paths, then marks the stages those artifacts stand in for as done:
        an imported baseline skips the benchmarker, an imported projection
        skips the projector, and imported findings put round 1's analyzer
        in plan-only mode (``reuse_pending``). Whatever the source lacks is
        simply produced normally.
        """
        source = self.reuse_analysis
        if source is None:
            return
        if self.workspace.resolve() == source.resolve():
            raise reuse.ReuseError(
                f"--reuse-analysis source is this run's own workspace ({source}); "
                f"point it at the previous run's workspace instead."
            )
        discovered = reuse.discover(source, reanalyze=self.reanalyze)
        imported = reuse.import_analysis(
            discovered,
            workspace=self.workspace,
            baseline_dir=self.baseline_dir,
            analysis_dir=self.rounds_dir / "round_1" / "analysis",
            profile_dir=self.rounds_dir / "round_1" / "profile",
            reanalyze=self.reanalyze,
            sol_projection_path=self.sol_projection_path,
            sol_work_dir=self.sol_work_dir,
            reuse_dir=self.reuse_dir,
        )
        state.reuse_analysis_dir = str(source)
        if imported.baseline_report:
            try:
                self._require_baseline_measurement()
            except RuntimeError as exc:
                print_message(
                    f"[yellow]imported baseline cannot be used; measuring it again: "
                    f"{escape(str(exc))}[/yellow]",
                    log,
                )
            else:
                state.benchmarker_done = True
                state.stage = STAGE_PROJECTOR
        if imported.sol_projection:
            # Only meaningful when the task enables the stage at all; the
            # flag is what makes the projector gate skip it, and a task
            # with ``sol.enabled: false`` skips it regardless.
            state.projector_done = self._sol_enabled()
        if imported.profile:
            state.last_profile_dir = str(self.rounds_dir / "round_1" / "profile")
            self._record_nsys_capture(state, Path(state.last_profile_dir))
        if self.reanalyze:
            state.reanalyze_pending = True
        elif imported.findings:
            state.reuse_pending = True
            if not imported.profile:
                self._record_nsys_capture(state, self.rounds_dir / "round_1" / "analysis")
        print_message(
            f"[bold cyan]reusing analysis from {source}: "
            f"{imported.summary()} (manifest: {imported.manifest_path})[/bold cyan]",
            log,
        )
        if not imported.baseline_report:
            print_message(
                "[yellow]no baseline report in the reuse source — measuring the "
                "baseline normally[/yellow]",
                log,
            )
        if not imported.findings and not self.reanalyze:
            print_message(
                "[yellow]no profile findings in the reuse source — round 1 will "
                "profile normally[/yellow]",
                log,
            )

    def _ensure_optimization_branch(self, state: WorkflowState, log) -> None:
        """Create (fresh run) or check out (resume) the optimization branch.

        Persist branch creation intent first, then reconcile it with git.
        A resumed creation uses the recorded base even if HEAD has moved.
        """
        repo = self._trtllm_repo_path()
        if not repo:
            raise RuntimeError(
                f"trtllm_repo_path missing from {self.task_path}; cannot manage the "
                f"optimization branch."
            )
        if not gitops.is_git_repo(repo):
            raise RuntimeError(
                f"trtllm_repo_path ({repo}) is not a git repository. perf-optimize "
                f"needs git to commit accepted optimizations and revert rejected "
                f"ones — clone the checkout with git and retry."
            )
        if not gitops.worktree_clean(repo):
            raise RuntimeError(
                f"trtllm_repo_path ({repo}) has uncommitted changes. Commit or stash "
                "them before starting or resuming perf-optimize so the measured "
                "campaign and candidate worktrees share a committed source snapshot."
            )
        if state.campaign_git_branch:
            if gitops.branch_exists(repo, state.campaign_git_branch):
                gitops.checkout(repo, state.campaign_git_branch)
            elif state.campaign_git_base_commit:
                gitops.create_branch(
                    repo, state.campaign_git_branch, state.campaign_git_base_commit
                )
            else:
                raise RuntimeError(
                    f"Campaign branch {state.campaign_git_branch!r} is missing and "
                    "the checkpoint records no base commit to recover it from."
                )
            return
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        state.campaign_git_branch = f"perf-optimize/{self.workspace.name}-{timestamp}"
        state.campaign_git_base_commit = gitops.rev_parse_head(repo)
        self._checkpoint(state)
        gitops.create_branch(repo, state.campaign_git_branch)
        print_message(
            f"[bold cyan]optimizing on branch {state.campaign_git_branch} "
            f"(base {state.campaign_git_base_commit[:12]})[/bold cyan]",
            log,
        )

    def _checkpoint(self, state: WorkflowState) -> None:
        with self._checkpoint_lock:
            save_state(self.state_path, state)

    def _update_batch_item(
        self, state: WorkflowState, item_id: str, **updates: Any
    ) -> dict[str, Any]:
        """Update one batch row and checkpoint it as one locked operation."""
        with self._checkpoint_lock:
            for entry in state.item_batch:
                if entry["current_item_id"] == item_id:
                    entry.update(updates)
                    save_state(self.state_path, state)
                    return dict(entry)
        raise RuntimeError(f"item batch has no item {item_id!r}")

    def _remove_worktree_best_effort(self, repo: str, path: str, log) -> None:
        """Retry a failed worktree cleanup once without aborting the campaign."""
        try:
            gitops.remove_worktree(repo, path)
            return
        except (gitops.GitOpsError, OSError):
            time.sleep(1)

        try:
            gitops.remove_worktree(repo, path)
        except (gitops.GitOpsError, OSError) as exc:
            print_message(
                "[yellow]⚠ worktree cleanup failed after one retry; "
                f"leaving it in place and continuing: {escape(path)}\n"
                f"{escape(str(exc))}[/yellow]",
                log,
            )

    def _prepare_item_batch(
        self,
        state: WorkflowState,
        items: list[dict[str, Any]],
        *,
        eager_runtime: bool = True,
    ) -> None:
        """Checkpoint a batch and optionally create every item runtime eagerly."""
        repo = self._trtllm_repo_path()
        base = gitops.rev_parse_head(repo)
        round_no = state.round_index + 1
        batch: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            item_id = str(item["id"])
            safe = re.sub(r"[^A-Za-z0-9._-]+", "-", item_id).strip("-.")[:48] or "item"
            worktree = self.worktrees_dir / f"round_{round_no}" / f"item_{index + 1}_{safe}"
            branch = f"{state.campaign_git_branch}-round-{round_no}-item-{index + 1}-{safe}"
            batch.append(
                {
                    "current_item_id": item_id,
                    "item_index": index,
                    "attempt_index": 0,
                    "approach_violation": "",
                    "item_worktree_path": str(worktree),
                    "item_branch": branch,
                    "item_base_commit": base,
                    "phase": STAGE_OPTIMIZER,
                    "status": "pending",
                    "candidate_commit": "",
                    "candidate_config_path": "",
                    "last_error": "",
                    "finalized": False,
                }
            )
        state.item_batch = batch
        state.batch_started = False
        state.batch_completed = False
        self._checkpoint(state)
        if eager_runtime:
            for entry in state.item_batch:
                self._ensure_item_runtime(state, entry)

    def _ensure_item_runtime(self, state: WorkflowState, entry: dict[str, Any]) -> None:
        """Create any missing worktree/config/progress parts for a batch row."""
        if "reference_measurement" not in entry:
            self._update_batch_item(
                state,
                str(entry["current_item_id"]),
                reference_measurement=roadmap_schema.load_roadmap(self.roadmap_path)[
                    "current_best"
                ],
            )
        worktree = Path(str(entry["item_worktree_path"]))
        if not worktree.exists():
            gitops.create_worktree(
                self._trtllm_repo_path(),
                worktree,
                str(entry["item_branch"]),
                str(entry["item_base_commit"]),
            )
        local_state = self._local_item_state(state, entry)
        item_dir = self._item_dir(local_state)
        tuning_dir = item_dir / "tuning"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        live = tuning_dir / "extra_llm_api_options.yaml"
        accepted = tuning_dir / "extra_llm_api_options.accepted.yaml"
        if not live.exists():
            shutil.copyfile(self.tuning_accepted_path, live)
        if not accepted.exists():
            shutil.copyfile(self.tuning_accepted_path, accepted)
        progress_path = item_dir / "progress.yaml"
        if not progress_path.exists():
            init_progress_file(progress_path)

    def _local_item_state(self, state: WorkflowState, entry: dict[str, Any]) -> WorkflowState:
        """Build a worker-local state view for the existing item helpers."""
        return replace(
            state,
            current_item_id=str(entry["current_item_id"]),
            item_index=int(entry["item_index"]),
            attempt_index=int(entry.get("attempt_index", 0)),
            approach_violation=str(entry.get("approach_violation", "")),
            item_worktree_path=str(entry["item_worktree_path"]),
            item_branch=str(entry["item_branch"]),
            item_base_commit=str(entry["item_base_commit"]),
            item_batch=[],
            stage=str(entry.get("phase", STAGE_OPTIMIZER)),
        )

    def _run_opt_items_parallel(self, state: WorkflowState, log) -> None:
        for entry in state.item_batch:
            self._ensure_item_runtime(state, entry)
        item_ids = [str(entry["current_item_id"]) for entry in state.item_batch]
        if not state.batch_started:
            append_workflow_event(
                self.progress_path,
                self._progress_lock,
                event="batch_started",
                round_no=state.round_index + 1,
                item_ids=item_ids,
                summary=f"Started {len(item_ids)} parallel optimizer/evaluator item loops.",
            )
            state.batch_started = True
            self._checkpoint(state)

        # Every item has its own worktree and agent sessions. Local agents
        # coordinate shared runtime operations through their flock instructions;
        # reasoning and source edits can overlap regardless of execution environment.
        errors: list[BaseException] = []
        with ThreadPoolExecutor(
            max_workers=max(1, len(state.item_batch)),
            thread_name_prefix="perf-opt-item",
        ) as executor:
            futures = {
                executor.submit(self._run_opt_item, state, dict(entry), log): entry
                for entry in state.item_batch
                if entry.get("status") not in ("candidate_ready", "failed")
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:  # preserve other completed item results
                    errors.append(exc)
        if errors:
            raise RuntimeError(
                f"{len(errors)} parallel optimization item(s) failed; "
                f"first error: {type(errors[0]).__name__}: {errors[0]}"
            ) from errors[0]

        if not state.batch_completed:
            append_workflow_event(
                self.progress_path,
                self._progress_lock,
                event="batch_completed",
                round_no=state.round_index + 1,
                item_ids=item_ids,
                summary="All parallel optimizer/evaluator item loops reached a terminal state.",
            )
            state.batch_completed = True
            self._checkpoint(state)

    def _run_opt_items_serial(self, state: WorkflowState, log) -> None:
        """Run the preselected batch one item at a time, accepting directly."""
        repo = self._trtllm_repo_path()
        for entry in state.item_batch:
            item_id = str(entry["current_item_id"])
            if entry.get("finalized"):
                continue

            status = str(entry.get("status", "pending"))
            if status not in ("candidate_ready", "failed"):
                if status == "pending":
                    # The previous serial item may have advanced the campaign.
                    # Delay freezing this item's base/config until it actually runs.
                    self._update_batch_item(
                        state,
                        item_id,
                        item_base_commit=gitops.rev_parse_head(repo),
                    )
                    roadmap_schema.mark_in_progress(self.roadmap_path, item_id)
                self._ensure_item_runtime(state, entry)
                self._run_opt_item(state, dict(entry), log)

            self._finalize_serial_item(state, item_id, log)
            met, _ = self._target_met()
            if met:
                break

    def _finalize_serial_item(self, state: WorkflowState, item_id: str, log) -> None:
        """Promote one terminal serial worker result into the campaign state."""
        entry = next(
            (row for row in state.item_batch if row["current_item_id"] == item_id),
            None,
        )
        if entry is None:
            raise RuntimeError(f"serial batch has no item {item_id!r}")
        if entry.get("finalized"):
            return

        status = str(entry.get("status", ""))
        if status not in ("candidate_ready", "failed"):
            raise RuntimeError(
                f"serial item {item_id!r} reached finalization with status {status!r}"
            )

        attempts = int(entry.get("attempts", 0))
        gain = entry.get("measured_gain_pct")
        repo = self._trtllm_repo_path()
        if status == "candidate_ready":
            self._validate_evaluator_approval(
                entry, item_id, reference=entry.get("reference_measurement")
            )
            # Persist the stale-profile decision before mutating the accepted
            # campaign state so a crash cannot resume into replan-only mode.
            state.profile_required = True
            self._checkpoint(state)
            if entry.get("candidate_commit"):
                gitops.fast_forward(repo, str(entry["item_branch"]))
            candidate_config = Path(str(entry["candidate_config_path"]))
            shutil.copyfile(candidate_config, self.tuning_config_path)
            shutil.copyfile(candidate_config, self.tuning_accepted_path)
            roadmap_schema.apply_evaluation(
                self.roadmap_path,
                item_id,
                status="accepted",
                attempts=attempts,
                measured_gain_pct=gain,
                gap_implication=entry.get("gap_implication"),
            )
            value = entry.get("measured_value")
            local_state = self._local_item_state(state, entry)
            if value is not None:
                curve = entry.get("curve") if isinstance(entry.get("curve"), list) else None
                roadmap_schema.set_current_best(
                    self.roadmap_path,
                    float(value),
                    str(
                        (self._attempt_dir(local_state) / "evaluation.md").relative_to(
                            self.workspace
                        )
                    ),
                    curve=curve,
                )
            self._record_nsys_capture(
                state, self._attempt_dir(local_state) / "profile", invalidate=True
            )
            final_status = "accepted"
            print_message(
                f"[bold green]✔ serial evaluator APPROVE — {item_id} accepted[/bold green]",
                log,
            )
        else:
            roadmap_schema.apply_evaluation(
                self.roadmap_path,
                item_id,
                status="failed",
                attempts=attempts,
                measured_gain_pct=gain,
                gap_implication=entry.get("gap_implication"),
            )
            final_status = "failed"

        self._update_batch_item(
            state,
            item_id,
            status=final_status,
            phase="complete",
            finalized=True,
        )
        worktree = str(entry.get("item_worktree_path", ""))
        if worktree and Path(worktree).exists():
            self._remove_worktree_best_effort(repo, worktree, log)

    def _run_opt_item(self, state: WorkflowState, entry: dict[str, Any], log) -> None:
        """Run one existing optimizer/evaluator attempt loop in isolation."""
        item_state = self._local_item_state(state, entry)
        item_id = item_state.current_item_id
        progress_path = self._item_progress_path(item_state)
        progress_ctx = ProgressContext(
            path=progress_path,
            global_path=self.progress_path,
            global_lock=self._progress_lock,
        )
        tools = build_progress_tools(progress_ctx)
        optimizer = _make_agent(
            f"optimizer-{item_id}",
            self.prompts.optimizer,
            tools["optimizer"],
            required_tools=["append_optimizer_progress"],
            cwd=Path(item_state.item_worktree_path),
        )
        evaluator = _make_agent(
            f"evaluator-{item_id}",
            self.prompts.evaluator,
            tools["evaluator"],
            required_tools=["append_evaluator_progress"],
            session_mode="stateless",
            cwd=Path(item_state.item_worktree_path),
        )
        repo = item_state.item_worktree_path
        live_config, accepted_config = self._state_tuning_paths(item_state)
        try:
            while item_state.attempt_index < item_state.max_attempts_per_item:
                attempt_no = item_state.attempt_index + 1
                item_state.stage = str(entry.get("phase", STAGE_OPTIMIZER))
                self._attempt_dir(item_state).mkdir(parents=True, exist_ok=True)

                if item_state.stage == STAGE_OPTIMIZER:
                    gitops.reset_to(repo, item_state.item_base_commit)
                    shutil.copyfile(accepted_config, live_config)
                    self._update_batch_item(
                        state,
                        item_id,
                        phase=STAGE_OPTIMIZER,
                        attempt_index=item_state.attempt_index,
                        status="running",
                    )
                    self._run_optimizer(item_state, agent=optimizer, progress_ctx=progress_ctx)
                    self._require_stage_outputs(
                        STAGE_OPTIMIZER,
                        [self._attempt_dir(item_state) / "optimization_summary.md"],
                    )
                    violation = self._detect_approach_violation(item_state)
                    if violation is not None:
                        gitops.reset_to(repo, item_state.item_base_commit)
                        shutil.copyfile(accepted_config, live_config)
                        if attempt_no >= item_state.max_attempts_per_item:
                            self._update_batch_item(
                                state,
                                item_id,
                                status="failed",
                                phase="complete",
                                attempts=attempt_no,
                                approach_violation=violation,
                            )
                            return
                        item_state.attempt_index += 1
                        item_state.approach_violation = violation
                        entry["phase"] = STAGE_OPTIMIZER
                        self._update_batch_item(
                            state,
                            item_id,
                            phase=STAGE_OPTIMIZER,
                            attempt_index=item_state.attempt_index,
                            approach_violation=violation,
                        )
                        continue
                    item_state.approach_violation = ""
                    entry["phase"] = STAGE_EVALUATOR
                    self._update_batch_item(
                        state,
                        item_id,
                        phase=STAGE_EVALUATOR,
                        approach_violation="",
                    )

                item_state.stage = STAGE_EVALUATOR
                cached_verdict = latest_entry(progress_path, "evaluator")
                cached_attempt = (
                    cached_verdict.get("attempt") if cached_verdict is not None else None
                )
                evaluation_path = self._attempt_dir(item_state) / "evaluation.md"
                reuse_verdict = self._is_nonempty(evaluation_path) and cached_attempt == attempt_no
                validation_feedback = ""
                if reuse_verdict and cached_verdict.get("decision") == "APPROVE":
                    try:
                        self._validate_evaluator_approval(
                            cached_verdict,
                            item_id,
                            reference=entry.get("reference_measurement"),
                        )
                    except RuntimeError as exc:
                        reuse_verdict = False
                        validation_feedback = str(exc)
                if not reuse_verdict:
                    feedback = {}
                    if validation_feedback:
                        # Correct a cached verdict against the existing evidence;
                        # preserve both the candidate and the measured results.
                        feedback["validation_feedback"] = validation_feedback
                    else:
                        clear_stale_benchmark_results(self._attempt_dir(item_state))
                    self._run_evaluator(
                        item_state, agent=evaluator, progress_ctx=progress_ctx, **feedback
                    )
                    self._require_stage_outputs(
                        STAGE_EVALUATOR,
                        [evaluation_path],
                    )
                decision = self._latest_evaluator_decision(progress_path)
                gain = self._latest_evaluator_measured_gain(progress_path)
                value = self._latest_evaluator_measured_value(progress_path)
                curve = self._latest_evaluator_curve(progress_path)
                # The experiment evidence: what this verdict proved,
                # against which parts, spending which lever. Carried on
                # the batch row because `_finish_item_batch` is the one
                # choke point every path converges on, and by then the
                # per-item progress files are out of scope.
                ledger_fields = self._latest_evaluator_ledger_fields(progress_path)
                ledger_fields["evaluation_path"] = str(
                    evaluation_path.relative_to(self.workspace)
                    if evaluation_path.is_relative_to(self.workspace)
                    else evaluation_path
                )
                if decision == "APPROVE":
                    self._validate_evaluator_approval(
                        latest_entry(progress_path, "evaluator"),
                        item_id,
                        reference=entry.get("reference_measurement"),
                    )
                    if not gitops.worktree_clean(repo):
                        commit = gitops.commit_all(
                            repo,
                            f"perf-optimize: {item_id} candidate-ready "
                            f"[round {item_state.round_index + 1}]",
                        )
                    else:
                        # A crash can occur after git commit but before the
                        # candidate-ready checkpoint. Recover the same commit
                        # from git when reusing that attempt's cached approval.
                        commit = gitops.rev_parse_head(repo)
                    if commit == item_state.item_base_commit:
                        commit = ""  # A config-only candidate has no source commit.
                    self._update_batch_item(
                        state,
                        item_id,
                        status="candidate_ready",
                        phase="complete",
                        last_error="",
                        attempts=attempt_no,
                        candidate_commit=commit,
                        candidate_config_path=str(live_config),
                        measured_gain_pct=gain,
                        measured_value=value,
                        curve=curve,
                        **ledger_fields,
                    )
                    return
                if decision == "REJECT" or attempt_no >= item_state.max_attempts_per_item:
                    gitops.reset_to(repo, item_state.item_base_commit)
                    shutil.copyfile(accepted_config, live_config)
                    self._update_batch_item(
                        state,
                        item_id,
                        status="failed",
                        phase="complete",
                        attempts=attempt_no,
                        measured_gain_pct=gain,
                        **ledger_fields,
                    )
                    return

                gitops.reset_to(repo, item_state.item_base_commit)
                shutil.copyfile(accepted_config, live_config)
                item_state.attempt_index += 1
                entry["phase"] = STAGE_OPTIMIZER
                self._update_batch_item(
                    state,
                    item_id,
                    phase=STAGE_OPTIMIZER,
                    attempt_index=item_state.attempt_index,
                    status="running",
                )
        except BaseException as exc:
            self._update_batch_item(
                state,
                item_id,
                status="error",
                phase=item_state.stage,
                attempt_index=item_state.attempt_index,
                last_error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            optimizer.__exit__(None, None, None)
            evaluator.__exit__(None, None, None)

    def _finalize_failed_parallel_batch(self, state: WorkflowState, log) -> None:
        """Record an all-failed parallel batch and close its round."""
        for entry in state.item_batch:
            roadmap_schema.apply_evaluation(
                self.roadmap_path,
                str(entry["current_item_id"]),
                status="failed",
                attempts=int(entry.get("attempts", 0)),
                measured_gain_pct=entry.get("measured_gain_pct"),
                gap_implication=entry.get("gap_implication"),
            )
        self._finish_item_batch(state, log)

    def _validate_evaluator_approval(
        self,
        verdict: dict[str, Any],
        item_id: str,
        *,
        reference: dict[str, Any] | None = None,
    ) -> None:
        """Check every candidate before a commit or campaign promotion."""
        roadmap = roadmap_schema.load_roadmap(self.roadmap_path)
        item = roadmap_schema.find_item(roadmap, item_id)
        optimize = self._optimize_block()
        required = max(
            float(optimize["noise_floor_pct"]),
            float(optimize["accept_fraction"]) * float(item["expected_gain_pct"]),
        )
        self._validate_acceptance_measurement(
            verdict,
            reference if reference is not None else roadmap["current_best"],
            required,
            f"evaluator {item_id} APPROVE",
        )

    def _validate_acceptance_measurement(
        self,
        verdict: dict[str, Any],
        reference: dict[str, Any],
        required_gain: float,
        role: str,
    ) -> None:
        optimize = self._optimize_block()
        regression_budget = self._regression_budget()
        try:
            measurements.validate_acceptance(
                verdict,
                reference,
                metric=str(optimize["target_metric"]),
                required_gain=required_gain,
                points=self._curve_points() if self._curve_mode() else None,
                focus=self._focus_points(),
                allowed_regression=(
                    float(optimize["noise_floor_pct"])
                    if regression_budget is None
                    else regression_budget
                ),
            )
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"{role}: {exc}") from exc

    def _validate_integrator_verdict(
        self,
        verdict: dict[str, Any],
        candidates: list[dict[str, Any]],
        reference: dict[str, Any],
    ) -> None:
        decision = verdict.get("decision")
        if decision not in INTEGRATOR_DECISIONS:
            raise RuntimeError("integrator finished without a structured verdict")
        candidate_ids = {str(entry["current_item_id"]) for entry in candidates}
        included = {str(item_id) for item_id in verdict.get("included_item_ids", [])}
        unknown = included - candidate_ids
        if unknown:
            raise RuntimeError(
                "integrator included non-candidate item(s): " + ", ".join(sorted(unknown))
            )
        if decision == "REJECT":
            if included:
                raise RuntimeError("integrator REJECT verdict must include no candidates")
            return
        if not included:
            raise RuntimeError(f"integrator {decision} verdict included no candidates")
        noise_floor = float(self._optimize_block()["noise_floor_pct"])
        best = max(candidates, key=lambda entry: float(entry["measured_gain_pct"]))
        expected_required_gain = max(noise_floor, float(best["measured_gain_pct"]) - noise_floor)
        try:
            reported_required_gain = measurements.finite_number(
                verdict.get("required_gain_pct"), "required_gain_pct"
            )
        except ValueError as exc:
            raise RuntimeError(f"integrator: {exc}") from exc
        if abs(reported_required_gain - expected_required_gain) > 1e-6:
            raise RuntimeError(
                "integrator required_gain_pct mismatch: "
                f"reported {reported_required_gain}, expected {expected_required_gain}"
            )
        self._validate_acceptance_measurement(
            verdict, reference, expected_required_gain, f"integrator {decision}"
        )
        if decision == "FALLBACK_BEST" and (
            verdict.get("best_candidate_id") != best["current_item_id"]
            or included != {str(best["current_item_id"])}
        ):
            raise RuntimeError(
                "integrator FALLBACK_BEST must include exactly the highest-gain candidate "
                "as its best_candidate_id (manifest order breaks ties)"
            )

    def _integrate_batch(self, state: WorkflowState, log) -> None:
        """Combine candidate-ready items and validate the Integrator's verdict."""
        candidates = [
            dict(entry) for entry in state.item_batch if entry.get("status") == "candidate_ready"
        ]
        if not candidates:
            raise RuntimeError("integrator stage has no candidate-ready items")

        repo = self._trtllm_repo_path()
        round_no = state.round_index + 1
        integration_dir = self._round_dir(state) / "integration"
        integration_dir.mkdir(parents=True, exist_ok=True)
        worktree = self.worktrees_dir / f"round_{round_no}" / "integration"
        if not state.integration_worktree_path:
            state.integration_worktree_path = str(worktree)
            state.integration_branch = f"{state.campaign_git_branch}-round-{round_no}-integration"
            self._checkpoint(state)
        if not Path(state.integration_worktree_path).exists():
            gitops.create_worktree(
                repo,
                state.integration_worktree_path,
                state.integration_branch,
                gitops.rev_parse_head(repo),
            )

        integration_tuning_dir = integration_dir / "tuning"
        integration_tuning_dir.mkdir(parents=True, exist_ok=True)
        integration_config = integration_tuning_dir / "extra_llm_api_options.yaml"
        if not integration_config.exists():
            shutil.copyfile(self.tuning_accepted_path, integration_config)
        manifest_path = integration_dir / "candidate_manifest.yaml"
        if not manifest_path.exists():
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "round": round_no,
                        "campaign_base_commit": gitops.rev_parse_head(repo),
                        "candidates": candidates,
                    },
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
        reference_path = integration_dir / "reference_measurement.yaml"
        if not reference_path.exists():
            reference_path.write_text(
                yaml.safe_dump(roadmap_schema.load_roadmap(self.roadmap_path)["current_best"]),
                encoding="utf-8",
            )
        reference = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
        reference_results = Path(
            str(reference.get("source", "baseline/benchmark_results.md"))
        ).parent
        if not reference_results.is_absolute():
            reference_results = self.workspace / reference_results
        report_path = integration_dir / "integration.md"
        verdict = latest_entry(self.progress_path, "integrator")
        cached_verdict = (
            self._is_nonempty(report_path)
            and verdict is not None
            and verdict.get("round") == round_no
        )
        validation_feedback = ""
        if cached_verdict:
            try:
                self._validate_integrator_verdict(verdict, candidates, reference)
            except RuntimeError as exc:
                cached_verdict = False
                validation_feedback = (
                    f"Your previous verdict failed deterministic validation: {exc}. "
                    "Correct the evidence/verdict and append a new final progress entry. "
                    "Inspect the existing integration state before applying commits again.\n\n"
                )
        if not cached_verdict:
            clear_stale_benchmark_results(integration_dir)
            self._stamp_progress(state, round_no=round_no)
            self.integrator(
                self._disagg_directive() + validation_feedback + f"Workspace: {self.workspace}\n"
                f"Round: {round_no}\n"
                f"Integration worktree: {state.integration_worktree_path}\n"
                f"Integration branch: {state.integration_branch}\n"
                f"Active runtime checkout: `{state.integration_worktree_path}`\n"
                f"Candidate manifest: {manifest_path}\n"
                f"Live integration config: {integration_config}\n"
                f"Campaign accepted config (base): {self.tuning_accepted_path}\n"
                f"Task: {self.task_path}\n"
                f"Roadmap: {self.roadmap_path}\n"
                f"Frozen reference measurement: {reference_path}\n"
                f"Reference benchmark results: {reference_results}\n"
                f"Write the integration report to: {report_path}\n\n"
                f"Inside the Slurm job script, before any Python command or "
                f"`trtllm-serve` launch:\n\n"
                f'`export PYTHONPATH="{state.integration_worktree_path}'
                f'${{PYTHONPATH:+:$PYTHONPATH}}"`\n\n'
                f"Read the manifest in order. Cherry-pick every non-empty "
                f"candidate_commit into the integration branch, resolve only "
                f"merge conflicts/minimal combination defects, and combine the "
                f"candidate config files into the live integration config. Commit "
                f"any conflict-resolution code before finishing.\n\n"
                f"Launch and benchmark this combined state using "
                f"`--extra_llm_api_options {integration_config}` and the same "
                f"Evaluator measurement/Pareto rules. Compute the combined "
                f"required gain as `max(noise_floor_pct, best standalone "
                f"measured_gain_pct - noise_floor_pct)` from task.yaml and the "
                f"manifest. Report that threshold exactly; the Python "
                f"orchestrator cross-checks it and the measured gain before "
                f"applying your verdict. You may diagnose/remediate at most "
                f"twice. If the combined "
                f"state still fails, retain and validate only the highest standalone "
                f"gain candidate (manifest order breaks ties); if that also fails, "
                f"restore the base and REJECT.\n\n"
                f"Call `append_integrator_progress` exactly once with the final "
                f"APPROVE, FALLBACK_BEST, or REJECT decision and all required "
                f"fields. Leave precisely the accepted code/config state in the "
                f"integration worktree and integration config."
            )
        self._require_stage_outputs(STAGE_INTEGRATOR, [report_path])
        verdict = latest_entry(self.progress_path, "integrator")
        if (
            verdict is None
            or verdict.get("round") != round_no
            or verdict.get("decision") not in INTEGRATOR_DECISIONS
        ):
            raise RuntimeError("integrator finished without a structured verdict")

        self._validate_integrator_verdict(verdict, candidates, reference)
        decision = str(verdict["decision"])
        included = {str(item_id) for item_id in verdict.get("included_item_ids", [])}
        if decision != "REJECT":
            # Persist the stale-profile decision before mutating the accepted
            # campaign state so a crash cannot resume into replan-only mode.
            state.profile_required = True
            self._checkpoint(state)
            if not gitops.worktree_clean(state.integration_worktree_path):
                gitops.commit_all(
                    state.integration_worktree_path,
                    f"perf-optimize: integrate round {round_no} candidates",
                )
            gitops.fast_forward(repo, state.integration_branch)
            shutil.copyfile(integration_config, self.tuning_config_path)
            shutil.copyfile(integration_config, self.tuning_accepted_path)

        for entry in state.item_batch:
            item_id = str(entry["current_item_id"])
            accepted = decision != "REJECT" and item_id in included
            roadmap_schema.apply_evaluation(
                self.roadmap_path,
                item_id,
                status="accepted" if accepted else "failed",
                attempts=int(entry.get("attempts", 0)),
                measured_gain_pct=entry.get("measured_gain_pct"),
                gap_implication=entry.get("gap_implication"),
            )
        if decision != "REJECT":
            curve = verdict.get("curve") if isinstance(verdict.get("curve"), list) else None
            roadmap_schema.set_current_best(
                self.roadmap_path,
                float(verdict["measured_value"]),
                str(report_path.relative_to(self.workspace)),
                curve=curve,
            )
            self._record_nsys_capture(state, integration_dir / "profile", invalidate=True)
        print_message(
            f"[bold green]Integrator verdict: {decision}; included "
            f"{', '.join(sorted(included)) or 'none'}[/bold green]",
            log,
        )
        self._finish_item_batch(state, log)

    def _finish_item_batch(self, state: WorkflowState, log) -> None:
        """Durably close the round before removing its recoverable worktrees."""
        repo = self._trtllm_repo_path()
        paths = [str(entry.get("item_worktree_path", "")) for entry in state.item_batch]
        if state.integration_worktree_path:
            paths.append(state.integration_worktree_path)
        state.round_index += 1
        state.item_index = 0
        state.attempt_index = 0
        state.current_item_id = ""
        state.approach_violation = ""
        state.item_batch = []
        state.batch_started = False
        state.batch_completed = False
        state.integration_worktree_path = ""
        state.integration_branch = ""

        met, cumulative = self._target_met()
        if met:
            self._conclude_round_loop(
                state,
                f"target_improvement_pct reached (cumulative {cumulative:+.2f}%)",
                log,
            )
        elif state.round_index >= state.max_rounds:
            reason = "round budget exhausted"
            if state.profile_required:
                reason += " before the accepted runtime could be re-profiled"
            self._conclude_round_loop(state, reason, log)
        else:
            state.stage = STAGE_PROFILER
            self._checkpoint(state)

        # Every branch above checkpoints the completed batch. An interruption
        # during cleanup can leave an unused worktree, never a checkpoint that
        # still needs a deleted integration or candidate runtime.
        for path in paths:
            if path and Path(path).exists():
                self._remove_worktree_best_effort(repo, path, log)

    # ------------------------------------------------------------ approach guard

    def _detect_approach_violation(self, state: WorkflowState | None = None) -> str | None:
        """Return why the attempt violates ``optimize.approaches``, or ``None``.

        Purely mechanical checks, run after every optimizer turn: with
        ``config`` disallowed, the live tuning config must still match
        the accepted snapshot; with ``code`` disallowed, the checkout's
        worktree must still be clean. With both approaches allowed (the
        default) neither check runs.
        """
        approaches = self._allowed_approaches()
        active_state = state or WorkflowState(task_path=str(self.task_path))
        tuning_config, tuning_accepted = self._state_tuning_paths(active_state)
        if "config" not in approaches:
            live = tuning_config.read_text(encoding="utf-8")
            accepted = tuning_accepted.read_text(encoding="utf-8")
            if live != accepted:
                return (
                    "the attempt changed tuning/extra_llm_api_options.yaml, but "
                    "'config' is not in optimize.approaches"
                )
        if "code" not in approaches:
            repo = self._state_repo_path(active_state)
            if repo and not gitops.worktree_clean(repo):
                return (
                    "the attempt changed the TRT-LLM checkout, but 'code' is "
                    "not in optimize.approaches"
                )
        return None

    def _replan_only(self, state: WorkflowState) -> bool:
        """Whether the round about to open should re-plan instead of re-profile.

        True exactly when the standing profile is known to remain current:
        nothing was accepted and this campaign has produced a real profile
        to plan from. Rejected attempts run in isolated worktrees that are
        reset and removed, so the campaign checkout and accepted config still
        match the standing analysis. What *has* changed is the evidence — a
        batch of items now measured dead — so the round still runs an analyzer
        turn; it just plans from the standing profile and those verdicts
        rather than re-deriving the same traces.

        Round 1 (nothing profiled yet), the ``--reuse-analysis`` import
        turn, and a resumed pre-field checkpoint (whose profile currency
        is unknown, so ``load_state`` sets ``profile_required``) all
        profile normally.
        """
        return (
            state.round_index > 0
            and not state.reuse_pending
            and not state.reanalyze_pending
            and not state.profile_required
            and bool(state.last_profiled_analysis_dir)
        )

    def _conclude_round_loop(self, state: WorkflowState, reason: str, log) -> None:
        """End the optimization loop and park at the final verification.

        Callers have already counted the concluding round into
        ``round_index`` (so it reads as "rounds ran" from here on).
        """
        print_message(
            f"[bold green]✔ optimization loop done ({reason})[/bold green]",
            log,
        )
        state.stage = STAGE_QA
        self._checkpoint(state)

    # ---------------------------------------------------------------- gates

    @staticmethod
    def _is_nonempty(path: Path) -> bool:
        """True iff ``path`` exists and holds non-whitespace content."""
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())

    def _disagg_directive(self) -> str:
        """The disagg override every stage prompt opens with, or ``""``.

        The role prompts are composed of two layers: a system prompt built
        from shared fragments, and the per-stage instruction this
        orchestrator writes. ``DISAGG_CAMPAIGN`` supersedes the
        single-server guidance in the *first* layer — but the second layer
        also names ``trtllm-serve``, ``--extra_llm_api_options`` and a
        readiness poll, and it arrives last and reads as the more specific
        of the two. Without this the agent is handed a contradiction and
        the disagg section can lose on specificity.

        So the stage prompt states the mode up front and points at the
        section that governs it, rather than every stage's instruction
        growing a disagg variant of its own.
        """
        if not has_disagg(self._task_data()):
            return ""
        config = disagg_config_path(self._task_data())
        return (
            f"⚠️ **This campaign is DISAGGREGATED** (harness config: `{config}`). "
            f"Nothing below that mentions `trtllm-serve`, `--extra_llm_api_options` "
            f"or polling a server applies — your system prompt's "
            f"*Disaggregated serving* section replaces all of it.\n\n"
        )

    def _require_baseline_measurement(self) -> None:
        """Require successful numeric measurements at every configured curve point."""
        metric = str(self._optimize_block()["target_metric"])
        required_points = set(self._curve_points()) if self._curve_mode() else None
        measured_points: set[int] = set()
        for path in sorted(self.baseline_dir.rglob("*.json")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                measurements.finite_number(data.get(metric), metric, positive=True)
                measurements.finite_number(data.get("completed"), "completed", positive=True)
            except (OSError, ValueError, yaml.YAMLError):
                continue
            if required_points is None:
                return
            # The benchmark contract saves each run under concurrency_<c>.
            # A raw result can also identify its point through max_concurrency.
            point = data.get("max_concurrency")
            if isinstance(point, bool) or not isinstance(point, int):
                point = next(
                    (
                        int(match.group(1))
                        for part in path.relative_to(self.baseline_dir).parts[:-1]
                        if (match := re.fullmatch(r"concurrency_(\d+)", part))
                    ),
                    None,
                )
            if point in required_points:
                measured_points.add(point)
        if required_points is not None and measured_points == required_points:
            return
        missing = (
            f" Missing concurrency points: {sorted(required_points - measured_points)}."
            if required_points is not None
            else ""
        )
        raise RuntimeError(
            f"the baseline stage produced no measurement: require a positive finite "
            f"'{metric}' and a positive completed request count in result JSON under "
            f"{self.baseline_dir}.{missing} Read {self.baseline_results_path} for the "
            f"blocker. Fix it and re-run to retry the baseline, or pass --clean to start over."
        )

    def _require_stage_outputs(self, stage: str, paths: list[Path]) -> None:
        """Fail loudly if a stage finished without its required deliverable.

        An agent's turn can end early — e.g. after only launching a server
        and recording an interim progress entry — having written no
        deliverable. Without this gate the workflow would advance to a
        downstream stage that has nothing to work with. Raising here
        leaves the checkpoint un-advanced (``stage`` still names this
        role), so simply re-running the workflow retries the same stage;
        ``--clean`` starts over.
        """
        missing = [p.name for p in paths if not self._is_nonempty(p)]
        if missing:
            raise RuntimeError(
                f"{stage} stage finished but left required output "
                f"empty/missing: {', '.join(missing)}. The stage did not "
                f"complete its work (it likely ended its turn before writing "
                f"its deliverable). Re-run the workflow to retry this stage, "
                f"or pass --clean to start over."
            )

    def _require_profile_outputs(
        self, profile_dir: Path, *, imported: bool = False
    ) -> dict[str, Any]:
        """Validate a completed capture without advancing the profiler checkpoint."""
        try:
            return validate_profile_manifest(
                profile_dir,
                required_methods=None if imported else self._profile_methods(),
                require_raw=imported,
                require_report=not imported,
            )
        except ProfileError as exc:
            raise RuntimeError(
                f"profiler capture at {profile_dir} failed validation: {exc}. "
                f"Restore or repair the named capture artifacts and rerun; start a "
                f"fresh campaign if a new capture is needed."
            ) from exc

    def _validate_performance_model(self, analysis_dir: Path) -> dict[str, Any]:
        path = analysis_dir / performance_model.MODEL_FILENAME
        try:
            model = performance_model.load_model(path)
            problems = performance_model.validate_task_model(
                model,
                metric=str(self._optimize_block()["target_metric"]),
                concurrencies=self._curve_points() if self._curve_mode() else [None],
            )
            if problems:
                raise performance_model.ModelError("; ".join(problems))
            return model
        except performance_model.ModelError as exc:
            raise RuntimeError(
                f"analyzer stage finished but {path} failed validation: {exc}"
            ) from exc

    def _performance_model_instruction(self, state: WorkflowState) -> str:
        path = self._analysis_dir(state) / performance_model.MODEL_FILENAME
        previous = self._latest_performance_model(before_round=state.round_index + 1)
        prior_instruction = (
            f"Read `{previous}` as the prior model, retaining the evidence and "
            f"explanation for any changed assumption or bound. "
            if previous is not None
            else ""
        )
        imported_model = self.reuse_dir / reuse.PRIOR_PERFORMANCE_MODEL_NAME
        if previous is None and imported_model.is_file():
            prior_instruction += (
                f"Read `{imported_model}` as read-only prior art and "
                f"`{self.reuse_manifest_path}` for its source and selection scope. "
                f"Retain its corrected assumptions where capture, build, hardware, "
                f"workload and timing conditions match; do not silently replace them "
                f"with the original SOL projection. Resolve citations in the source "
                f"workspace. Source measurements do not establish this campaign's "
                f"current performance: write a fresh model below and mark mismatches "
                f"or missing evidence explicitly. "
            )
        return (
            prior_instruction
            + f"Write `{path}` as the single current theoretical best performance "
            f"model, even when SOL projection or kernel coverage is disabled. "
            f"Cover every configured concurrency (use null for scalar mode), "
            f"the task's target metric, matching runtime/workload/timing scope, "
            f"measured performance, a supported best-case bound or explicit unknown, "
            f"and an exhaustive non-overlapping gap decomposition. Every bound or "
            f"blocker needs evidence; unknowns need the next discriminating test. "
            f"Use this model for roadmap gain estimates and convergence. A stopped "
            f"campaign or failed attempt alone does not establish convergence. "
            f"Summarize the model in `analysis.md` and link detailed evidence.\n\n"
        )

    def _latest_performance_model(self, *, before_round: int | None = None) -> Path | None:
        candidates = []
        for path in self.rounds_dir.glob(f"round_*/analysis/{performance_model.MODEL_FILENAME}"):
            match = re.fullmatch(r"round_(\d+)", path.parent.parent.name)
            if match and (before_round is None or int(match.group(1)) < before_round):
                candidates.append((int(match.group(1)), path))
        return max(candidates)[1] if candidates else None

    def _report_performance_model(self) -> Path | None:
        final_model = self.final_analysis_dir / performance_model.MODEL_FILENAME
        return final_model if final_model.is_file() else self._latest_performance_model()

    def _model_needs_measurement(self, model: dict[str, Any]) -> bool:
        focus = self._focus_points()
        return any(
            point["status"] == "measurement_limited"
            or any(
                component["gap_kind"] == "measurement_limited" for component in point["components"]
            )
            for point in model["points"]
            if focus is None or point["concurrency"] in focus
        )

    def _record_analysis_source(
        self, state: WorkflowState, analysis_dir: Path, *, mode: str = "capture"
    ) -> None:
        """Record which immutable capture this completed analysis interprets."""
        if mode == "final_reconciliation" and not state.last_profile_dir:
            # Findings-only legacy reuse can optimize and reach QA without a
            # reusable capture. Preserve that explicit limit in the final model.
            (analysis_dir / "analysis_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "analysis_id": "final_reconciliation",
                        "capture_id": None,
                        "profile_dir": None,
                        "mode": mode,
                        "new_capture": False,
                        "capture_matches_model_build": False,
                        "capture_unavailable": "No reusable capture identity was recorded; "
                        "final measurements come from QA and prior findings remain scoped evidence.",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            return
        profile_dir = Path(state.last_profile_dir)
        manifest = validate_profile_manifest(profile_dir)
        relative_profile = profile_dir.relative_to(self.workspace)
        model_path = analysis_dir / performance_model.MODEL_FILENAME
        model_builds = reuse._model_builds(model_path)
        capture_build = json.dumps(manifest["runtime"]["build"], sort_keys=True)
        (analysis_dir / "analysis_manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "analysis_id": f"round_{state.round_index + 1}",
                    "capture_id": manifest["capture_id"],
                    "profile_dir": str(relative_profile),
                    "imported": state.reanalyze_pending,
                    "mode": mode,
                    "new_capture": mode == "capture" and not state.reanalyze_pending,
                    "capture_matches_model_build": bool(model_builds)
                    and all(build == capture_build for build in model_builds.values()),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def _validate_roadmap(self) -> dict[str, Any]:
        """Structurally validate roadmap.yaml as part of the analyzer gate."""
        try:
            roadmap = roadmap_schema.load_roadmap(self.roadmap_path)
        except RoadmapError as exc:
            raise RuntimeError(
                f"analyzer stage finished but roadmap.yaml failed validation:\n{exc}\n"
                f"Re-run the workflow to retry the analyzer stage, or pass --clean "
                f"to start over."
            ) from exc
        if self._curve_mode():
            # The per-point gate pairs curves by concurrency, so a
            # baseline curve at the wrong points poisons every later gain.
            points = self._curve_points()
            baseline = roadmap.get("baseline")
            curve = baseline.get("curve") if isinstance(baseline, dict) else None
            curve_points = (
                [p.get("concurrency") for p in curve] if isinstance(curve, list) else None
            )
            if curve_points != points:
                raise RuntimeError(
                    f"analyzer stage finished but roadmap.yaml's baseline.curve "
                    f"does not cover the task's concurrency points: expected "
                    f"{points}, got {curve_points}. Re-run the workflow to "
                    f"retry the analyzer stage, or pass --clean to start over."
                )
        return roadmap

    def _validate_kernel_ledger(
        self,
        roadmap: dict[str, Any],
        analysis_dir: Path,
        *,
        round_no: int,
    ) -> None:
        """Gate each analyzer output on coverage and evidence-backed model updates."""
        coverage = self._kernel_coverage()
        if coverage is None:
            return
        ledger_path = analysis_dir / kernel_ledger.LEDGER_FILENAME
        previous_path = self._latest_kernel_ledger(before_round=round_no)
        try:
            ledger = kernel_ledger.load_ledger(ledger_path)
            previous = kernel_ledger.load_ledger(previous_path) if previous_path else None
            problems = kernel_ledger.cross_validate(
                ledger,
                roadmap,
                float(coverage["coverage_target_pct"]),
                previous=previous,
                round_no=round_no,
            )
        except kernel_ledger.LedgerError as exc:
            raise RuntimeError(
                f"analyzer stage finished but {kernel_ledger.LEDGER_FILENAME} failed "
                f"validation:\n{exc}\nEvery kernel must carry all four dispositions "
                f"({' / '.join(kernel_ledger.QUESTIONS)}) and a performance model. "
                f"Re-run the workflow to retry the analyzer stage."
            ) from exc
        if problems:
            bullet = "\n  - "
            raise RuntimeError(
                f"analyzer stage finished but {ledger_path} failed the kernel ledger "
                f"contract:{bullet}{bullet.join(problems)}\n"
                f"Re-run the workflow to retry the analyzer stage."
            )

    def _validate_nsys_items(self, roadmap: dict[str, Any], analysis_dir: Path) -> None:
        """Hold the roadmap to the timeline analysis's own opportunity list.

        Self-gating on the artifact: enforced exactly when this round's
        ``nsys_analysis/items.json`` exists. A round that could not run
        the pipeline (nsys not requested, skill absent, export or
        pipeline error) writes none and owes nothing here — its
        *Caveats* line carries that. Raising leaves the checkpoint
        parked at the analyzer, so re-running retries the stage.
        """
        items_file = nsys_items.items_path(analysis_dir)
        if not items_file.is_file():
            return
        try:
            item_ids = nsys_items.load_item_ids(items_file)
        except nsys_items.NsysItemsError as exc:
            raise RuntimeError(
                f"analyzer stage finished but {exc}\nRe-run the workflow to retry "
                f"the analyzer stage, or pass --clean to start over."
            ) from exc
        problems = nsys_items.cross_validate(roadmap, item_ids)
        if problems:
            bullet = "\n  - "
            raise RuntimeError(
                f"analyzer stage finished but {self.roadmap_path} does not account "
                f"for {items_file}:{bullet}{bullet.join(problems)}\n"
                f"Re-run the workflow to retry the analyzer stage, or pass "
                f"--clean to start over."
            )

    # ------------------------------------------------------------ shared lookups

    def _task_data(self) -> dict[str, Any]:
        """Best-effort read of the resolved ``workspace/task.yaml``."""
        try:
            data = yaml.safe_load(self.task_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, yaml.YAMLError):
            return {}

    def _trtllm_repo_path(self) -> str:
        repo = self._task_data().get("trtllm_repo_path")
        return str(repo) if repo else ""

    def _curve_mode(self) -> bool:
        """Whether the resolved task spec runs in Pareto-curve mode."""
        return is_curve_mode(self._task_data())

    def _curve_points(self) -> list[int]:
        """The configured concurrency points (ascending in curve mode)."""
        return concurrency_points(self._task_data())

    def _sol_enabled(self) -> bool:
        """Whether the resolved task spec enables the projector stage.

        On by default — only ``sol.enabled: false`` turns it off.
        """
        return sol_enabled(self._task_data())

    def _focus_points(self) -> list[int] | None:
        """``optimize.focus_concurrencies`` when set, else ``None``.

        ``None`` means every configured point is scored (the default);
        the driving prompts only spell the focus rule out when a real
        subset is configured.
        """
        return focus_concurrencies(self._task_data())

    def _regression_budget(self) -> float | None:
        """``optimize.max_regression_pct`` when set, else ``None`` (strict)."""
        return max_regression_pct(self._task_data())

    def _kernel_coverage(self) -> dict[str, Any] | None:
        """``profile.kernel_coverage`` (defaults merged) when set, else ``None``."""
        return kernel_coverage(self._task_data())

    def _latest_kernel_ledger(self, *, before_round: int | None = None) -> Path | None:
        """The highest-round ``kernel_ledger.yaml``, or ``None``.

        The reporter's coverage proof is the final analyzer state; rounds
        are scanned numerically so ``round_10`` outranks ``round_9``.
        ``before_round`` excludes the current output during updates and retries.
        """
        candidates: list[tuple[int, Path]] = []
        for path in self.rounds_dir.glob(f"round_*/analysis/{kernel_ledger.LEDGER_FILENAME}"):
            match = re.fullmatch(r"round_(\d+)", path.parent.parent.name)
            if match and (before_round is None or int(match.group(1)) < before_round):
                candidates.append((int(match.group(1)), path))
        if not candidates:
            return None
        return max(candidates)[1]

    def _kernel_ledger_instruction(self, state: WorkflowState, *, reused: bool = False) -> str:
        """Direct every analyzer turn to maintain the same evidence-based ledger."""
        coverage = self._kernel_coverage()
        if coverage is None:
            return ""
        round_no = state.round_index + 1
        ledger_path = self._analysis_dir(state) / kernel_ledger.LEDGER_FILENAME
        previous = self._latest_kernel_ledger(before_round=round_no)
        prior = self.reuse_dir / reuse.PRIOR_KERNEL_LEDGER_NAME
        history = (
            f"Read `{previous}` as the previous ledger, including replan updates. "
            f"Carry forward its model revision history verbatim. For every changed "
            f"model assumption, derivation, operating point or prediction, append "
            f"a round {round_no} `model_revisions` entry with the exact `from` and "
            f"`to` values, reason, and supporting evidence. Preserve prior files. "
            if previous
            else "Start this campaign's `model_revisions` with an empty list. "
        )
        if previous is None and prior.is_file():
            history += (
                f"Read `{prior}` as read-only prior art. Check its capture and "
                f"operating conditions against the evidence being analyzed, cite "
                f"any models you retain, and bind question references to this "
                f"campaign's roadmap. Its revision history belongs to another "
                f"campaign and remains in the prior artifact. "
            )
        return (
            f"Apply the **per-kernel coverage contract**: write `{ledger_path}` "
            f"this round, including on replan-only and reused-analysis turns. "
            f"Enumerate kernels at/above {coverage['min_share_pct']}% of GPU time "
            f"and extend to {coverage['coverage_target_pct']}% coverage. Record "
            f"`coverage.gpu_busy_pct` and answer all four questions — eliminable? "
            f"faster? fusible? overlappable? — with roadmap references or cited "
            f"evidence. Keep the detailed dispositions in the machine-readable "
            f"ledger and link to them from the analysis. Maintain supporting "
            f"per-kernel and per-region models in this ledger, using shared models where kernel "
            f"boundaries change through fusion or elimination. {history}"
            f"Write `## Theoretical performance model` in "
            f"`{self._analysis_dir(state) / 'analysis.md'}` following the "
            f"report contract, including on replan/reuse turns. Give it the explicit "
            f"HTML anchor `theoretical-performance-model-round-{round_no}` "
            f"(prefix with `current-campaign-` if imported text already uses it). "
            f"Compare predictions with measured silicon performance under matching "
            f"conditions; retain source capture/build identities and measurement "
            f"evidence. On replan-only turns, keep standing measurements and "
            f"incorporate the experiments' new facts into the model and four "
            f"answers. An unsuccessful attempt alone establishes neither model "
            f"error nor exhausted headroom. Correct a prediction when evidence "
            f"falsifies its assumptions, and keep unknown predictions null and "
            f"remaining discrepancies explicit with the next discriminating test. "
            + (
                "These are inherited measurements; state their provenance and "
                "any mismatch with this campaign's requested conditions. "
                if reused
                else ""
            )
            + "The orchestrator validates the ledger and evidence-backed revisions "
            "when your turn ends.\n\n"
        )

    def _optimize_block(self) -> dict[str, Any]:
        block = self._task_data().get("optimize")
        merged = dict(OPTIMIZE_DEFAULTS)
        if isinstance(block, dict):
            merged.update(block)
        return merged

    def _allowed_approaches(self) -> tuple[str, ...]:
        """``optimize.approaches`` as a tuple, defensively defaulted.

        ``_task_data`` is best-effort, so a malformed value degrades to
        "everything allowed" (matching every other knob's fallback) —
        the restriction is only ever narrowed by a validated spec.
        """
        value = self._optimize_block().get("approaches")
        if isinstance(value, list) and value:
            return tuple(str(entry) for entry in value)
        return tuple(roadmap_schema.APPROACHES)

    def _accuracy_block(self) -> dict[str, Any] | None:
        block = self._task_data().get("accuracy")
        return block if isinstance(block, dict) else None

    def _profile_ranks(self) -> tuple[int, ...]:
        """The rank ids nsys must capture, from the resolved spec."""
        return profile_ranks(self._task_data())

    def _profile_methods(self) -> tuple[str, ...]:
        """``profile.methods`` from the resolved spec, defensively defaulted.

        The resolved ``task.yaml`` always carries the block, so the
        fallback only covers a hand-edited or unreadable file — and it
        defaults to nsys so the final profile is captured rather than
        silently skipped.
        """
        profile = self._task_data().get("profile")
        methods = profile.get("methods") if isinstance(profile, dict) else None
        if isinstance(methods, list) and methods:
            return tuple(str(entry) for entry in methods)
        return ("nsys",)

    def _record_nsys_capture(
        self, state: WorkflowState, directory: Path, *, invalidate: bool = False
    ) -> None:
        """Point ``last_nsys_dir`` at ``directory`` when it holds a capture.

        Called after the stages that may produce an nsys profile — the
        profiler's round profile, and an accepted attempt's
        accept-evidence capture — so the next evaluator's kernel
        comparison always names the freshest trace of the accepted state.
        An accepted runtime change invalidates the previous pointer even
        when its diagnostic capture failed. Refreshing an unchanged runtime
        may retain its earlier capture. The caller checkpoints.
        """
        if invalidate:
            state.last_nsys_dir = ""
        manifest_path = directory / PROFILE_MANIFEST_NAME
        if manifest_path.is_file():
            try:
                manifest = validate_profile_manifest(directory)
            except ProfileError as exc:
                if not invalidate:
                    raise
                # Candidate/integration captures are diagnostic. A broken
                # optional capture cannot invalidate an accepted measurement
                # or trap promotion retries on the same malformed manifest.
                print_message(
                    f"[yellow]accepted runtime has no usable nsys capture: "
                    f"{escape(str(exc))}[/yellow]",
                    get_logger().console,
                )
                return
            if manifest["methods"].get("nsys", {}).get("status") == "captured":
                state.last_nsys_dir = str(directory)
        elif any(directory.rglob("*.nsys-rep")) or (directory / "nsys_stats.txt").is_file():
            state.last_nsys_dir = str(directory)

    def _any_accepted_items(self) -> bool:
        """True iff the roadmap records at least one accepted item.

        Gates the final verification: with zero accepts the final state
        IS the baseline, and an independent re-measurement of it buys
        nothing. Defensive on an unreadable roadmap (e.g. the loop never
        got past the benchmarker) — no accepts, nothing to verify.
        """
        try:
            roadmap = roadmap_schema.load_roadmap(self.roadmap_path)
        except RoadmapError:
            return False
        return any(item.get("status") == "accepted" for item in roadmap.get("items", []))

    @staticmethod
    def _normalized_gain_pct(reference: float, measured: float, metric: str) -> float | None:
        """Signed % gain of ``measured`` vs ``reference``, positive = better.

        Mirrors the prompts' measurement protocol: throughput metrics
        improve upward, ``*_ms`` latency metrics improve downward.
        """
        if reference == 0:
            return None
        if metric.endswith("_ms"):
            return (reference - measured) / reference * 100.0
        return (measured - reference) / reference * 100.0

    def _target_met(self) -> tuple[bool, float | None]:
        """Whether ``optimize.target_improvement_pct`` is met, plus the gain.

        Computed deterministically from the roadmap ledger —
        ``current_best`` vs ``baseline`` on the target metric, both
        advanced only by accepted (evaluator-measured) items. Curve mode
        averages the per-point gains over the concurrency points the two
        curves share — restricted to ``optimize.focus_concurrencies``
        when configured, like every other curve→scalar derivation —
        falling back to the scalar means when either side carries no
        curve. Returns ``(False, None)`` when no target is set or the
        ledger is unreadable/incomplete.
        """
        optimize = self._optimize_block()
        target = optimize.get("target_improvement_pct")
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            return (False, None)
        try:
            roadmap = roadmap_schema.load_roadmap(self.roadmap_path)
        except RoadmapError:
            return (False, None)
        baseline = roadmap.get("baseline")
        best = roadmap.get("current_best")
        if not isinstance(baseline, dict) or not isinstance(best, dict):
            return (False, None)
        metric = str(roadmap.get("target_metric") or optimize["target_metric"])

        gains: list[float] = []
        base_curve = baseline.get("curve")
        best_curve = best.get("curve")
        if isinstance(base_curve, list) and isinstance(best_curve, list):
            focus = self._focus_points()
            reference_by_point = {
                point["concurrency"]: float(point["value"]) for point in base_curve
            }
            for point in best_curve:
                if focus is not None and point["concurrency"] not in focus:
                    continue
                reference = reference_by_point.get(point["concurrency"])
                if reference is None:
                    continue
                gain = self._normalized_gain_pct(reference, float(point["value"]), metric)
                if gain is not None:
                    gains.append(gain)
        if gains:
            cumulative = sum(gains) / len(gains)
        else:
            cumulative = self._normalized_gain_pct(
                float(baseline["value"]), float(best["value"]), metric
            )
        if cumulative is None:
            return (False, None)
        return (cumulative >= float(target), cumulative)

    def _reference_result_dir(self) -> Path:
        """Directory holding the reference measurement's result JSON(s).

        The last accepted attempt's directory, derived from
        ``current_best.source`` (the evaluation.md path the orchestrator
        recorded on accept), or ``baseline/`` while nothing has been
        accepted — the evaluator diffs its full metric set against the
        result JSONs found here.
        """
        try:
            roadmap = roadmap_schema.load_roadmap(self.roadmap_path)
        except RoadmapError:
            return self.baseline_dir
        best = roadmap.get("current_best")
        source = best.get("source") if isinstance(best, dict) else None
        if isinstance(source, str) and source.strip():
            parent = Path(source).parent
            if parent.is_absolute():
                candidates = [parent]
            else:
                # Workspace-relative per the roadmap spec; pre-fix state
                # files stored the path already workspace-prefixed, so try
                # it as-is (CWD-relative) too.
                candidates = [self.workspace / parent, parent]
            for candidate in candidates:
                if candidate.is_dir() and candidate != self.workspace:
                    return candidate
        return self.baseline_dir

    def _trtllm_hint(self) -> str:
        """Best-effort grep root for the source-search hints in prompts."""
        repo = self._trtllm_repo_path()
        if repo:
            return f"{repo}/tensorrt_llm"
        return "<trtllm_repo_path>/tensorrt_llm"

    # -------------------------------------------------------- decision readers

    def _latest_evaluator_decision(self, path: Path | None = None) -> str | None:
        entry = latest_entry(path or self.progress_path, "evaluator")
        if entry is None:
            return None
        d = str(entry.get("decision", "")).strip().upper()
        return d if d in EVALUATOR_DECISIONS else None

    def _latest_evaluator_measured_gain(self, path: Path | None = None) -> float | None:
        entry = latest_entry(path or self.progress_path, "evaluator")
        if entry is None:
            return None
        try:
            value = entry.get("measured_gain_pct")
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _latest_evaluator_measured_value(self, path: Path | None = None) -> float | None:
        entry = latest_entry(path or self.progress_path, "evaluator")
        if entry is None:
            return None
        try:
            value = entry.get("measured_value")
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _latest_evaluator_curve(self, path: Path | None = None) -> list[dict[str, Any]] | None:
        """The latest evaluator entry's per-point curve, or ``None``.

        Agent-supplied data: returned only when every point is well
        shaped (all four fields, numeric, int concurrency), so
        ``set_current_best`` never raises on it — a malformed curve
        degrades to the scalar path instead of crashing the accept.
        """
        entry = latest_entry(path or self.progress_path, "evaluator")
        if entry is None or not isinstance(entry.get("curve"), list) or not entry["curve"]:
            return None
        curve: list[dict[str, Any]] = []
        for point in entry["curve"]:
            if not isinstance(point, dict):
                return None
            concurrency = point.get("concurrency")
            if isinstance(concurrency, bool) or not isinstance(concurrency, int):
                return None
            try:
                curve.append(
                    {
                        "concurrency": concurrency,
                        "value": float(point["value"]),
                        "tok_s_user": float(point["tok_s_user"]),
                        "tok_s_gpu": float(point["tok_s_gpu"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                return None
        if [p["concurrency"] for p in curve] != sorted({p["concurrency"] for p in curve}):
            return None
        return curve

    def _latest_evaluator_ledger_fields(self, path: Path | None = None) -> dict[str, Any]:
        """The latest verdict's experimental evidence, defensively read.

        Agent-supplied data is validated before propagation. Returns only
        supported evidence fields with the expected types and enum values.
        """
        entry = latest_entry(path or self.progress_path, "evaluator")
        if entry is None:
            return {}
        fields: dict[str, Any] = {}
        implication = entry.get("gap_implication")
        if implication in GAP_IMPLICATIONS:
            fields["gap_implication"] = implication
        for key in ("gap_implication_note", "lever"):
            if isinstance(entry.get(key), str) and entry[key].strip():
                fields[key] = entry[key].strip()
        if entry.get("measurement_confidence") in MEASUREMENT_CONFIDENCES:
            fields["measurement_confidence"] = entry["measurement_confidence"]
        pooled = entry.get("measured_gain_pooled_pct")
        if isinstance(pooled, (int, float)) and not isinstance(pooled, bool):
            # The evaluator's own better estimate when it repeated the
            # measurement: a recorded gain its author already disowned
            # must not be inherited downstream as fact.
            fields["measured_gain_pooled_pct"] = float(pooled)
        blocker = entry.get("target_blocker")
        if isinstance(blocker, dict) and blocker.get("confirmed") and blocker.get("cause"):
            # Only a blocker the evaluator verified against the diff and
            # the source; an unconfirmed claim has not falsified anything.
            fields["target_blocker"] = {
                "cause": str(blocker.get("cause", "")),
                "detail": str(blocker.get("detail", "")),
                "evidence": str(blocker.get("evidence", "")),
            }
        return fields

    # ------------------------------------------------------------- round paths

    def _round_dir(self, state: WorkflowState) -> Path:
        return self.rounds_dir / f"round_{state.round_index + 1}"

    def _profile_dir(self, state: WorkflowState) -> Path:
        return self._round_dir(state) / "profile"

    def _analysis_dir(self, state: WorkflowState) -> Path:
        return self._round_dir(state) / "analysis"

    def _item_dir(self, state: WorkflowState) -> Path:
        """Directory for the item currently in the optimizer ⇄ evaluator loop.

        Namespaced per item so a round applying several items never
        collides attempt directories (a stale ``optimization_summary.md``
        from item 1 would otherwise satisfy item 2's output gate). The
        analyzer authors the ids, so they are sanitized for path use.
        """
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", state.current_item_id).strip("-.")[:48]
        return self._round_dir(state) / f"item_{state.item_index + 1}_{safe or 'item'}"

    def _attempt_dir(self, state: WorkflowState) -> Path:
        return self._item_dir(state) / f"attempt_{state.attempt_index + 1}"

    def _item_progress_path(self, state: WorkflowState) -> Path:
        return self._item_dir(state) / "progress.yaml"

    def _state_repo_path(self, state: WorkflowState) -> str:
        return state.item_worktree_path or self._trtllm_repo_path()

    def _state_tuning_paths(self, state: WorkflowState) -> tuple[Path, Path]:
        if not state.item_worktree_path:
            return self.tuning_config_path, self.tuning_accepted_path
        tuning_dir = self._item_dir(state) / "tuning"
        return (
            tuning_dir / "extra_llm_api_options.yaml",
            tuning_dir / "extra_llm_api_options.accepted.yaml",
        )

    # ------------------------------------------------------------------ agents

    def _stamp_progress(
        self,
        state: WorkflowState,
        *,
        round_no: int | None = None,
        with_attempt: bool = False,
        ctx: ProgressContext | None = None,
    ) -> None:
        """Position the progress context for the next agent call."""
        progress_ctx = ctx or self._progress_ctx
        data = read_progress(progress_ctx.path)
        progress_ctx.current_step = len(data[OPTIMIZATION_STAGE]) + 1
        progress_ctx.current_round = round_no if round_no is not None else state.round_index
        progress_ctx.current_attempt = state.attempt_index + 1 if with_attempt else None
        progress_ctx.current_item_id = state.current_item_id if with_attempt else ""

    def _runtime_checkout_instruction(self, state: WorkflowState) -> str:
        """Bind measurements to the source whose changes the campaign owns."""
        return (
            f"Active runtime checkout: `{self._state_repo_path(state)}`. "
            "Before launching the server, prepend this checkout to `PYTHONPATH` "
            "in the same execution shell/container and verify "
            '`python -c "import tensorrt_llm, os; '
            'print(os.path.realpath(tensorrt_llm.__file__))"`. '
            "The import must resolve under that checkout. For remote execution, "
            "use and verify its isolated staged copy. Record the resolved import "
            "path; a different installed package is a blocker.\n\n"
        )

    def _local_runtime_instruction(self, state: WorkflowState) -> str:
        """Coordinate shared local runtime sessions without serializing agent turns."""
        task = self._task_data()
        if state.item_execution != "parallel" or has_slurm_environment(task) or has_disagg(task):
            return ""
        lock_path = shlex.quote(str(self.workspace.resolve() / ".local_runtime.lock"))
        return (
            "**Concurrent local items — shared runtime protocol.** Other optimizer/evaluator "
            "pairs are active in their own worktrees. Reasoning, source edits, CPU-only checks, "
            "and offline result analysis may run concurrently outside the lock. Before any "
            "build/install that changes the runtime, GPU test or microbenchmark, server/port "
            "operation, benchmark replay, or profiling capture, write an item-owned shell "
            "script under your attempt directory and run it in the foreground with "
            f'`flock -x --close -- {lock_path} bash "<runtime-session-script>"`. '
            "All local items use this exact lock file; never unlink it or substitute an "
            "item-specific lock. If another item holds it, wait for the lock; never bypass "
            "it or kill the holder. If flock fails or is unavailable, report the blocker "
            "instead of running the session without it.\n\n"
            "One locked script must own the complete runtime session: establish this "
            "candidate's build/install and PYTHONPATH, verify its import/build identity, "
            "launch, poll readiness, exercise/measure, and tear down. Re-establish and "
            "verify your candidate after every acquisition because a sibling may have "
            "changed the shared installation. Install EXIT/INT/TERM cleanup traps before "
            "launch; tear down and wait for all owned server/profiler process groups before "
            "the script exits and releases the lock, including on failure or interruption. "
            "Do not background the flock wrapper or split a live server's lifecycle across "
            "separate lock acquisitions. Later experiments may use another complete locked "
            "session. All port inspection and stale-listener cleanup instructions apply "
            "only after acquiring this lock: a busy port while waiting may belong to a "
            "sibling, and you must never kill or benchmark that sibling's server.\n\n"
        )

    def _run_benchmarker(self, state: WorkflowState) -> None:
        self._stamp_progress(state, round_no=0)
        if self._curve_mode():
            points = self._curve_points()
            mean_scope = (
                f"mean over scored concurrency points {self._focus_points()}"
                if self._focus_points()
                else "mean over all configured concurrency points"
            )
            load_instruction = (
                f"then run `benchmark_serving.py` **once per concurrency "
                f"point {points}**, sequentially ascending, against the same "
                f"server (Pareto-curve mode — do not relaunch between "
                f"points). Use the **canonical `benchmark_serving.py` "
                f"command in your system prompt** — fill in the paths and "
                f"`benchmark` values, keep the other flags as given, and do "
                f"not improvise. Pass "
                f"`--result-dir {self.baseline_dir}/concurrency_<c>` for the "
                f"run at point `<c>` so each point's result JSON lands under "
                f"`baseline/`"
            )
            baseline_note = (
                f"naming the target metric's per-point values and their "
                f"**{mean_scope}** explicitly — this mean becomes the roadmap's "
                "`baseline.value` and the per-point rows become "
                "`baseline.curve`. "
            )
        else:
            load_instruction = (
                f"then run `benchmark_serving.py` at the single "
                f"configured operating point from the `benchmark` block. Use the "
                f"**canonical `benchmark_serving.py` command in your system "
                f"prompt** — fill in the paths and `benchmark` values, keep the "
                f"other flags as given, and do not improvise. Pass "
                f"`--result-dir {self.baseline_dir}` so the result JSON lands "
                f"under `baseline/`"
            )
            baseline_note = (
                "naming the target metric's value explicitly — it becomes "
                "the roadmap's `baseline.value`. "
            )
        self.benchmarker(
            self._disagg_directive()
            + f"Workspace: {self.workspace}\n\n"
            + self._runtime_checkout_instruction(state)
            + f"Read `{self.task_path}` for the spec — resolve `checkpoint_path`, "
            f"`trtllm_repo_path`, and the `benchmark` / `optimize` blocks.\n\n"
            f"Then **load the `perf-optimization-casebook` skill** (via the "
            f"`Skill` tool) as read-only reference, as your system prompt "
            f"directs, so your Configuration/Notes are grounded in known "
            f"TRT-LLM performance precedents.\n\n"
            f"Launch `trtllm-serve` with "
            f"`--extra_llm_api_options {self.tuning_config_path}` (the live "
            f"tuning config — always passed in this workflow), poll it to "
            f"readiness, {load_instruction}, and tear the server down "
            f"(always).\n\n"
            f"Do **all** of this within this single turn — poll readiness in "
            f"the foreground and do not yield to a background poll.\n\n"
            f"`Write` your baseline report to `{self.baseline_results_path}` "
            f"using the required structure in your system prompt "
            f"(Configuration / Metrics / Notes), {baseline_note}"
            f"Record the **exact** serve and benchmark commands so every "
            f"later stage can replay the same load.\n\n"
            f"Before completing your turn, call `append_benchmarker_progress` "
            f"with a `summary` of the commands you ran, the operating point, "
            f"the headline metrics, and the files you wrote."
        )

    def _run_projector(self, state: WorkflowState) -> None:
        self._stamp_progress(state, round_no=0)
        output = output_instruction(
            self.sol_methodology,
            str(self.sol_projection_path),
            f"{self.workspace}/sol_work/peaks.json",
            "the Analyzer's per-round measured\u2194SOL correlation",
        )
        self.projector(
            f"Workspace: {self.workspace}\n\n"
            f"You run once per campaign — your projection guides the "
            f"Analyzer's roadmap ranking and the Reporter's headroom story "
            f"for every later round.\n\n"
            f"Read `{self.task_path}` — the `sol` block (optional `gpu` "
            f"part-name hint) and the `benchmark` block "
            f"— and `{self.baseline_results_path}` (or call "
            f'`read_latest_progress` with `agent: "benchmarker"`) to recover '
            f"the measured baseline operating point, GPU, and headline "
            f"metrics. The parallel mapping (tp/pp/ep) comes from "
            f"`{self.tuning_config_path}` — the live tuning config every "
            f"server in this workflow runs with.\n\n"
            f"{projector_instruction(self.sol_methodology)}\n\n"
            f"Do **all** of this within this single turn; the stage only "
            f"counts as done once `sol_projection.md` is written.\n\n"
            f"{output}\n\n"
            f"Before completing your turn, call `append_projector_progress` "
            f"with a `summary` of the sources you used, the mapping, the "
            f"headline SOL ceiling and baseline-vs-SOL gap, and the files "
            f"you wrote."
        )

    def _baseline_curve_note(self) -> str:
        """How to seed ``baseline`` / ``current_best``, in curve mode.

        Empty in scalar mode — the analyzer's roadmap contract already
        covers the single-value case.
        """
        if not self._curve_mode():
            return ""
        focus = self._focus_points()
        if focus:
            mean_scope = (
                f"the **mean over the scored subset "
                f"`optimize.focus_concurrencies` {focus}** (the "
                f"campaign's focus regime — expected gains target it too)"
            )
        else:
            mean_scope = f"the **mean** across the concurrency points {self._curve_points()}"
        return (
            f" Pareto-curve mode: `baseline.value` is {mean_scope} "
            f"and `baseline.curve` carries the per-point "
            f"`{{concurrency, value, tok_s_user, tok_s_gpu}}` rows "
            f"for **all** configured points from the baseline "
            f"report's curve summary table — seed "
            f"`current_best` equal, curve included."
        )

    def _run_reused_analyzer(self, state: WorkflowState) -> None:
        """Plan-only analyzer turn over imported artifacts (no profiling).

        Runs in place of round 1's profile when ``--reuse-analysis``
        seeded the workspace: every trace this round would have captured
        is already on disk, so the analyzer's whole job is to turn that
        evidence into ``roadmap.yaml``. It launches no server and runs no
        profiler — the entire point of the reuse is not to spend that GPU
        time again. Round 2 then profiles for real: imported traces
        describe another run's build, so they never set the "last
        profiled" pointer the replan rule keys off (see the guard beside
        that assignment in ``run``).
        """
        analysis_dir = self._analysis_dir(state)
        findings_path = analysis_dir / "analysis.md"
        profile_dir = Path(state.last_profile_dir) if state.last_profile_dir else analysis_dir
        prior_roadmap_context = ""
        if self.prior_roadmap_path.is_file():
            prior_roadmap_context = (
                f"The source run was itself a perf-optimize campaign: its "
                f"roadmap is parked at `{self.prior_roadmap_path}` as "
                f"**read-only prior art**. Its `accepted` / `failed` items "
                f"and their `measured_gain_pct` describe *that* campaign's "
                f"checkout, not this one — never copy its statuses, "
                f"`current_best`, or ids into the roadmap you write. Use it "
                f"the way you would use evidence: carry forward the pending "
                f"items its findings still support, and do not re-propose "
                f"what it recorded as failed unless this checkout changes "
                f"the premise (say so in `evidence` when you do).\n\n"
            )
        projection_context = ""
        if self._sol_enabled():
            projection_context = (
                f"Read `{self.sol_projection_path}` (imported with the rest) "
                f"as the initial theoretical model. Reconcile its assumptions "
                f"with imported measurements into `performance_model.yaml`, "
                f"the current basis for gap estimates and convergence. Any measured↔SOL "
                f"correlation the source produced is already in "
                f"`{analysis_dir}` — use it as the initial model and revise assumptions "
                f"when the saved evidence warrants a correction.\n\n"
            )
        self.analyzer(
            self._disagg_directive() + f"Workspace: {self.workspace}\n"
            f"Round: 1 (**reused analysis** — no profiling this round)\n"
            f"Analysis directory (already populated): {analysis_dir}\n\n"
            f"This campaign was launched with "
            f"`--reuse-analysis {state.reuse_analysis_dir}`: a previous run's "
            f"analysis has been imported into this workspace, so **round 1 "
            f"skips profiling entirely**. Do **not** launch `trtllm-serve`, "
            f"do **not** run nsys / ncu, and do **not** "
            f"run the benchmark — every trace this round would have captured "
            f"is already on disk, and re-deriving it is exactly the cost the "
            f"reuse exists to avoid.\n\n"
            f"Read `{self.task_path}` (the spec this campaign runs under), "
            f"`{self.reuse_manifest_path}` (what was imported, and from "
            f"where), `{findings_path}` **in full** plus the traces and "
            f"summaries in `{analysis_dir}` plus preserved captures in "
            f"`{profile_dir}` (read-only, including `{PROFILE_REPORT_NAME}` when "
            f"present in the imported source), and "
            f"`{self.baseline_results_path}` (the validated baseline "
            f"measurement, imported or measured by this campaign — the anchor for the roadmap's `baseline` "
            f"block).\n\n"
            + projection_context
            + prior_roadmap_context
            + self._performance_model_instruction(state)
            + self._kernel_ledger_instruction(state, reused=True)
            + f"Then **load the `perf-optimization-casebook` skill** (via the "
            f"`Skill` tool) as your system prompt directs, and tag each "
            f"roadmap item's `casebook_ref` with the matching *bottleneck "
            f"signal → candidate pattern* row.\n\n"
            f"Two checks you still owe — both read-only, neither needs a "
            f"GPU: verify the imported analysis actually describes **this** "
            f"task (same model/checkpoint, parallel mapping in "
            f"`{self.tuning_config_path}`, and operating point as "
            f"`{self.task_path}`), and run the **dormant-capability sweep** "
            f"per your system prompt (checkpoint config + weight index, "
            f"unset serving knobs, gated code paths — inspect files and "
            f"`grep -rn`/`rg` under `{self._trtllm_hint()}`; launch "
            f"nothing). Where the imported evidence does not fit this task, "
            f"say so plainly rather than planning on it — a mismatch is a "
            f"finding, not a blocker to hide.\n\n"
            f"`Write` `{self.roadmap_path}` from scratch per the roadmap "
            f"contract in your system prompt — the `baseline` block from "
            f"`{self.baseline_results_path}` (the target metric's value), "
            f"`current_best` seeded equal to it, and `items` ordered by "
            f"`expected_gain_pct` descending, each grounded in the imported "
            f"evidence with a quantified `expected_gain_rationale`."
            f"{self._baseline_curve_note()}\n\n"
            f"Preserve the imported analysis verbatim at "
            f"`{self.reuse_dir / reuse.PRIOR_ANALYSIS_DIRNAME / reuse.FINDINGS_NAME}` "
            f"before replacing `{findings_path}` with this campaign's concise analysis: "
            f"`## Result`, `## Theoretical performance model`, `## Gap analysis`, "
            f"and `## Next actions`. Cite the imported source and capture identity; "
            f"state fit limitations and only capability findings that change the model "
            f"or next actions. Keep detailed derivations in linked artifacts.\n\n"
            f"Before completing your turn, call `append_analyzer_progress` "
            f"with a `summary` naming the reuse source, which imported "
            f"artifacts you planned from, the fit check's outcome, and the "
            f"roadmap items you authored with their expected gains."
        )

    def _run_profiler(self, state: WorkflowState) -> None:
        """Capture immutable runtime evidence for the analyzer's offline turn."""
        round_no = state.round_index + 1
        self._stamp_progress(state, round_no=round_no)
        profile_dir = self._profile_dir(state)
        replay_note = " (scalar mode: one replay at the configured concurrency)"
        if self._curve_mode() and self._curve_points():
            replay_note = (
                " (Pareto-curve mode: one replay at the largest concurrency "
                f"point, {self._curve_points()[-1]}, only"
            )
            replay_note += (
                "; when `benchmark.num_prompts` is a list, use the entry paired "
                "with each selected point in `benchmark.concurrency`; otherwise "
                "use the scalar prompt count)"
            )
        prior_model = self._latest_performance_model(before_round=round_no)
        measurement_context = ""
        if prior_model is not None:
            model = performance_model.load_model(prior_model)
            if self._model_needs_measurement(model):
                measurement_context = (
                    f"Read `{prior_model}` before choosing captures. Its scored points "
                    f"need new evidence: follow each `next_test` and measurement-limited "
                    f"component to target the missing concurrency, phase, kernel or "
                    f"counter. Prioritize those captures even when the build is unchanged. "
                    f"Record which model question each capture resolves, or the precise "
                    f"reason the requested evidence remains unavailable.\n\n"
                )
                replay_note = (
                    " at the operating points and iteration phases requested by the "
                    "prior model's next tests, preserving their matching prompt counts"
                )
        coverage = self._kernel_coverage()
        if coverage is not None:
            ncu_scope = (
                f"For the per-kernel coverage contract, enumerate every kernel "
                f"at/above {coverage['min_share_pct']}% of GPU time and extend "
                f"until {coverage['coverage_target_pct']}% is covered. Group "
                f"honestly-shared rows and capture over bounded ncu passes, "
                f"re-filtering on still-missing stems. Preserve the kernel list, "
                f"pass coverage, and the window's GPU busy share. The analyzer "
                f"will author the kernel dispositions and ledger offline."
            )
        else:
            ncu_scope = (
                "Target the top nsys kernels with the canonical ncu flags and "
                "bounded `--launch-count`; save `server_ncu.ncu-rep` and exports."
            )
        self.profiler(
            self._disagg_directive() + f"Workspace: {self.workspace}\n"
            f"Round: {round_no}\n"
            f"Profile directory (write capture artifacts here): {profile_dir}\n"
            f"Active runtime checkout: `{self._trtllm_repo_path()}`\n"
            f"Active tuning config: `{self.tuning_config_path}`\n\n"
            + measurement_context
            + f"Read `{self.task_path}` and `{self.baseline_results_path}` to "
            f"recover the serving commands and operating point. Verify this "
            f"checkout's profiling knobs with `rg` via `Bash` under "
            f"`{self._trtllm_hint()}` and record the actual build, import path, "
            f"effective config, hardware, ranks, and exact commands.\n\n"
            f"Capture only the methods in `profile.methods`: relaunch "
            f"`trtllm-serve` with `--extra_llm_api_options {self.tuning_config_path}`, "
            f"replay the canonical benchmark load" + replay_note + f", and drive "
            f"nsys from the canonical `nsys profile` command in your system prompt. "
            f"{profile_ranks_note(self._profile_ranks())} Export reports with "
            f"`nsys export --type sqlite` and preserve `nsys_stats.txt`. Load "
            f"`internal-perf-nsight-system-analysis` (fully-qualified "
            f"`trtllm-agent-toolkit:internal-perf-nsight-system-analysis` if needed) "
            f"and run `run_all.py` into `{profile_dir}/capture_preprocessing` "
            f"only as needed to check capture quality and select ncu targets. "
            f"Save a reusable taxonomy there when available.\n\n"
            f"Keep the **Run A2** GPU-metrics and backtrace captures separate "
            f"from the timing capture: use `--gpu-metrics-devices` / "
            f"`--gpu-metrics-frequency` for utilization, and the backtrace flags "
            f"for call sites. Preserve each export; record unavailable auxiliary "
            f"passes with a reason. For ncu load `perf-nsight-compute-analysis` "
            f"(fully-qualified `trtllm-agent-toolkit:perf-nsight-compute-analysis` "
            f"if needed) as the capture methodology. {ncu_scope}\n\n"
            f"Poll readiness in the foreground and tear every server down before "
            f"completing this turn. Write `{profile_dir / PROFILE_REPORT_NAME}` with "
            f"a short capture summary, operating points and runtime identity, coverage "
            f"and limitations, plus links to raw evidence. Keep analytical conclusions "
            f"for the analyzer. Write `{profile_dir / PROFILE_MANIFEST_NAME}` last "
            f"using the manifest contract in your system prompt: a unique "
            f"`capture_id`, runtime provenance (`serve_command`, "
            f"`benchmark_command`, `config`, `build`, and `import_path`), and a "
            f"`methods` entry for every requested method. Each entry must be "
            f"`captured` with the command and relative nonempty artifacts, or "
            f"`unavailable` with an explicit reason. Partial work is not a "
            f"completed capture. Preserve these artifacts for read-only reuse; "
            f"the analyzer will write findings, SOL correlation, dispositions, "
            f"and the roadmap in a separate directory.\n\n"
            f"Before completing your turn, call `append_profiler_progress` "
            f"with a `summary` of capture quality and coverage, methods captured "
            f"or unavailable, the artifact paths, and server cleanup."
        )

    def _run_analyzer(self, state: WorkflowState) -> None:
        round_no = state.round_index + 1
        self._stamp_progress(state, round_no=round_no)
        analysis_dir = self._analysis_dir(state)
        if state.reuse_pending and not state.reanalyze_pending:
            self._run_reused_analyzer(state)
            return
        if self._replan_only(state):
            self._run_replan_analyzer(state)
            return
        profile_dir = (
            Path(state.last_profile_dir) if state.last_profile_dir else self._profile_dir(state)
        )
        if round_no == 1:
            curve_note = self._baseline_curve_note()
            round_context = (
                f"This is **round 1**: run the **dormant-capability sweep** "
                f"per your system prompt (checkpoint config + weight index, "
                f"unset serving knobs, gated code paths — record details in "
                f"`dormant_capabilities.md` and link consequential findings from Next actions), "
                f"then author `{self.roadmap_path}` from scratch "
                f"per the roadmap contract in your system prompt — the `baseline` "
                f"block from `{self.baseline_results_path}` (the target metric's "
                f"value), `current_best` seeded equal to it, and `items` ordered "
                f"by `expected_gain_pct` descending.{curve_note}"
            )
        else:
            profile_reason = (
                "**the profiler has captured the current runtime** — "
                "interpret that completed evidence without recapturing it"
            )
            round_context = (
                f"This is **round {round_no}**: the roadmap at "
                f"`{self.roadmap_path}` already exists, and {profile_reason}. "
                f"This round rebuilds the analysis from its capture. "
                f'Call `read_latest_progress` with `agent: "evaluator"` for '
                f"the verdicts on the items that **failed**. Establish which "
                f"mechanism actually ran and what each experiment proved; a "
                f"REJECT alone does not disprove the optimization premise. "
                f"Analyze the saved evidence "
                f"and update the roadmap in place: re-order / revise "
                f"still-pending items, add newly exposed ones, mark stale "
                f"pending items `obsolete`. Never rewrite accepted/failed "
                f"history, `baseline`, `current_best`, or existing ids."
            )
        coverage_context = self._kernel_ledger_instruction(state)
        projection_context = ""
        if self._sol_enabled():
            projection_context = (
                f"Also read `{self.sol_projection_path}` (or call "
                f'`read_latest_progress` with `agent: "projector"`) as '
                f"the initial theoretical model. Reconcile its assumptions with "
                f"the measured evidence into `performance_model.yaml`; use that "
                f"single current model to rank roadmap items and bound expected "
                f"end-to-end gains. Preserve the original projection as provenance, "
                f"not a competing headline ceiling. Run the offline **measured↔SOL "
                f"correlation** per your system prompt: load the "
                f"`internal-perf-sol-analysis` skill (via the `Skill` tool; "
                f"fully-qualified "
                f"`trtllm-agent-toolkit:internal-perf-sol-analysis` if the "
                f"bare name is not found), build "
                f"`{analysis_dir}/regions.json` from this round's traces "
                f"(structural facts only — never invented params or "
                f"measured_ms), run the skill's `sol_calc.py analyze` "
                f"against the Projector's "
                f"`{self.workspace}/sol_work/peaks.json`, write "
                f"`{analysis_dir}/sol.json`, and transcribe the joined "
                f"per-op evidence into the linked model derivations supporting "
                f"`## Theoretical performance model` (or `Correlation "
                f"unavailable: <reason>` when a precondition fails). If you "
                f"leave the roadmap with no "
                f"actionable pending item while projected headroom remains, "
                f"explain that stop in `analysis.md`'s **Gap analysis** — every part "
                f"of the gap gets a supported item, an evidence-backed constraint, "
                f"or is marked unexplained. Keep it brief and link to the "
                f"comparison's explanations and next tests.\n\n"
            )
        import_context = ""
        if state.reanalyze_pending:
            import_context = (
                f"This is **re-analysis of imported captures** from "
                f"`{state.reuse_analysis_dir}`. Read `{self.reuse_manifest_path}` "
                f"and preserve the source evidence verbatim. Rebuild the derived "
                f"analysis using the current methodology, taxonomy, and "
                f"hypotheses; previous findings under "
                f"`{self.reuse_dir / 'prior_analysis'}` are read-only prior art. "
                f"These measurements describe the source run, so verify their "
                f"fit to this task and record differences in model, mapping, "
                f"config, runtime, and operating point. They do not establish "
                f"that this campaign's checkout was profiled.\n\n"
            )
        self.analyzer(
            f"Workspace: {self.workspace}\n"
            f"Round: {round_no} (**offline analysis**)\n"
            f"Source profile directory (read-only): {profile_dir}\n"
            f"Analysis directory (write derived artifacts here): {analysis_dir}\n\n"
            + import_context
            + f"Read `{self.task_path}`, `{self.baseline_results_path}`, and "
            f"`{profile_dir / PROFILE_MANIFEST_NAME}` to establish capture "
            f"provenance, methods, ranks, and the measured operating points. "
            f"Do not launch servers, replay workloads, run benchmarks, or "
            f"capture nsys / ncu data. Offline exports and analysis commands "
            f"may read the saved reports, with every output under "
            f"`{analysis_dir}`; never modify `{profile_dir}`.\n\n"
            f"{round_context}\n\n"
            + projection_context
            + f"Early on, **load the `perf-optimization-casebook` skill** as "
            f"read-only reference via the `Skill` tool; tag each roadmap item's "
            f"`casebook_ref` with the matching bottleneck signal → candidate "
            f"pattern.\n\n"
            f"Decompose the saved timeline with "
            f"`internal-perf-nsight-system-analysis` (fully-qualified "
            f"`trtllm-agent-toolkit:internal-perf-nsight-system-analysis` if "
            f"needed). Read saved sqlite exports, or export a saved report "
            f"using `nsys export --type sqlite` into `{analysis_dir}`. Run "
            f"`run_all.py` single-variant into `{analysis_dir}/nsys_analysis`; "
            f"when a GPU-metrics capture exists, pass `--metrics-profile` "
            f"pointed at its sqlite. Verify the taxonomy and author "
            f"`{analysis_dir}/nsys_analysis/items.json`, then account for every "
            f"id in `roadmap.yaml`'s `nsys_items` block. Derive per-iteration "
            f"time, busy/idle rungs, and the compute-absent split "
            f"(launch-starved / blocking / dependency-stalled) from that "
            f"pipeline, not from the `nsys stats` table alone.\n\n"
            f"Read `{profile_dir / PROFILE_REPORT_NAME}` for capture coverage and "
            f"limitations (legacy imports may lack it). Interpret the saved ncu reports (`server_ncu.ncu-rep` or the "
            f"per-pass reports) and exports using "
            f"`perf-nsight-compute-analysis` (fully-qualified "
            f"`trtllm-agent-toolkit:perf-nsight-compute-analysis` if needed) "
            f"as the offline interpretation methodology: classify SOL%, bound "
            f"class, occupancy, and stalls. Respect unavailable methods and "
            f"capture caveats. If evidence is insufficient, name the needed "
            f"additional capture in findings; do not obtain it this turn.\n\n"
            + self._performance_model_instruction(state)
            + coverage_context
            + f"Write `{analysis_dir / 'analysis.md'}` with only `## Result`, "
            f"`## Theoretical performance model`, `## Gap analysis`, and "
            f"`## Next actions`. Link detailed traces, per-kernel dispositions and "
            f"derivations instead of duplicating them. Then write "
            f"`{self.roadmap_path}` with items ordered by expected benefit "
            f"and quantified `expected_gain_rationale` grounded across the "
            f"available analyses.\n\n"
            f"Before completing your turn, call `append_analyzer_progress` "
            f"with a `summary` of the source capture, analyses regenerated, "
            f"evidence limitations, and roadmap items added / re-ordered / "
            f"marked obsolete with their expected gains."
        )

    def _run_replan_analyzer(self, state: WorkflowState) -> None:
        """Replan-only analyzer turn: no server, no profiler (see ``_replan_only``).

        Opens after a round that accepted nothing. Every attempt ran in an
        isolated worktree that was reset and removed, so the standing
        analysis still describes the campaign runtime and re-deriving it
        would buy the campaign nothing. What the round *did* produce is
        verdicts — items now measured dead — and turning those into roadmap
        edits is this turn's whole job.
        """
        round_no = state.round_index + 1
        analysis_dir = self._analysis_dir(state)
        profiled_dir = state.last_profiled_analysis_dir
        prev_round_dir = self.rounds_dir / f"round_{state.round_index}"
        attempted = [d for d in prev_round_dir.glob("item_*") if d.is_dir()]
        if len(attempted) == 1:
            attempted_note = (
                f"Its one attempted item is the `item_*` directory under `{prev_round_dir}`"
            )
        elif attempted:
            attempted_note = (
                f"Its {len(attempted)} attempted items are the `item_*` "
                f"directories under `{prev_round_dir}`"
            )
        else:
            attempted_note = f"Its attempted items are under `{prev_round_dir}`"
        projection_context = ""
        if self._sol_enabled():
            projection_context = (
                f"`{self.sol_projection_path}` and any measured↔SOL "
                f"correlation already in `{profiled_dir}` provide the standing "
                f"comparison; new facts may correct the model. If you "
                f"leave the roadmap with no actionable pending item while "
                f"projected headroom remains, explain the stop in this round's "
                f"**Gap analysis** — every part of the gap gets a supported item, "
                f"an evidence-backed constraint, or is marked unexplained, "
                f"with links to existing explanations and next tests.\n\n"
            )
        self.analyzer(
            self._disagg_directive() + f"Workspace: {self.workspace}\n"
            f"Round: {round_no} (**replan only** — no profiling this round)\n"
            f"Analysis directory (write your artifacts here): {analysis_dir}\n\n"
            f"Round {state.round_index} accepted **nothing**. "
            f"{attempted_note}; each ran in an isolated worktree that the "
            f"orchestrator reset and removed. The campaign checkout and "
            f"accepted tuning config remain unchanged, so the runtime is "
            f"still the state the analysis in `{profiled_dir}` describes. "
            f"Do **not** launch "
            f"`trtllm-serve`, "
            f"do **not** run nsys / ncu, and do **not** "
            f"run the benchmark: a fresh profile of an unchanged build "
            f"would reproduce those traces at full GPU cost. Plan from them "
            f"instead.\n\n"
            f"What *has* changed is the evidence. Call "
            f'`read_latest_progress` with `agent: "evaluator"` (raise '
            f"`steps` until it reaches back through round "
            f"{state.round_index}) for each attempt's `decision`, "
            f"`reason_category`, and measured gain, and read the "
            f"`evaluation.md` files under `{prev_round_dir}`. Those verdicts "
            f"describe attempts against **this** build. Determine which mechanism "
            f"actually ran and what each outcome established. A performance "
            f"shortfall alone does not bound the bottleneck's recoverable time; "
            f"a functionality failure may expose an implementation bug. An "
            f"attempt the orchestrator auto-rejected for "
            f"violating the approach restriction never reached the "
            f"evaluator and has no `evaluation.md` — its "
            f"`optimization_summary.md` is the record, and what it proves is "
            f"about the item's *realizability* under this campaign's "
            f"`optimize.approaches`, not about the bottleneck.\n\n"
            + projection_context
            + self._performance_model_instruction(state)
            + self._kernel_ledger_instruction(state)
            + f"Then update `{self.roadmap_path}` **in place** against that "
            f"evidence: mark `obsolete` every pending item the round's "
            f"verdicts disprove or whose premise they undercut, revise the "
            f"`expected_gain_pct` / `evidence` of pending items the "
            f"measurements bound, re-order what survives, and add items the "
            f"failures themselves imply (a REJECT often names the real "
            f"constraint) — **load the `perf-optimization-casebook` skill** "
            f"(via the `Skill` tool) as your system prompt directs before "
            f"authoring any, and tag each new item's `casebook_ref` with the "
            f"matching *bottleneck signal → candidate pattern* row. Never "
            f"rewrite `accepted` / `failed` history, "
            f"`baseline`, `current_best`, or existing ids; new items get "
            f"fresh ids continuing the sequence.\n\n"
            f"**If the evidence leaves nothing actionable, leave the roadmap "
            f"with no actionable pending item and say so.** The orchestrator "
            f"reads that as the campaign's end and closes the loop — the "
            f"administrative stop; only the current model can establish convergence. "
            f"Do not invent items to keep the "
            f"loop alive; an unfounded item costs a full benchmark to "
            f"disprove.\n\n"
            f"`Write` `{analysis_dir / 'analysis.md'}` as this "
            f"round's concise analysis: `## Result`, `## Theoretical performance "
            f"model`, `## Gap analysis`, and `## Next actions`. Link the standing "
            f"capture and analysis (`{profiled_dir}`). Explain only experiment results "
            f"that change the model, gap attribution, or next actions; retain "
            f"standing measurement provenance and identify untested gaps.\n\n"
            f"Before completing your turn, call `append_analyzer_progress` "
            f"with a `summary` naming the round that accepted nothing, the "
            f"verdicts you planned from, and the items you marked obsolete / "
            f"revised / added with their expected gains."
        )

    def _run_optimizer(
        self,
        state: WorkflowState,
        *,
        agent: AgentLayer | None = None,
        progress_ctx: ProgressContext | None = None,
    ) -> None:
        round_no = state.round_index + 1
        attempt_no = state.attempt_index + 1
        self._stamp_progress(state, round_no=round_no, with_attempt=True, ctx=progress_ctx)
        attempt_dir = self._attempt_dir(state)
        repo = self._state_repo_path(state)
        tuning_config, _ = self._state_tuning_paths(state)
        retry_context = ""
        if attempt_no > 1 and state.approach_violation:
            allowed = ", ".join(f"`{a}`" for a in self._allowed_approaches())
            retry_context = (
                f"\n\nThis is a **retry** (attempt {attempt_no} of "
                f"{state.max_attempts_per_item}): the orchestrator "
                f"auto-REJECTED the previous attempt **without evaluation** "
                f"because {state.approach_violation}, and has already "
                f"reverted the checkout and the tuning config to the last "
                f"accepted state. There is no evaluator feedback for it. "
                f"Re-implement the item strictly through the allowed "
                f"approach(es) — {allowed} — per the approach restriction in "
                f"your system prompt; if the item cannot be realized that "
                f"way, make no change and record the blocker in your summary."
            )
        elif attempt_no > 1:
            retry_context = (
                f"\n\nThis is a **retry** (attempt {attempt_no} of "
                f"{state.max_attempts_per_item}): the Evaluator PUSHED BACK "
                f"the previous attempt and the orchestrator has already "
                f"reverted the checkout and the tuning config to the last "
                f"accepted state. First call `read_latest_progress` with "
                f'`agent: "evaluator"` and read the previous attempt\'s '
                f"`evaluation.md` under `{self._item_dir(state)}` — then fix "
                f"the PUSH_BACK reason, not a different problem."
            )
        projection_context = ""
        if self._sol_enabled():
            projection_context = (
                f"Also read `{self.sol_projection_path}` (or call "
                f'`read_latest_progress` with `agent: "projector"`) as '
                f"**context, not spec**: where the item leaves you a choice "
                f"of realization variants or knob values, aim at the binding "
                f"ceiling per the SOL guidance in your system prompt, and "
                f"record the `SOL alignment:` line in your summary — the "
                f"item's `how_to_apply` outranks the projection, and the "
                f"projection never expands the item.\n\n"
            )
        verdict_context = ""
        if state.round_index > 0 or (state.item_execution == "serial" and state.item_index > 0):
            # Verdicts land after the roadmap is authored, so an earlier
            # item's REJECT can invalidate a premise this item's text
            # still carries — the re-profile only corrects it next round.
            verdict_context = (
                f"Earlier items' verdicts may have corrected facts this "
                f"item's text still relies on (the roadmap predates them): "
                f"skim the completed items' `evaluation.md` files under "
                f"`{self.rounds_dir}` — their Verdict / `Gap implication:` "
                f"lines outrank this item's `evidence` where they conflict, "
                f"and a premise they disprove is a blocker to record in "
                f"your summary, not a claim to re-assert.\n\n"
            )
        (agent or self.optimizer)(
            self._disagg_directive()
            + self._local_runtime_instruction(state)
            + f"Workspace: {self.workspace}\n"
            f"Round: {round_no} — item {state.item_index + 1} of at most "
            f"{state.max_items_per_round} this round — attempt {attempt_no} "
            f"of {state.max_attempts_per_item}\n"
            f"Roadmap item to implement: **{state.current_item_id}** (read it in "
            f"`{self.roadmap_path}`)\n"
            f"Optimization branch: `{state.item_branch or state.campaign_git_branch}` "
            f"at `{repo}`\n"
            f"Active runtime checkout: `{repo}`\n"
            f"Attempt directory (write your artifacts here): {attempt_dir}"
            f"{retry_context}\n\n"
            f"Inside the Slurm job script, before any Python command or "
            f"`trtllm-serve` launch:\n\n"
            f'`export PYTHONPATH="{repo}${{PYTHONPATH:+:$PYTHONPATH}}"`\n\n'
            f"Read `{self.task_path}` and the roadmap item, then **load the "
            f"`perf-optimization-casebook` skill** (via the `Skill` tool) as "
            f"your system prompt directs and implement **exactly this one "
            f"item** following its `how_to_apply` and the matched casebook "
            f"case: `approach: config` → edit `{tuning_config}`; "
            f"`approach: code` → edit the source under `{repo}` under "
            f"the git discipline in your system prompt (active-runtime "
            f"check first; locate code paths with shell `grep -rn`/`rg` via "
            f"`Bash`; never commit).\n\n"
            + projection_context
            + verdict_context
            + f"Then smoke-check: launch `trtllm-serve` with "
            f"`--extra_llm_api_options {tuning_config}`, poll to "
            f"readiness in the foreground within this turn, send one "
            f"completion request, and tear the server down (always). Do "
            f"**not** run the full benchmark — measuring is the Evaluator's "
            f"job.\n\n"
            f"`Write` your summary to "
            f"`{attempt_dir / 'optimization_summary.md'}` using the required "
            f"structure in your system prompt (What changed / Files touched / "
            f"Mapping to the roadmap item / Expected gain / Smoke check / "
            f"Risks).\n\n"
            f"Before completing your turn, call `append_optimizer_progress` "
            f"with a `summary` of the item you implemented, what you changed, "
            f"the smoke-check result, and any risks or blockers."
        )

    def _evaluator_capture_context(self, state: WorkflowState) -> str:
        """The accept-evidence capture directive for this attempt, or ``""``.

        Composed by the orchestrator so the stateless evaluator gets the
        two things it cannot know: whether nsys is configured at all, and
        where the previous capture of the accepted state lives (the
        deterministic ``last_nsys_dir`` pointer).
        """
        if "nsys" not in self._profile_methods():
            return ""
        profile_dir = self._attempt_dir(state) / "profile"
        if state.last_nsys_dir:
            previous_capture = Path(state.last_nsys_dir)
            if state.last_nsys_dir == state.last_profile_dir:
                # Round captures have a separate completed analysis. Imported
                # evidence uses this round's analysis without claiming local
                # runtime freshness through last_profiled_analysis_dir.
                taxonomy_dir = (
                    Path(state.last_profiled_analysis_dir)
                    if state.last_profiled_analysis_dir
                    else self._analysis_dir(state)
                )
            else:
                # Evaluator captures carry their own analysis and may postdate
                # the standing round analysis by several accepted items.
                taxonomy_dir = previous_capture
            decompose = (
                f"its `run_all.py` **comparative** — `--variant before` on the "
                f"`.sqlite` under `{state.last_nsys_dir}` and `--variant after` "
                f"on this capture's, reusing `{taxonomy_dir / 'taxonomy.json'}` "
                f"for both variants (fall back to "
                f"`{previous_capture / 'taxonomy.json'}` when the separate "
                f"analysis has no taxonomy) so both sides classify identically — into "
                f"`{profile_dir}/nsys_analysis`, whose `difference/rank-0/` "
                f"holds the signed per-iteration and module-slice deltas "
                f"(single-variant, compared by hand, only if that capture kept "
                f"no `.sqlite`)"
            )
            compare = (
                f"report those deltas against the previous capture of the "
                f"accepted state at `{state.last_nsys_dir}`"
            )
        else:
            decompose = f"its `run_all.py` single-variant into `{profile_dir}/nsys_analysis`"
            compare = (
                "there is no previous capture to compare against — report "
                "this capture's kernel picture on its own"
            )
        curve_note = ""
        if self._curve_mode() and self._curve_points():
            curve_note = (
                f" (Pareto-curve mode: one replay at the largest concurrency "
                f"point, {self._curve_points()[-1]}, only)"
            )
        return (
            f"**Accept-evidence duty — only if your verdict is APPROVE.** "
            f"After your clean measurement and gate arithmetic, capture the "
            f"candidate state per the accept-evidence procedure in your "
            f"system prompt: tear down the measurement server, relaunch "
            f"under the canonical `nsys profile` wrap, replay the canonical "
            f"load once{curve_note}, tear down, and save the `.nsys-rep`, "
            f"the `nsys stats` output as `nsys_stats.txt`, and the replay "
            f"log into `{profile_dir}`. Then **decompose that capture with "
            f"the `internal-perf-nsight-system-analysis` skill** (via the `Skill` "
            f"tool; fully-qualified "
            f"`trtllm-agent-toolkit:internal-perf-nsight-system-analysis` if the bare "
            f"name is not found) as your system prompt directs — `nsys "
            f"export --type sqlite`, then {decompose} — so the mechanism check "
            f"below rests on a measured per-iteration budget rather than an "
            f"eyeballed one. In `evaluation.md`'s *Kernel "
            f"evidence* section, {compare}, and state whether the item's "
            f"claimed mechanism is visible in the trace. On REJECT or "
            f"PUSH_BACK, skip the capture entirely.\n\n"
        )

    def _run_evaluator(
        self,
        state: WorkflowState,
        *,
        agent: AgentLayer | None = None,
        progress_ctx: ProgressContext | None = None,
        validation_feedback: str = "",
    ) -> None:
        round_no = state.round_index + 1
        attempt_no = state.attempt_index + 1
        self._stamp_progress(state, round_no=round_no, with_attempt=True, ctx=progress_ctx)
        attempt_dir = self._attempt_dir(state)
        repo = self._state_repo_path(state)
        tuning_config, tuning_accepted = self._state_tuning_paths(state)
        optimize = self._optimize_block()
        reference_dir = self._reference_result_dir()
        if self._curve_mode():
            points = self._curve_points()
            focus = self._focus_points()
            budget = self._regression_budget()
            if budget is not None:
                regress_rule = (
                    f"regress by more than the task's declared regression "
                    f"budget `optimize.max_regression_pct` = {budget} "
                    f"(name any point kept inside it)"
                )
            else:
                regress_rule = "regress by more than the noise floor"
            if focus:
                mean_rule = (
                    f"the mean over the **scored subset "
                    f"`optimize.focus_concurrencies` {focus}** must pass "
                    f"both thresholds AND no point (scored or not) may "
                    f"{regress_rule}"
                )
                mean_fields = (
                    f"`measured_gain_pct` (the mean of per-point gains over "
                    f"the scored subset {focus}), "
                    f"`measured_value` (the mean of per-point values over "
                    f"that subset), and "
                    "`curve` (the per-point rows for ALL points)"
                )
            else:
                mean_rule = f"the mean must pass both thresholds AND no point may {regress_rule}"
                mean_fields = (
                    "`measured_gain_pct` (the mean of per-point gains), "
                    "`measured_value` (the mean of per-point values), and "
                    "`curve` (the per-point rows)"
                )
            measure_instruction = (
                f"then measure with the **canonical `benchmark_serving.py` "
                f"command in your system prompt** once per concurrency point "
                f"{points}, sequentially ascending over the same server, "
                f"passing `--result-dir {attempt_dir}/concurrency_<c>` per "
                f"point. Curve mode: apply the **Pareto gate** — per-point "
                f"gains vs `current_best.curve` (same concurrency), "
                f"{mean_rule} — per the acceptance gate in your "
                f"system prompt"
            )
            full_diff_note = (
                f"the reference result JSONs for the full-metric diff are "
                f"under `{reference_dir}` (per-point `concurrency_<c>/` "
                f"subdirectories; diff at the largest point)"
            )
            progress_fields = (
                "with all six fields — `summary`, `decision` "
                "(APPROVE|REJECT|PUSH_BACK), `reason_category` "
                "(none|code_quality|functionality|perf_shortfall), "
                f"{mean_fields} — exactly as measured; the "
                "orchestrator acts on them"
            )
        else:
            measure_instruction = (
                f"then measure with the "
                f"**canonical `benchmark_serving.py` command in your system "
                f"prompt** at the configured operating point, passing "
                f"`--result-dir {attempt_dir}`. Compute `measured_gain_pct` "
                f"against `current_best.value` per the measurement protocol, "
                f"and apply the acceptance gate"
            )
            full_diff_note = (
                f"the reference result JSON for the full-metric diff is under `{reference_dir}`"
            )
            progress_fields = (
                "with all five fields — `summary`, `decision` "
                "(APPROVE|REJECT|PUSH_BACK), `reason_category` "
                "(none|code_quality|functionality|perf_shortfall), "
                "`measured_gain_pct`, `measured_value` — exactly as "
                "measured; the orchestrator acts on them"
            )
        if attempt_no >= state.max_attempts_per_item:
            attempt_note = (
                " This is the item's **final attempt**: PUSH_BACK is not "
                "available (the orchestrator treats it as REJECT) — decide "
                "APPROVE or REJECT."
            )
        else:
            attempt_note = ""
        correction = ""
        if validation_feedback:
            correction = (
                "**Correct the previous evaluator submission.** The orchestrator "
                f"rejected its structured approval: {validation_feedback}\n"
                "The candidate source, tuning config, and measurement artifacts "
                "have been preserved. For this corrective turn, reuse valid "
                "existing evidence and fix the report/arithmetic; repeat only "
                "checks or measurements whose evidence is missing or invalid. "
                "Keep the candidate unchanged and append a corrected verdict.\n\n"
            )
        (agent or self.evaluator)(
            correction
            + self._disagg_directive()
            + self._local_runtime_instruction(state)
            + f"Workspace: {self.workspace}\n"
            f"Round: {round_no} — item {state.item_index + 1} of at most "
            f"{state.max_items_per_round} this round — attempt {attempt_no} "
            f"of {state.max_attempts_per_item}\n"
            f"Roadmap item under review: **{state.current_item_id}** (read it in "
            f"`{self.roadmap_path}`)\n"
            f"Optimization branch: `{state.item_branch or state.campaign_git_branch}` "
            f"at `{repo}`\n"
            f"Active runtime checkout: `{repo}`\n"
            f"Active tuning config: `{tuning_config}`\n"
            f"Accepted tuning config snapshot: `{tuning_accepted}`\n"
            f"Attempt directory (write your artifacts here): {attempt_dir}\n"
            f"Acceptance gate: accept_fraction={optimize['accept_fraction']}, "
            f"noise_floor_pct={optimize['noise_floor_pct']}, "
            f"target_metric={optimize['target_metric']}"
            + (
                f", focus_concurrencies={self._focus_points()}"
                if self._curve_mode() and self._focus_points()
                else ""
            )
            + (
                f", max_regression_pct={self._regression_budget()}"
                if self._curve_mode() and self._regression_budget() is not None
                else ""
            )
            + "\n\n"
            f"Inside the Slurm job script, before any Python command or "
            f"`trtllm-serve` launch:\n\n"
            f'`export PYTHONPATH="{repo}${{PYTHONPATH:+:$PYTHONPATH}}"`\n\n'
            f"Read `{self.task_path}`, the roadmap item and `current_best` in "
            f"`{self.roadmap_path}`, and the Optimizer's "
            f"`{attempt_dir / 'optimization_summary.md'}`.\n\n"
            f"Review the change (`git -C {repo} diff` + `--stat` "
            f"and `git status --porcelain`; diff "
            f"`{tuning_config}` against "
            f"`{tuning_accepted}` for config edits), verify "
            f"functionality (launch `trtllm-serve` with "
            f"`--extra_llm_api_options {tuning_config}`, poll to "
            f"readiness in the foreground within this turn, send completion "
            f"requests; targeted tests for code items — locate them with "
            f"shell `grep -rn`/`rg` via `Bash`), {measure_instruction}. "
            f"Tear every server down (always).\n\n"
            f"`Write` your report to `{attempt_dir / 'evaluation.md'}` using "
            f"the required structure (Change review / Functionality / "
            f"Performance / Kernel evidence / Verdict), showing the gate "
            f"arithmetic and the full-metric diff — {full_diff_note}."
            f"{attempt_note}\n\n"
            f"{self._evaluator_capture_context(state)}"
            f"Before completing your turn, call `append_evaluator_progress` "
            f"{progress_fields}."
        )

    def _run_qa(self, state: WorkflowState) -> None:
        self._stamp_progress(state)
        accuracy = self._accuracy_block()
        if accuracy:
            accuracy_context = (
                f"`task.yaml` **has** an `accuracy` block: run its `command` "
                f"verbatim against the live server, record the score under "
                f"`{self.final_verification_dir}`, and compare it to "
                f"`baseline_score` / `max_drop_pct` as your system prompt "
                f"directs."
            )
        else:
            accuracy_context = (
                "`task.yaml` has **no** `accuracy` block: skip the accuracy "
                'step entirely and note "accuracy: not configured" in your '
                "report."
            )
        if self._curve_mode():
            points = self._curve_points()
            focus = self._focus_points()
            if focus:
                mean_scope = (
                    f"the **mean over the scored subset `optimize.focus_concurrencies` {focus}**"
                )
            else:
                mean_scope = "the **mean across concurrency points**"
            benchmark_instruction = (
                f"run the **canonical `benchmark_serving.py` command in "
                f"your system prompt** once per concurrency point {points}, "
                f"sequentially ascending over the same server, with "
                f"`--result-dir {self.final_verification_dir}/concurrency_<c>` "
                f"per point"
            )
            cumulative_instruction = (
                f"Compute `cumulative_improvement_pct` from your own "
                f"measurement — {mean_scope} of "
                f"the per-point gain vs the roadmap's `baseline.curve` entry "
                f"with the same concurrency"
            )
            progress_fields = (
                f"with `summary`, `cumulative_improvement_pct` ({mean_scope} "
                f"vs baseline.curve), and `curve` (your per-point rows, all "
                f"points) from your own measurement"
            )
        else:
            benchmark_instruction = (
                f"run the **canonical `benchmark_serving.py` command in "
                f"your system prompt** at the configured operating point with "
                f"`--result-dir {self.final_verification_dir}`"
            )
            cumulative_instruction = (
                "Compute `cumulative_improvement_pct` from your own "
                "measurement vs the roadmap's `baseline.value`"
            )
            progress_fields = (
                "with both fields — `summary` and "
                "`cumulative_improvement_pct` — from your own measurement"
            )
        self.qa(
            self._disagg_directive()
            + f"Workspace: {self.workspace}\n"
            + self._runtime_checkout_instruction(state)
            + f"Campaign: the optimization loop is over ({state.round_index} "
            f"round(s) ran); the system under test is the final accepted "
            f"state.\n"
            f"Verification directory (write your artifacts here): "
            f"{self.final_verification_dir}\n\n"
            f"You are the campaign's final verification. Ground yourself "
            f"ONLY in `{self.task_path}`, `{self.roadmap_path}`, and your own "
            f"runs this turn — do not read other agents' reports or progress "
            f"entries.\n\n"
            f"Launch `trtllm-serve` with "
            f"`--extra_llm_api_options {self.tuning_config_path}` (the live "
            f"tuning config), poll to readiness in the foreground within this "
            f"turn, {benchmark_instruction}, and send a few completion requests "
            f"as a sanity check. {accuracy_context} Tear every server down "
            f"(always).\n\n"
            f"{cumulative_instruction}.\n\n"
            f"`Write` your report to `{self.verification_report_path}` using "
            f"the required structure (Independent benchmark / Sanity / "
            f"Accuracy / Conclusion).\n\n"
            f"Before completing your turn, call `append_qa_progress` "
            f"{progress_fields}."
        )

    def _run_final_analyzer(self, state: WorkflowState) -> None:
        """Reconcile final measurements and accepted changes without new runtime work."""
        self._stamp_progress(state)
        previous = self._latest_performance_model()
        prior_instruction = (
            f"Read `{previous}` and `{previous.with_name('analysis.md')}` as the last "
            f"round's model and supporting analysis. "
            if previous is not None
            else "No prior model is available; explicitly retain unknown bounds. "
        )
        self.analyzer(
            f"Workspace: {self.workspace}\n"
            f"Final model reconciliation (offline only)\n"
            f"Write final analysis artifacts to `{self.final_analysis_dir}`.\n\n"
            + prior_instruction
            + f"Read `{self.task_path}`, `{self.roadmap_path}`, "
            f"`{self.verification_report_path}` and benchmark result JSONs under "
            f"`{self.final_verification_dir}`. Read accepted evaluation and integration "
            f"reports under `{self.rounds_dir}`, `{self.tuning_accepted_path}`, and "
            f"the accepted git diff `{state.campaign_git_base_commit}..HEAD` in "
            f"`{self._trtllm_repo_path()}` as read-only evidence.\n\n"
            f"This turn reconciles the final model only: do not edit the roadmap, "
            f"checkout, tuning configuration, captures or prior models; do not launch "
            f"servers, benchmarks, profilers, or optimization attempts. Preserve "
            f"existing artifacts and write only inside `{self.final_analysis_dir}` "
            f"apart from your required progress entry.\n\n"
            f"Write `{self.final_analysis_dir / performance_model.MODEL_FILENAME}` "
            f"for the task's target metric and every configured concurrency (null "
            f"in scalar mode), identifying the final accepted build and QA workload. "
            f"Use QA's matching final measurements. Retain supported bounds only "
            f"where assumptions still hold after accepted changes; explain revisions "
            f"with evidence. Preserve the old capture identity beside inherited "
            f"component measurements and never relabel them as final-build captures. "
            f"Leave unmodeled final components and unsupported bounds unknown with "
            f"their next discriminating test. Missing final profiling means an "
            f"unresolved comparison, not convergence. Reconcile all gap arithmetic "
            f"against this one final model.\n\n"
            f"Write `{self.final_analysis_dir / 'analysis.md'}` with only `## Result`, "
            f"`## Theoretical performance model`, `## Gap analysis`, and "
            f"`## Next actions`; link source derivations, captures and experiments "
            f"instead of repeating them. Call `append_analyzer_progress` summarizing "
            f"the final model status, corrected assumptions and unresolved evidence."
        )

    def _run_reporter(self, state: WorkflowState) -> None:
        self._stamp_progress(state)
        model_path = self._report_performance_model()
        model_read = (
            f" `{model_path}` (the authoritative current model and gap accounting; "
            f"check that its build and workload still match the final measurement), "
            f"`{model_path.with_name('analysis.md')}` (its current analysis),"
            if model_path is not None
            else " `performance_model.yaml` (unavailable; report an unresolved "
            "model and do not claim convergence),"
        )
        if self._curve_mode():
            pareto_chart = "; a Pareto curve may show the operating-point tradeoff"
            focus = self._focus_points()
            if focus:
                pareto_headline = (
                    f" (curve mode with `optimize.focus_concurrencies` "
                    f"{focus}: the ledger means and the headline score the "
                    f"focus subset — say so wherever a mean is presented — "
                    f"with every point still shown in the model table)"
                )
            else:
                pareto_headline = (
                    " (curve mode: the mean across concurrency points, with the "
                    "per-point curve in the model table)"
                )
        else:
            pareto_chart = ""
            pareto_headline = ""
        if self.verification_report_path.is_file():
            headline_source = (
                f"comes from the final verification's independent "
                f"measurement (`{self.verification_report_path}`)"
            )
        else:
            headline_source = (
                "comes from the roadmap ledger (`current_best` vs "
                "`baseline`) — the final verification did not run (no "
                "accepted items), so say so"
            )
        if state.last_nsys_dir:
            after_profile = (
                f"`{state.last_nsys_dir}` holds the freshest nsys capture "
                f"of the final accepted state — prefer it as the 'after' "
                f"side of the kernel comparison"
            )
        else:
            after_profile = (
                "no nsys capture postdates the last accepted item — the "
                "kernel comparison falls back to the latest round profile "
                "and must say which accepted items it misses"
            )
        if self._sol_enabled():
            projection_read = (
                f" `{self.sol_projection_path}` (the Projector's SOL "
                f"projection, retained as the original model assumptions; "
                f"use the latest validated `performance_model.yaml` for all "
                f"current ceilings and remaining-gap arithmetic),"
            )
        else:
            projection_read = ""
        coverage_read = ""
        if self._kernel_coverage() is not None:
            final_ledger = self._latest_kernel_ledger()
            ledger_name = (
                f"`{final_ledger}`"
                if final_ledger is not None
                else "the final round's `analysis/kernel_ledger.yaml` (none "
                "was written — the section must say the ledger is "
                "unavailable)"
            )
            coverage_read = (
                f" {ledger_name} (the final round's per-kernel disposition "
                f"ledger — supporting evidence for the central model; link "
                f"its detailed dispositions without reproducing the ledger),"
            )
            coverage_read += (
                " In **Theoretical performance model**, link to the "
                "**Theoretical performance model** in the latest "
                "analyzer's `analysis/analysis.md`, beside that ledger. "
                "Use a relative section link to its actual current-campaign "
                "round anchor in Markdown and HTML, not an imported section "
                "with the same heading. The analyzer "
                "owns the full derivations and layer table; do not reproduce or "
                "re-derive them. Summarize the supported iteration bound, measured "
                "performance, remaining headroom and unresolved gaps. If the "
                "section is missing, say it is unavailable and link to the ledger. "
                "Distinguish the last modeled capture from subsequent "
                "accepted changes and final measurements; an unprofiled final "
                "build has no validated model comparison. A closed roadmap "
                "does not establish model/implementation convergence."
            )
        reuse_read = ""
        if state.reuse_analysis_dir:
            baseline_origin = (
                "The baseline was measured by this campaign; the benchmarker progress "
                "entry records that run. "
                if latest_entry(self.progress_path, "benchmarker") is not None
                else "The baseline was measured by that run, not this one. "
            )
            reuse_read = (
                f" `{self.reuse_manifest_path}` (this campaign was launched "
                f"with `--reuse-analysis {state.reuse_analysis_dir}`. {baseline_origin}"
                f"Name which profiles and analyses were imported using their manifests, "
                f"and state their provenance beside the model comparison),"
            )
        self.reporter(
            f"Workspace: {self.workspace}\n"
            f"Optimization branch: `{state.campaign_git_branch}` — base commit "
            f"`{state.campaign_git_base_commit}` in `trtllm_repo_path`\n\n"
            f"The campaign is over ({state.round_index} round(s) ran). Read "
            f"**all** inputs listed in your system prompt: `{self.task_path}`, "
            f"`{self.baseline_results_path}`,"
            f"{reuse_read}{projection_read}{model_read}{coverage_read} "
            f"`{self.roadmap_path}` (final "
            f"statuses, expected vs measured gains, baseline/current_best), "
            f"every round's `integration/integration.md` and "
            f"`integration/candidate_manifest.yaml` when present, "
            f"every `optimization_summary.md` / `evaluation.md` under "
            f"`{self.rounds_dir}`, every round's "
            f"`analysis/analysis.md` + `profile/{PROFILE_REPORT_NAME}` + "
            f"`profile/{PROFILE_MANIFEST_NAME}` "
            f"+ `profile/nsys_stats.txt` (legacy rounds may keep stats under "
            f"`analysis/`) and "
            f"every accepted attempt's `profile/nsys_stats.txt` (the "
            f"kernel-level before/after evidence; {after_profile}), "
            f"`{self.verification_report_path}` when it exists (the final "
            f"verification's independent benchmark + accuracy), "
            f"`{self.progress_path}` (the chronological trail the "
            f"trajectory is reconstructed from: serial mode advances on accepted "
            f"evaluator results; parallel mode advances once per accepted integrator "
            f"result, whose combined measurement is authoritative. Standalone "
            f"candidate measurements share a base and are not sequential gains), "
            f"`{self.tuning_accepted_path}` (the final accepted config), and "
            f"— read-only — `git -C <trtllm_repo_path> log --oneline` and "
            f"`git diff --stat` over `{state.campaign_git_base_commit[:12]}..HEAD` for "
            f"the code-diff summary. Launch no servers and run no "
            f"benchmarks.\n\n"
            f"`Write` `{self.report_path}` with only `## Result`, "
            f"`## Theoretical performance model`, `## Gap analysis`, and "
            f"`## Changes and next actions`. Put baseline, final measured performance, current "
            f"theoretical best, remaining gap and convergence status in one table "
            f"for every operating point. State scoring scope and final verification "
            f"status beside the table. Summarize accepted changes and only failed "
            f"experiments that explain the gap; link detailed reports, configuration, "
            f"kernel coverage, revisions and roadmap instead of duplicating them. "
            f"Use the same current model for the headline and every breakdown. "
            f"Then `Write` `{self.report_html_path}` mirroring the content 1:1, "
            f"self-contained and using charts only when they clarify the model or "
            f"gap{pareto_chart}. The cumulative improvement {headline_source}"
            f"{pareto_headline}. Distinguish hardware limits, scope limits, failed "
            f"attempts, measurement limits and unresolved gaps.\n\n"
            f"Before completing your turn, call `append_reporter_progress` "
            f"with a `summary` of the cumulative improvement headline, the "
            f"accepted/failed item counts, and confirmation that both files "
            f"were written."
        )


if __name__ == "__main__":
    from .cli import main

    main()
