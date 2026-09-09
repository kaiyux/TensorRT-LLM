from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_flow.workflows.perf_analyze.sol_methodology import resolve_sol_methodology

from .disagg import has_disagg
from .prompts import PROMPTS_DIRNAME, build_perf_optimize_prompts, dump_prompt_bundle
from .state import STATE_FILENAME
from .task_schema import (
    TaskSchemaError,
    has_slurm_environment,
    headroom_ledger,
    kernel_coverage,
    load_and_validate_task_yaml,
    sol_enabled,
)
from .workflow import PerfOptimizeWorkflow


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteratively optimize a trtllm-serve deployment: benchmark the "
        "baseline, profile and rank optimizations into roadmap.yaml, apply the "
        "top items serially or concurrently in isolated worktrees (up to "
        "optimize.max_items_per_round per round), gate each candidate on "
        "code quality / functionality / measured "
        "perf (the evaluator approves, rejects, or pushes back each attempt, "
        "profiling candidate-ready states under nsys), directly accept serial "
        "candidates or integrate and benchmark a parallel batch, run the full optimize.max_rounds "
        "budget unless the roadmap exhausts or the improvement target is met, "
        "verify the final state with one independent QA benchmark, and report "
        "expected-vs-measured gains — via a benchmarker -> [optional profiler -> analyzer -> "
        "(optimizer <-> evaluator) items -> optional integrator] x rounds "
        "-> qa -> reporter loop. "
        "A one-shot SOL projector stage runs between the baseline and "
        "round 1 (sol_projection.md) unless task.yaml sets "
        "`sol.enabled: false`. "
        "--reuse-analysis seeds a fresh run from a previous perf-analyze / "
        "perf-optimize workspace so the campaign starts at the optimize stage."
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Path to the task.yaml spec. Requires `checkpoint_path` and "
        "`trtllm_repo_path`; optional top-level `extra_llm_api_options` "
        "path, optional `benchmark` / `profile` / `optimize` / `accuracy` "
        "blocks, an optional `slurm-environment` block, and an optional "
        "`sol` block (all fields optional: `enabled` gates the one-shot "
        "SOL projector stage — on by default — and `gpu` names the GPU "
        "part for the SOL skill's peaks calculator). "
        "An optional `profile.kernel_coverage` block "
        "activates the per-kernel coverage contract: the profiler's ncu "
        "dive covers every kernel above the share bar, and the analyzer answers "
        "eliminable?/faster?/fusible?/overlappable? per kernel in a "
        "schema-validated kernel_ledger.yaml each round. An optional "
        "`profile.headroom_ledger` block (requires the two above) adds the "
        "campaign's per-part gap accounting in headroom_ledger.yaml: where "
        "the remaining gap-to-SOL sits, which lever each failed item spent "
        "against it, and what a named buildable implementation would achieve. "
        "See task.example.yaml.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace/perf-optimize"),
        help="Workspace directory for shared state (task.yaml, roadmap.yaml, "
        "sol_projection.md, headroom_ledger.yaml, baseline/, tuning/, rounds/, "
        "optimization_report.md/.html, progress.yaml, prompts/) and run "
        "artifacts. Each launch snapshots every role's composed system "
        "prompt to prompts/<role>.md.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the workspace checkpoint and managed files/directories "
        f"({STATE_FILENAME}, sol_projection.md, roadmap.yaml, "
        "headroom_ledger.yaml, optimization_report.md/.html, "
        "progress.yaml, baseline/, rounds/, worktrees/, tuning/, sol_work/, "
        "reused_analysis/) and start fresh. The "
        "TRT-LLM checkout is not touched (abandoned perf-optimize/* branches "
        "are left for inspection). Without this flag the workflow resumes "
        "from the checkpoint when one is present, and starts fresh otherwise.",
    )
    parser.add_argument(
        "--reuse-analysis",
        default=None,
        metavar="DIR",
        help="Seed a fresh run from a previous perf-analyze workspace or "
        "perf-optimize campaign workspace instead of re-deriving its "
        "analysis: its baseline report (+ result JSONs), SOL projection "
        "(+ sol_work/), and newest profile findings (+ traces and "
        "kernel_ledger.yaml) are copied into this workspace, the "
        "benchmarker/projector stages are skipped, and round 1's analyzer "
        "runs plan-only by default — authoring roadmap.yaml from the imported evidence "
        "with no server, profiler, or benchmark. A source roadmap.yaml is "
        "kept aside as read-only prior art (reused_analysis/), never as this "
        "campaign's ledger. Add --reanalyze to reinterpret the saved captures "
        "offline before optimizing. Whatever the source lacks is produced normally. "
        "Fresh runs only — ignored on resume.",
    )
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="With --reuse-analysis DIR, regenerate round 1's findings and "
        "analysis products from saved profiler captures without launching "
        "a server, profiler, or benchmark for that analysis, then continue "
        "the optimization campaign normally. Requires reusable captures. "
        "Fresh runs only; omit this flag when resuming, since the requested "
        "analysis mode is checkpointed.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Override `optimize.max_rounds` from task.yaml on a fresh run "
        "(each round opens with an optional profiler when the standing profile "
        "is stale, then an analyzer turn — reusing findings for a replan "
        "otherwise — then evaluates up "
        "to `optimize.max_items_per_round` roadmap items per the configured "
        "`optimize.item_execution` mode). "
        "Ignored on resume — the checkpointed budget wins.",
    )
    args = parser.parse_args(argv)
    if args.reanalyze and not args.reuse_analysis:
        parser.error("--reanalyze requires --reuse-analysis DIR")
    if args.reanalyze and not args.clean and (args.workspace / STATE_FILENAME).is_file():
        parser.error(
            "--reanalyze is for fresh runs only; omit it to resume the checkpointed "
            "analysis mode, or use a new --workspace or --clean to start fresh"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        task_data = load_and_validate_task_yaml(
            args.task,
            max_rounds_override=args.max_rounds,
        )
    except TaskSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    if args.reuse_analysis is not None and not Path(args.reuse_analysis).expanduser().is_dir():
        print(
            f"error: --reuse-analysis source is not a directory: {args.reuse_analysis}",
            file=sys.stderr,
        )
        sys.exit(2)
    # Resolve the projector's methodology skill once, before the run, so
    # it is told to load a skill this session actually has. Skipped (free)
    # when the stage is off.
    methodology = resolve_sol_methodology(sol_enabled(task_data))
    note = methodology.console_note()
    if note:
        print(note, file=sys.stderr)
    prompts = build_perf_optimize_prompts(
        include_slurm_environment=has_slurm_environment(task_data),
        remote_execution=task_data,
        campaign_name=args.workspace.resolve().name,
        approaches=task_data["optimize"]["approaches"],
        include_sol=sol_enabled(task_data),
        kernel_coverage=kernel_coverage(task_data),
        headroom_ledger=headroom_ledger(task_data),
        sol_methodology=methodology.name,
        include_disagg=has_disagg(task_data),
    )
    with PerfOptimizeWorkflow(
        workspace=args.workspace,
        clean=args.clean,
        prompts=prompts,
        max_rounds_override=args.max_rounds,
        reuse_analysis=args.reuse_analysis,
        reanalyze=args.reanalyze,
        sol_methodology=methodology,
    ) as workflow:
        prompt_dir = args.workspace / PROMPTS_DIRNAME
        dump_prompt_bundle(prompts, prompt_dir)
        workflow.run(args.task)


if __name__ == "__main__":
    main()
