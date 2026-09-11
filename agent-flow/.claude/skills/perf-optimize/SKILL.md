---
name: perf-optimize
description: Launch and operate this repo's perf-optimize workflow to apply and verify TensorRT-LLM serving optimizations. Benchmarks configured operating points, maintains performance_model.yaml as the current theoretical-best model and convergence basis, captures nsys/ncu evidence, and evaluates roadmap items against measured gain. Use when the user wants to optimize or speed up trtllm-serve throughput or latency, or says "run perf-optimize". Use perf-analyze for diagnosis without applying changes.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Operating the perf-optimize workflow

`perf-optimize` is this repo's iterative optimization campaign for
`trtllm-serve`: benchmarker (baseline) → projector (a one-shot
analytical speed-of-light ceiling; on unless task.yaml sets
`sol.enabled: false`) → a fixed budget of rounds of
[(profiler, when fresh captures are needed) → analyzer (offline analysis
or replan-only) → (optimizer ⇄ evaluator) per roadmap item] → one
final-verification QA pass → final_analyzer → reporter. Every change is gated on code
quality, functionality, and measured gain vs the last accepted
measurement — the evaluator's verdict is three-way (APPROVE / REJECT
terminally / PUSH_BACK for a bounded retry), and every accept is
profiled under nsys (accept-evidence capture). No agent decides when to
stop: the loop runs `optimize.max_rounds` rounds unless the roadmap
runs out of actionable items with no outstanding measurement request,
or the optional improvement target is met.
These stop conditions do not establish convergence. The deliverable is
`<workspace>/optimization_report.md` (+ `.html`), organized around the
current `analysis/performance_model.yaml` from the latest completed
round or final reconciliation.

The profiler owns server lifecycle, nsys/ncu capture, exports, capture
coverage, and provenance, summarized in `profile/profiler_report.md`
alongside `profile_manifest.json`. The analyzer interprets saved evidence
and writes `analysis/analysis.md`, updates `performance_model.yaml`, and
authors the roadmap without launching a server or GPU profiler. The model
is required with or without the projector or kernel-coverage option, and
is updated on every turn, including reuse and replan-only turns. Capture
and analysis have separate checkpoints, so a failed analysis can resume
from a completed capture. Each successful full
analysis records its source capture in `analysis/analysis_manifest.yaml`.
When changes were accepted, independent QA is followed by a checkpointed
`final_analyzer` stage using the same offline analyzer. It reconciles the
final measurements and runtime against current theoretical bounds, writes
`final_verification/analysis/{analysis.md,performance_model.yaml}`, and
leaves the roadmap unchanged. It uses no GPUs; retrying it does not repeat
QA. Both outputs must be nonempty and the model must pass validation
before reporting. The reporter prefers this reconciled final model.
Use
`--reuse-analysis <dir> --reanalyze` to reinterpret existing captures in
round 1 of a fresh campaign;
plain `--reuse-analysis` plans from existing findings instead.

Do not hand-roll a serve/benchmark/tune loop when the user asks to
optimize serving performance — drive this workflow instead. Authoritative
references (read them before answering detailed questions):

- `<agent_flow>/workflows/perf_optimize/README.md` — contract, acceptance
  gate, workspace layout, git requirements.
- `<agent_flow>/workflows/perf_optimize/task.example.yaml` — fully commented
  task template.

Use perf-optimize when the user wants changes applied and verified.
The workflow itself now ships from TensorRT-LLM, under `agent-flow/`;
the paths below are relative to the INSTALLED `agent_flow` package. Print
its location with
`python -c "import agent_flow,pathlib;print(pathlib.Path(agent_flow.__file__).parent)"`,
or read them out of `<trtllm_repo_path>/agent-flow/agent_flow/`.

Use `perf-analyze` (same repo) when they only want a bottleneck diagnosis.

## 1. Preflight — verify before launching

Run these checks; fix or ask only where noted.

1. **CLI installed**: `perf-optimize --help` works. If not:
   `pip install -e <trtllm_repo_path>/agent-flow`. The workflow drives
   the Claude Code backend, so the `claude` CLI must be installed and
   signed in; `CLAUDE_CODE_DEFAULT_MODEL` overrides the model if the
   user wants.
2. **Paths from the user**: `checkpoint_path` (model checkpoint dir) and
   `trtllm_repo_path` (TensorRT-LLM checkout) must exist — the CLI
   refuses to start otherwise.
3. **Git safety (STOP if it fails)**: `trtllm_repo_path` must be a
   TensorRT-LLM git repo, and its worktree should be clean
   (`git -C <repo> status --porcelain` empty). The workflow mutates the
   checkout: rejected attempts are reverted with `git reset --hard` +
   `git clean -fd`, which destroys any uncommitted, unignored changes
   present when the run starts. If the worktree is dirty, do NOT launch —
   ask the user to commit/stash (or confirm the changes are disposable).
   Work happens on a fresh `perf-optimize/<workspace>-<timestamp>` branch
   off the current HEAD; one commit per accepted item.
4. **Editable install (affects scope, not launch)**: source-level
   (`approach: code`) items only take effect when
   `python -c "import tensorrt_llm; print(tensorrt_llm.__file__)"`
   resolves into `trtllm_repo_path`. If it doesn't, tell the user the run
   will be config-only (the agents detect this too) — still fine to run.
5. **Environment**: either the current node has GPUs (`nvidia-smi`
   works), or the task carries a `slurm-environment` block that routes
   the server + benchmark through a Slurm-launched container
   (`task.example.yaml:139`). On a login or head node with no local
   GPUs — the most common case — bring up the node and container with
   your own site's recipe first.
6. **Internal toolkit skills (affects depth, not launch)**: the
   default-on SOL projector wants `internal-perf-sol-analysis`. That is
   an `internal-`prefixed skill, so **open-source builds of the
   `trtllm-agent-toolkit` plugin strip it** while keeping
   `perf-analysis`. **The CLI checks this itself** at launch and prints
   one line when the skill is missing, so this is context for the user
   rather than something you must gate on. The campaign completes either
   way: without the SOL skill the projector falls back to
   `perf-analysis`, grounds the peaks from named sources rather than the
   skill's calculator, marks them as such, and writes no
   `sol_work/peaks.json` — so the analyzer skips the per-round
   measured↔SOL correlation, and every roadmap item's
   `expected_gain_pct` rests on a coarser ceiling. Say so up front; if
   the user doesn't want the degraded stage, write
   `sol: {enabled: false}` to skip its wall-clock outright. Never work
   around a missing skill by having an agent recall hardware peaks.
7. **Disaggregated deployment?** A `disagg:` block in `task.yaml` makes the
   workflow drive the checkout's own harness
   (`examples/disaggregated/slurm/benchmark/submit.py`) instead of launching
   `trtllm-serve`, so this host must be able to `sbatch` **and** see the
   cluster paths the harness config names. Everything else about the mode
   is documented on the `disagg` block in `task.example.yaml`.

## 2. Write task.yaml (if the user didn't provide one)

Copy `<agent_flow>/workflows/perf_optimize/task.example.yaml` into the
workspace-to-be and fill it in. Required: `checkpoint_path`,
`trtllm_repo_path`. Ask the user for anything they haven't stated rather
than inventing values:

- `benchmark`: the operating point(s) every measurement replays
  (ISL/OSL/concurrency/num_prompts). Defaults exist, but confirm they
  match the user's deployment shape. `concurrency` is a single int or a
  list of ints — a list turns on Pareto-curve mode: every measurement
  (baseline, each evaluator attempt, each QA round) runs once per point,
  and the evaluator applies the Pareto gate (mean per-point gain must
  pass the thresholds AND no point may regress beyond the noise floor);
  the report includes the per-point comparison in **Result**, with a
  Pareto chart when useful (x = tok/s/user, y = tok/s/gpu). Benchmark cost multiplies by the point count — keep
  the list short (~3-5 points), or pair it with a `num_prompts` **list**
  (same length, each entry ≥ its point) so low-concurrency points run
  far fewer prompts than high-concurrency ones; that is what makes a
  wide curve affordable.
- **Size the per-measurement cost before launching** — this is the #1
  preflight trap. Estimate one full measurement as
  `Σ_points (num_prompts_i × OSL / expected_agg_tok_s(c_i))` and keep it
  well under the allocation walltime minus ~30 min of env build (the
  campaign replays it for the baseline, every evaluator attempt, and
  the final verification; a measurement that cannot finish inside one
  allocation restarts forever and the campaign never completes).
  Example: 9 points [1..256] × 4096 prompts × OSL 6144 ≈ 30-70 h per
  measurement — infeasible; scaling num_prompts [8..1024] with the
  points brings it to ~1 h.
- `optimize.target_metric` (default `output_throughput`; latency keys
  like `median_ttft_ms`/`median_tpot_ms` also work — gains are
  normalized so positive = better) and optional
  `optimize.target_improvement_pct` (orchestrator-enforced early stop).
- `optimize` budgets — defaults `max_rounds: 5`,
  `max_items_per_round: 3`, `max_attempts_per_item: 3`,
  `accept_fraction: 0.5`, `noise_floor_pct: 1.0` — are sensible. The
  loop **runs the full round budget** (it only ends early when the
  target is met, or an analyzer turn finds no actionable item and has no
  outstanding measurement request), so
  `max_rounds × max_items_per_round`
  is what bounds the items a campaign can attempt — 15 with the defaults.
  `optimize.item_execution` defaults to `parallel`; set it to `serial`
  to apply approved items directly in order from the latest campaign state.
  Two things worth telling the user when sizing a run:
  - **Rounds are not equally expensive.** A round pays for a profile
    after an accept, after a reverted code attempt may have changed
    gitignored build output, or when an older checkpoint cannot prove its
    profile is current. The profiler captures first and the analyzer
    interprets those saved artifacts. A targeted measurement request also
    schedules fresh profiling while rounds remain, even with no pending
    item. Otherwise it opens replan-only —
    the analyzer plans from the standing profile and the round's verdicts without
    touching the GPU. On a config-only plateau `max_rounds` costs almost
    nothing beyond the per-attempt benchmarks; a productive or
    code-mutating campaign pays more profiling turns.
  - The per-item cost driver is the **evaluator benchmark**, one per
    attempt, which no mode avoids. `noise_floor_pct` and `approaches`
    are what keep the item list short.
- `optimize.approaches` (default `[config, code]`): set `[code]` when the
  user wants genuine source-level optimizations only (no tuning-YAML knob
  changes — attempts that touch the tuning file are auto-rejected;
  requires the editable install from preflight step 4), or `[config]`
  when the TRT-LLM checkout must not be modified at all.
- `profile`: `methods` (subset of `[nsys, ncu]`, default both — what a
  profiling round captures: nsys timeline + a bounded ncu per-kernel
  deep dive on the top nsys kernels, interpreted with the
  `perf-nsight-compute-analysis` skill; drop
  entries to trim the cost of the rounds that pay it) and
  `nsys_iter_range` (default `"100-150"`).
- `profile.kernel_coverage`: an empty mapping enables defaults
  `min_share_pct: 0.5`, `coverage_target_pct: 95`. It requires nsys+ncu
  coverage of every kernel above the share threshold and evidence-backed
  answers to elimination, faster execution, fusion and overlap in each
  round's `analysis/kernel_ledger.yaml`. The ledger is validated after
  every analyzer turn, including reuse, re-analysis and replan-only turns.
  Replan uses standing measurements without another capture. Kernel and
  region models support the central `performance_model.yaml`; full tables
  remain linked artifacts rather than another report section. GPU-time
  shares convert through `coverage.gpu_busy_pct` before comparing with
  wall-clock gain thresholds.
- `accuracy`: include only if the user has an eval command they want the
  final verification to run at campaign end; omit the block otherwise.
- `extra_llm_api_options`: starting server tuning YAML, if they have one.
  It seeds `<workspace>/tuning/extra_llm_api_options.yaml`, which the
  optimizer then evolves — the workflow always passes the workspace copy
  to `trtllm-serve`.
- `sol`: the initial SOL projection runs by default. Set
  `sol: {enabled: false}` to skip the projector or use `gpu` as a hardware
  part-name hint. The projector writes `sol_projection.md` and available
  `sol_work/peaks.json` inputs once. These preserve the initial derivation;
  subsequent comparisons and ranking use the analyzer's current
  `performance_model.yaml`. The projector follows
  `internal-perf-sol-analysis` when installed, otherwise `perf-analysis`
  with named hardware sources. Missing peaks or latency measurements
  remain explicit. Disabling the projector does not disable the model.

The current model pairs measured and theoretical performance at every
benchmark point on the same workload, runtime, timing window and statistic.
It explains physical costs, recoverable implementation costs, scope
constraints, measurement limitations and unresolved residual separately.
Unknown theory stays explicit with a next measurement. Kernel sums,
decode critical paths and serving throughput need an explicit derivation
before comparison. A failed experiment cannot alone relax the theoretical
bound, and a below-noise-floor gain cannot prove a hardware limit.
Convergence concerns all focus points (all points when no subset is set),
while every configured point remains represented in the model.

## 3. Launch (long-running — background it)

```bash
perf-optimize --task <path>/task.yaml --workspace workspace/perf-optimize/<model-name> \
    > <somewhere>/perf_optimize.log 2>&1
```

A campaign runs for hours (each round: a full benchmark per attempt + a
profiled replay per accept, plus one re-profile when the standing runtime
evidence became stale; plus one final-verification benchmark at the end). Launch it in the background
(`run_in_background` / `nohup`) with output captured to a log file, then
monitor.

- **Resume**: re-running the identical command resumes from
  `<workspace>/.perf_optimize_state.json` — interruption (Ctrl-C, crash,
  node loss) is safe. If the launch used `--reanalyze`, omit that flag
  when resuming; the checkpoint preserves the mode. A completed profiler
  stage is not repeated when only the analyzer needs to retry.
- **Fresh start**: `--clean` wipes the workspace's managed state but
  never touches the TRT-LLM checkout (abandoned `perf-optimize/*`
  branches are left for inspection).
- `--max-rounds N` overrides the round budget on a fresh run only;
  ignored on resume.
- `--reuse-analysis <dir>` starts a fresh run **at the optimize stage**
  by importing a previous `perf-analyze` workspace or `perf-optimize`
  campaign workspace: its baseline report (+ result JSONs), SOL
  projection (+ `sol_work/`), current model, and newest analysis (+ traces,
  `kernel_ledger.yaml`) are copied in, the benchmarker/projector are
  skipped, and round 1's analyzer plans from the imported evidence
  without launching a server or a profiler. It writes the local current
  model and `analysis.md`; legacy `profile_findings.md` remains readable
  as import provenance. The latest corrected source model is kept as
  `reused_analysis/prior_performance_model.yaml`, preserving revisions
  and source measurement identity. It is read-only prior evidence; round 1
  must still write a new current model. Rounds 2+ profile normally.
  Reach for this when the user has *just* run `perf-analyze` on the same
  deployment, or is starting a follow-up campaign on a machine whose
  baseline has not changed — it saves the two most expensive stages.
  Check before proposing it: the imported baseline is what every gain is
  measured against, so the source run must describe the **same** model,
  checkpoint, hardware, parallel mapping and operating point. When in
  doubt, don't reuse the baseline (the import is per-artifact — point at
  a source with only findings, or just run normally). A source
  `roadmap.yaml` is kept as read-only prior art in
  `reused_analysis/prior_roadmap.yaml`, never as this campaign's ledger;
  `reused_analysis/manifest.md` records the provenance and the report
  repeats it. Fresh runs only — ignored on resume.
- `--reuse-analysis <dir> --reanalyze` imports saved captures and runs
  round 1's analyzer offline to regenerate its derivations, findings,
  ledgers, and roadmap. Use it when methodology, taxonomy, or hypotheses
  changed and the existing captures still cover the required evidence.
  Capture discovery supports both the separate `profile/` layout and
  legacy layouts, including a completed capture whose original analysis
  failed. It requires raw evidence; findings alone are insufficient.
  This is a fresh-campaign option, so an existing checkpoint rejects
  `--reanalyze` unless `--clean` is also supplied. Resume without the
  flag. Later optimization and profiling rounds proceed normally: the
  option does not make the whole campaign offline, and imported captures
  do not prove that this campaign's local runtime is current.

## 4. Monitor

Poll the workspace (and the launch log) rather than waiting silently:

- `progress.yaml` — append-only audit log; new entries mean it's alive.
- `roadmap.yaml` — the ranked plan; watch item `status` and measured
  gains vs `expected_gain_pct`; `current_best` tracks the last accepted
  measurement.
- The latest `rounds/round_<n>/analysis/performance_model.yaml` — the
  current measured-to-theoretical comparison, model revisions,
  evidence-backed costs and unresolved residual per point. Status is `open`, `converged`,
  `measurement_limited`, `scope_limited` or `model_invalid`. Outstanding
  targeted measurement requests persist across rejected optimization
  items and cause another profiler turn while rounds remain.
- `baseline/benchmark_results.md` (and `sol_projection.md` right after
  it unless `sol.enabled: false`), then per-round
  `rounds/round_<n>/` (`profile/profiler_report.md`,
  `profile/profile_manifest.json` and raw captures,
  `analysis/analysis_manifest.yaml` linking the source capture,
  `analysis/analysis.md` and offline derivations/ledgers,
  `item_<j>_<id>/attempt_<k>/` — accepted attempts also grow a
  `profile/` nsys capture), and at the end
  `final_verification/verification_report.md` followed by
  `final_verification/analysis/{analysis.md,performance_model.yaml}` when
  changes were accepted. A round with current evidence and no outstanding
  capture request updates analysis and its model without new traces.
- `git -C <trtllm_repo_path> log --oneline` on the `perf-optimize/*`
  branch — one commit per accepted item.

If it dies, read the tail of the launch log, fix the environment issue,
and re-run the command to resume (omit `--reanalyze` on resume).

## 5. Wrap up

Use the four sections of `optimization_report.md` / `.html`:

1. **Result** — final verified gain, benchmark provenance, workflow stop
   reason and model convergence status.
2. **Theoretical performance model** — current measured versus theoretical
   best at every point, the common timing basis and essential assumptions.
3. **Gap analysis** — the largest remaining costs, evidence for physical
   or scope limits, measurement limitations and explicit unresolved residual.
4. **Changes and next actions** — consequential accepted changes or failed
   experiments and the highest-value next optimization or measurement.

Analyzer `analysis.md` files, including final reconciliation, use the
same first three sections and end with **Next actions**.

Keep commands, full configurations, kernel tables, detailed arithmetic and
attempt histories in linked artifacts. Use the current model for the
remaining-gap headline, preferring
`final_verification/analysis/performance_model.yaml` after reconciliation;
the initial `sol_projection.md` is provenance. Only a capture matching the final accepted runtime can explain final-state
behavior. The final analyzer must reconcile QA measurements before the
reporter runs. Any missing evidence or unresolved incompatibility stays
explicit in the final model and prevents an unsupported convergence claim.
An empty roadmap or spent budget alone cannot establish convergence.

Point the user at the accepted branch and
`tuning/extra_llm_api_options.accepted.yaml` for the resulting changes.
The workflow never pushes.

## Pitfalls

A Slurm allocation is time-limited; when its walltime expires the node
is taken back, which also kills the agent-flow process. When that
happens, allocate a fresh node and launch the perf-optimize workflow
again with the same command — it resumes from `<workspace>/.perf_optimize_state.json`
at the interrupted stage.

The walltime ceiling is usually enforced by the *partition*, not by the
QoS list your account carries: a partition can set `DenyQos=...`, so a
long QoS that `sacctmgr show assoc` happily lists is still rejected at
submit with "Invalid qos specification" and only a short QoS passes.
Read the partition's own limits (`scontrol show partition <partition>`)
rather than your association, and size the benchmark block to the
window you actually get (see the per-measurement cost note above).

For unattended multi-window campaigns, don't rely on the driving
session staying alive to resubmit: run a small nohup'd **keeper loop**
on the launch host that every ~5 min resubmits the sbatch iff the
workflow state JSON has `done: false` and the queue has no job of that
name (double-check an empty queue reading ~60 s apart before
resubmitting; guard with a pidfile and a stop-sentinel file). A
session restart otherwise turns a walltime kill into a silent
hours-long stall.

## Improvement suggestions

This workflow and this skill are under active development. If driving
a run surfaces an issue — a bug or crash in agent-flow, a misleading
log or report, a preflight check this skill is missing, a stale or
wrong instruction — don't just work around it silently. Note it while
it's concrete, and when reporting results to the user, include a short
list of workflow improvement suggestions: what went wrong, where
(file / step), and the fix you'd propose. If a fix is a small edit to
this skill or the workflow docs, offer to apply it.
