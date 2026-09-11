---
name: perf-analyze
description: Launch and operate this repo's perf-analyze workflow to diagnose TensorRT-LLM serving performance without applying changes. Benchmarks configured operating points, captures nsys/ncu evidence, and explains the remaining gap using the current theoretical-best performance_model.yaml. Use when the user wants to analyze, profile or diagnose trtllm-serve throughput or latency, or says "run perf-analyze". Use perf-optimize to apply and verify optimizations.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Operating the perf-analyze workflow

`perf-analyze` is this repo's one-shot diagnosis pipeline for
`trtllm-serve`: benchmarker → projector → analyzer → reporter. It measures
configured operating points, derives an initial SOL estimate unless
`sol.enabled: false`, and profiles the load under nsys and bounded ncu.
The combined analyzer writes `profiler_report.md` for capture provenance
and quality, `analysis.md` for conclusions, and `performance_model.yaml`
for the current measured-to-theoretical comparison. The final
`<workspace>/performance_report.md` (+ `.html`) explains the remaining
performance gap and next actions. It never applies optimizations or
mutates the TRT-LLM checkout.

Do not hand-roll a serve/benchmark/profile loop when the user asks to
diagnose serving performance — drive this workflow instead.
Authoritative references (read them before answering detailed
questions):

- `<agent_flow>/workflows/perf_analyze/README.md` — pipeline contract,
  task schema, workspace layout, resume semantics.
- `<agent_flow>/workflows/perf_analyze/task.example.yaml` — fully
  commented task template.

The workflow itself now ships from TensorRT-LLM, under `agent-flow/`;
the paths below are relative to the INSTALLED `agent_flow` package. Print
its location with
`python -c "import agent_flow,pathlib;print(pathlib.Path(agent_flow.__file__).parent)"`,
or read them out of `<trtllm_repo_path>/agent-flow/agent_flow/`.

Use perf-analyze when the user only wants a bottleneck diagnosis.
Use `perf-optimize` (same repo) when they want changes applied and
verified.

## 1. Preflight — verify before launching

Run these checks; fix or ask only where noted.

1. **CLI installed**: `perf-analyze --help` works. If not:
   `pip install -e <trtllm_repo_path>/agent-flow`. The workflow drives
   the Claude Code backend, so the `claude` CLI must be installed and
   signed in; `CLAUDE_CODE_DEFAULT_MODEL` overrides the model if the
   user wants.
2. **Paths from the user**: `checkpoint_path` (model checkpoint dir) and
   `trtllm_repo_path` (TensorRT-LLM checkout) must exist — the CLI
   refuses to start otherwise. The same applies to
   `extra_llm_api_options` when set.
3. **No git requirements**: unlike perf-optimize, this workflow is
   read-only with respect to the TRT-LLM checkout — no branch, no
   commits, no reverts. A dirty worktree is fine.
4. **Install matches checkout**: the server runs the *installed*
   `tensorrt_llm`, while `benchmark_serving.py` and the profiling
   env-var names are read from `trtllm_repo_path`. They should be the
   same version — `python -c "import tensorrt_llm; print(tensorrt_llm.__file__)"`
   resolving into `trtllm_repo_path` (editable install) is the clean
   setup. A mismatch won't stop the run but can skew the analyzer's
   knob-verification grep; warn the user.
5. **Environment**: either the current node has GPUs (`nvidia-smi`
   works), or the task carries a `slurm-environment` block that routes
   the server + benchmark through a Slurm-launched container. On a
   login or head node with no local GPUs, bring up the node and
   container with your own site's recipe first. When
   `profile.methods` includes `nsys` (the default), `nsys` must be on
   PATH where the server runs — it is in the dev container;
   likewise `ncu` for the `ncu` method (also in the dev container — a
   missing binary or profiling permission degrades that run gracefully;
   on multi-rank serves expect *partial* per-kernel coverage — TRT-LLM's
   fixed 300 s executor hang-watchdog kills the server after a few
   replayed CUDA-graph launches, and the findings state the achieved
   coverage).
6. **Internal toolkit skills (affects depth, not launch)**: the
   default-on SOL projector wants `internal-perf-sol-analysis`. That is
   an `internal-`prefixed skill, so **open-source builds of the
   `trtllm-agent-toolkit` plugin strip it** while keeping
   `perf-analysis`. **The CLI checks this itself** at launch and prints
   one line when the skill is missing, so this is context for the user
   rather than something you must gate on. The run completes either way:
   without the SOL skill the projector falls back to `perf-analysis`,
   grounds the peaks from named sources rather than the skill's
   calculator, marks them as such, and writes no `sol_work/peaks.json` —
   so the analyzer skips the measured↔SOL correlation. Say so up front;
   if the user doesn't want the degraded stage, write
   `sol: {enabled: false}` to skip its wall-clock outright. Never work
   around a missing skill by having an agent recall hardware peaks.

## 2. Write task.yaml (if the user didn't provide one)

Copy `<agent_flow>/workflows/perf_analyze/task.example.yaml` into the
workspace-to-be and fill it in. Required: `checkpoint_path`,
`trtllm_repo_path`. Ask the user for anything they haven't stated rather
than inventing values:

- `benchmark`: the operating point(s). Defaults: `dataset_name: random`,
  ISL 1024 / OSL 128, `num_prompts: 200`, `concurrency: 64`; optional
  `request_rate` and `dataset_path`. `concurrency` is a single int
  (one operating point) **or a list of ints** — a list turns on
  Pareto-curve mode: one benchmark run per point over the same server,
  profiling at the largest point, and **Result** includes a measured
  Pareto curve (x = tok/s/user = 1000/mean_tpot_ms, y = tok/s/gpu =
  output_throughput/num_gpus). Benchmark time scales with the point
  count; in curve mode `num_prompts` may also be a **list** paired
  index-by-index with the concurrency list (each entry ≥ its point) so
  low-concurrency points run far fewer prompts — estimate
  `Σ num_prompts_i × OSL / agg_tok_s(c_i)` and keep it well inside the
  allocation walltime. Confirm the point(s) match the user's deployment
  shape — the whole diagnosis is anchored to them.
- `extra_llm_api_options`: server tuning YAML passed verbatim to
  `trtllm-serve --extra_llm_api_options` — the single place for all
  server knobs (parallelism, batch sizes, KV-cache fraction,
  CUDA-graph config, ...). Omit for server defaults. The server always
  runs the `pytorch` backend on `127.0.0.1:8000`.
- `profile`: `methods` (subset of `[nsys, ncu]`, default both) and
  `nsys_iter_range` (default `"100-150"`, the steady-state
  iteration window `TLLM_PROFILE_START_STOP` captures; the ncu deep
  dive arms on the same window via `--profile-from-start off`).
- `slurm-environment`: include only when the server + benchmark must
  run inside a Slurm-launched container; both `slurm_partition` and
  `docker_image` are then required.
- `sol`: the initial SOL projection runs by default. Set
  `sol: {enabled: false}` to skip the projector or use `gpu` as a hardware
  part-name hint. The projector follows `internal-perf-sol-analysis` when
  installed, otherwise `perf-analysis` with named hardware sources. It
  preserves its initial estimate and assumptions in `sol_projection.md`
  and writes `sol_work/peaks.json` when available. Missing hardware or
  latency inputs remain explicit.

The analyzer always produces `performance_model.yaml`, including when the
projector is disabled. It pairs measured and theoretical performance at
every benchmark point on the same workload, runtime, timing window and
statistic. The initial projection is provenance for this current model.
Kernel sums, decode critical paths and whole-run serving measurements
need an explicit derivation before comparison. The model distinguishes
physical limits, recoverable costs, scope constraints, measurement
limitations and unresolved residual. Unknown bounds remain explicit with
the next measurement needed. A profile at one concurrency cannot supply
measurements at another point. Convergence requires consistent evidence
explaining the residual across all focus points (all configured points
when no subset is set); all points remain represented in the model.

## 3. Launch (long-running — background it)

```bash
perf-analyze --task <path>/task.yaml --workspace workspace/perf-analyze/<model-name> \
    > <somewhere>/perf_analyze.log 2>&1
```

A run takes on the order of one to a few hours (server spin-up + one
benchmark, plus a profiled replay per profiling method). Launch it in
the background (`run_in_background` / `nohup`) with output captured to
a log file, then monitor.

- **Resume**: re-running the identical command resumes from
  `<workspace>/.perf_analyze_state.json` at the stage that was
  interrupted — Ctrl-C, crash, node loss are all safe. Pass the **same
  `--task` file** when resuming: stage gating reads the checkpointed
  workspace `task.yaml` while prompt selection reads the `--task` file,
  and they must agree about the `sol` / `slurm-environment` blocks.
- **Fresh start**: `--clean` wipes the checkpoint and managed outputs
  (`benchmark_results.md`, `sol_projection.md`, `performance_model.yaml`,
  `profiler_report.md`, `analysis.md`, `performance_report.md/.html`,
  `progress.yaml`); run artifacts (`serve.log`, result JSON,
  `*.nsys-rep`, `*.ncu-rep`) are left alone.
- A workspace holding non-empty outputs but **no checkpoint** refuses
  to start (`FileExistsError`) — pass `--clean` or pick a new
  workspace directory.

## 4. Monitor

Poll the workspace (and the launch log) rather than waiting silently:

- `progress.yaml` — append-only audit log; new entries mean it's alive.
- Stage deliverables appear in order: `benchmark_results.md` →
  `sol_projection.md` (unless `sol.enabled: false`) →
  `profiler_report.md`, `analysis.md` and `performance_model.yaml` →
  `performance_report.md` / `.html`.
- Run artifacts: `serve.log` (server health), the raw benchmark result
  JSON, `server_nsys.nsys-rep` + `nsys_stats.txt`,
  `server_ncu.ncu-rep` + `ncu_details.txt` / `ncu_raw.csv`,
  `perf_metrics.json`.

If it dies, read the tail of the launch log. A
`RuntimeError: ... left required output empty/missing` means a stage
ended its turn without writing its deliverable — the checkpoint is left
un-advanced, so re-running the same command retries that stage. Fix any
environment issue and re-run.

## 5. Wrap up

Use the four sections of `performance_report.md` / `.html`:

1. **Result** — measured performance and runtime/coverage provenance.
2. **Theoretical performance model** — current measured versus theoretical
   best per point, the common timing basis and essential assumptions.
3. **Gap analysis** — material remaining costs, evidence for each
   explanation, constraints and explicit unresolved residual.
4. **Next actions** — consequential findings and the
   highest-value optimization or measurement recommendations.

Keep commands, full configurations, kernel tables and detailed arithmetic
in linked artifacts. The initial `sol_projection.md` provides provenance;
headlines use the current model. Unavailable theory or missing evidence
cannot establish convergence. Model status is `open`, `converged`,
`measurement_limited`, `scope_limited` or `model_invalid`.

Point the user to the self-contained HTML report and its linked evidence.
If they want changes applied, perf-optimize can reuse this analysis.

## Pitfalls

A Slurm allocation is time-limited; when its walltime expires the node
is taken back, which also kills the agent-flow process. When that
happens, allocate a fresh node and launch the perf-analyze workflow
again with the same command — it resumes from `<workspace>/.perf_analyze_state.json`
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
