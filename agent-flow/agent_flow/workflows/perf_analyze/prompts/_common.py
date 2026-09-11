"""Shared prose blocks for the perf-analyze role prompts.

The exact ``trtllm-serve`` / ``benchmark_serving.py`` / ``nsys`` command
knowledge lives here (not in optional agent-toolkit skills) so the
workflow is self-contained. The benchmarker and analyzer prompts import
the blocks they need; the ``EXECUTION_SLURM_BOOTSTRAP`` block is appended
only when ``task.yaml`` carries a ``slurm-environment`` section, and the
``SOL_ANALYZER_CONTEXT`` / ``SOL_REPORTER_GUIDANCE`` blocks are
appended (to the analyzer / reporter) only when the projector stage is
enabled — the default, unless ``task.yaml`` sets ``sol.enabled: false``.

Three blocks point at agent-toolkit skills: ``CASEBOOK_CONSULTATION``
tells both serving roles to load ``perf-optimization-casebook`` as
read-only reference so their analysis is grounded in known TRT-LLM
performance precedents; ``PROFILING_RUNS_REFERENCE`` has the analyzer
load a methodology skill per profiler run, unprompted —
``internal-perf-nsight-system-analysis`` to read the Run A timeline (per-iteration
anchor, the busy/idle rungs, and what caused each compute-absent
stretch) and ``perf-nsight-compute-analysis`` to capture and interpret
the Run B ncu per-kernel deep dive; and the
projector's own prompt (in ``projector.py``) builds on
``internal-perf-sol-analysis`` as its projection methodology (the
analyzer loads the same skill for the measured↔SOL correlation when the
projector stage is enabled). All are written to degrade gracefully when
the skill is not installed, so they do not turn the workflow into a
hard dependency on the toolkit. ``internal-perf-sol-analysis`` carries
the ``internal-`` prefix, so open-source builds of the toolkit strip it
while keeping ``perf-analysis``; which of the two this session has is
resolved in Python before the stage runs (see
``sol_methodology.resolve_sol_methodology``), and only the fallback case
appends ``SOL_METHODOLOGY_FALLBACK`` to the projector's prompt.

The role-neutral blocks shared with perf-optimize live here rather than
inline in the role modules so both workflows compose the same
single-sourced text: the projector methodology
(``SOL_PROJECTOR_METHODOLOGY`` / ``SOL_PROJECTOR_INTERNAL_KNOWLEDGE``),
the analyzer's findings contract (``PROFILE_FINDINGS_CONTRACT``), and
the measured↔SOL correlation recipe (``SOL_CORRELATION_METHOD``) —
perf-optimize's analyzer is this workflow's analyzer plus the roadmap
machinery, and its prompts compose these same fragments.
"""

from pathlib import Path
from typing import Sequence

# The starting taxonomy the nsys-timeline pipeline classifies kernels with.
# The skill ships a generic template shaped for *training* frameworks
# (Transformer Engine, Adam, cuDNN convolutions); almost nothing in a
# TRT-LLM decode step matches it, and with no call-stack correlation
# available the regexes are the only classifier there is — an unmatched
# kernel lands in `uncategorized` and drags the Step 5 mode decision with
# it. This file is the same shape, written against TRT-LLM's own kernel
# names. It is a starting point, not a golden file: the analyzer still
# owes the skill's verify-and-extend loop per workload.
TRTLLM_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "assets" / "taxonomy_trtllm.json"

# --------------------------------------------------------------------------- #
# Server lifecycle (shared by benchmarker + analyzer)
# --------------------------------------------------------------------------- #

_SERVER_LIFECYCLE_TEMPLATE = """\
## Running `trtllm-serve` (local default)

Use `Bash` to complete launch, readiness, workload, teardown and reporting
in one turn. Wait with foreground polling; repeat the poll if loading needs
more time. Ending the turn does not schedule another invocation.
Run one server at a time and verify its identity before measuring.

1. **Check port 8000 before launch.** An interrupted stage may leave a
   detached server serving an old checkpoint/config:
   ```bash
   ss -ltn 'sport = :8000' | grep -q LISTEN && {
     echo "FATAL: port 8000 already in use — a stale server is still up"
     ss -ltnp 'sport = :8000'   # shows the owning pid/cmd
     exit 1
   }
   ```
   Identify and tear down the stale server by PID, then confirm the port
   is free. Do not switch ports. If `ss` is unavailable, use
   `lsof -i :8000` or `curl -fsS http://127.0.0.1:8000/health`;
   a pre-launch health response means a stale server.

2. **Launch detached**, saving logs and PID. `setsid` and `< /dev/null`
   preserve the process group across separate `Bash` calls:
   ```bash
   cd <trtllm_repo_path>
   setsid trtllm-serve <checkpoint_path> \\
       --backend pytorch \\
       --host 127.0.0.1 --port 8000 \\
       <serve_config_flag> \\
       > <workspace>/serve.log 2>&1 < /dev/null &
   echo $! > <workspace>/serve.pid
   ```
   <serve_config_policy>

3. **Poll readiness in the foreground.** Check liveness first and verify
   the listener belongs to the recorded process group before declaring READY:
   ```bash
   PID=$(cat <workspace>/serve.pid)
   # setsid made the server a process-group leader, so its PGID == its PID
   # and every worker it forks inherits that PGID: the listener on :8000 is
   # ours only if its process group is $PID.
   owns_port() {
     local owner
     owner=$(ss -ltnp 'sport = :8000' 2>/dev/null \\
             | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
     [ -n "$owner" ] || return 2          # could not resolve — do NOT assume ours
     [ "$(ps -o pgid= -p "$owner" 2>/dev/null | tr -d ' ')" = "$PID" ]
   }
   for i in $(seq 1 360); do
     kill -0 "$PID" 2>/dev/null || { echo "server exited"; tail -40 <workspace>/serve.log; break; }
     if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
       owns_port \\
         && { echo READY; break; } \\
         || { echo "FATAL: :8000 answered but is not owned by PID $PID"; \\
              ss -ltnp 'sport = :8000'; break; }
     fi
     sleep 5
   done
   ```
   <startup_failure_policy>

   An unverified or foreign port owner blocks measurement. Resolve it
   with `ss -ltnp` / `lsof`, tear down the stale server, and relaunch.

4. **Always tear down**, including after failure. Signal the recorded PID
   and process group, escalating to SIGKILL:
   ```bash
   PID=$(cat <workspace>/serve.pid)
   # Negative PID signals the whole process group (setsid made the server
   # a group leader), reaping its MPI workers / child procs with it.
   kill -TERM -"$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
   for i in $(seq 1 24); do            # up to ~120s for a graceful exit
     kill -0 "$PID" 2>/dev/null || break
     sleep 5
   done
   kill -0 "$PID" 2>/dev/null && { kill -KILL -"$PID" 2>/dev/null; kill -KILL "$PID" 2>/dev/null; } || true
   ```
   Confirm memory is freed with `nvidia-smi` and port 8000 has no LISTEN
   row before relaunch. Find surviving MPI daemons (`orted`) or workers
   with read-only `ps`/`pgrep` and reap their exact PIDs.
   Never `pkill -f 'trtllm-serve'`: it can match the agent or teardown shell.
   If a straggler requires pattern matching, include the checkpoint path
   and verify the match with read-only `pgrep` before killing.
"""


def build_server_lifecycle(
    active_tuning_config: bool = False, *, allow_config_changes: bool = False
) -> str:
    """Render the shared lifecycle with the workflow's configuration policy."""
    if active_tuning_config:
        config_flag = "--extra_llm_api_options <active tuning config>"
        config_policy = (
            "Use the exact active tuning config named in the turn instructions; "
            "see *The active tuning config*."
        )
        failure_policy = (
            "If startup fails, diagnose `serve.log` and report the blocker. "
            "Do not change the read-only tuning config or benchmark an unready server."
        )
        if allow_config_changes:
            failure_policy = (
                "If startup fails, diagnose `serve.log`. Correct only defects within "
                "your assigned item or candidate-combination scope; never change "
                "unrelated performance knobs to make startup succeed. Report other "
                "blockers and do not benchmark an unready server."
            )
    else:
        config_flag = "[--extra_llm_api_options <extra_llm_api_options>]"
        config_policy = (
            "Pass `--extra_llm_api_options <path>` **only when** `task.yaml` sets the "
            "top-level `extra_llm_api_options` key; omit the flag otherwise. "
            "All other server tuning lives in that YAML."
        )
        failure_policy = (
            "If startup fails, read `serve.log`, fix the cause (OOM: lower "
            "`kv_cache_config.free_gpu_memory_fraction` or `max_batch_size`; "
            "bad option: correct it), and retry. Do not benchmark an unready server."
        )
    return (
        _SERVER_LIFECYCLE_TEMPLATE.replace("<serve_config_flag>", config_flag)
        .replace("<serve_config_policy>", config_policy)
        .replace("<startup_failure_policy>", failure_policy)
    )


SERVER_LIFECYCLE = build_server_lifecycle()


SERVE_FLAGS_REFERENCE = """\
### Server configuration

`trtllm-serve` always runs with `--backend pytorch` on `127.0.0.1:8000`.
The model checkpoint to serve is the top-level `checkpoint_path` (the
positional `model` argument).

**All** other server tuning lives in one optional YAML, passed verbatim as
`--extra_llm_api_options <path>` when `task.yaml` sets the top-level
`extra_llm_api_options` key (omit the flag when it is absent). Those keys
map to LLM API fields, e.g.:

| `extra_llm_api_options` key                  | tunes                          |
| -------------------------------------------- | ------------------------------ |
| `tensor_parallel_size`                       | tensor parallelism (TP)        |
| `pipeline_parallel_size`                     | pipeline parallelism (PP)      |
| `moe_expert_parallel_size`                   | expert parallelism (EP)        |
| `max_batch_size`                             | max batch size                 |
| `max_num_tokens`                             | max num tokens                 |
| `kv_cache_config.free_gpu_memory_fraction`   | KV-cache memory fraction       |

Read the `extra_llm_api_options` YAML (when present) to learn the parallel
sizes and other knobs actually in effect. Check `trtllm-serve --help` and
the LLM API reference in `<trtllm_repo_path>` if unsure of a field name.
"""


_BENCHMARK_FLAGS_TEMPLATE = """\
## Running the benchmark — `benchmark_serving.py`

Run `tensorrt_llm/serve/scripts/benchmark_serving.py` as a module. Fill
placeholders from `task.yaml` and the workspace; keep other flags as shown.

```bash
cd <trtllm_repo_path>
python -m tensorrt_llm.serve.scripts.benchmark_serving \\
    --model <checkpoint_path> \\
    --tokenizer <checkpoint_path> \\
    --trust-remote-code \\
    --backend openai \\
    --host 127.0.0.1 --port 8000 \\
    --dataset-name <benchmark.dataset_name> \\
    --random-input-len <benchmark.random_input_len> \\
    --random-output-len <benchmark.random_output_len> \\
    --random-ids \\
    --tokenize-on-client \\
    --num-prompts <the point's num_prompts — see below> \\
    --max-concurrency <one value from benchmark.concurrency> \\
    [--request-rate <benchmark.request_rate>] \\
    [--dataset-path <benchmark.dataset_path>] \\
    --ignore-eos \\
    --no-test-input \\
    --percentile-metrics ttft,tpot,itl,e2el \\
    --metric-percentiles 90,99 \\
    --save-result --save-detailed --result-dir <workspace>
```

<benchmark_point_policy>
- **Per-point `--num-prompts`:** `benchmark.num_prompts` is a single
  integer (use it at every point) or a list paired index-by-index with
  `benchmark.concurrency`. Use the paired entry; keep ISL / OSL fixed.
- **Results:** create `--result-dir` before each run. In curve mode use
  `<artifact dir>/concurrency_<c>`; scalar runs use `<artifact dir>`.
  Treat missing result JSON as a failed run and read metrics from that
  JSON, including the per-request rows written by `--save-detailed`.
- Keep `--model` and `--tokenizer` at `checkpoint_path` with
  `--trust-remote-code`. `--ignore-eos` fixes the requested OSL.
- For `random`, use `--random-ids` with `--tokenize-on-client` to send
  exact token-ID lists without a ShareGPT download or retokenization.
  Verify `total_input_tokens == num_prompts × ISL` in the result JSON.
  For `sharegpt`/`hf`, replace `--random-ids` with `--dataset-path` and
  the dataset's required flags; retain `--tokenize-on-client`.
- Keep `--no-test-input`: a test prompt advances the server iteration
  counter before the real load and shifts the profiling window.
- Record throughput and TTFT / TPOT / ITL / E2EL from the result JSON;
  retain the requested mean / median / p90 / p99 metrics.
"""


_BENCHMARK_SWEEP_POLICY = """\
- **One run per concurrency point.** A scalar `benchmark.concurrency`
  runs once. For a list, launch one server and measure every configured
  point sequentially in ascending order, then tear down. Do not relaunch
  between points, run points in parallel, or change the point list/ISL/OSL.
"""


def build_benchmark_flags_reference(point_policy: str | None = None) -> str:
    """Render canonical load flags with either a scoring or profiling point policy."""
    return _BENCHMARK_FLAGS_TEMPLATE.replace(
        "<benchmark_point_policy>",
        _BENCHMARK_SWEEP_POLICY if point_policy is None else point_policy,
    )


BENCHMARK_FLAGS_REFERENCE = build_benchmark_flags_reference()


DERIVED_METRICS_REFERENCE = """\
## Derived per-user / per-GPU metrics (the Pareto axes)

Two derived metrics accompany every benchmark measurement, computed from
the result JSON plus the serving world size:

- `tok/s/user = 1000 / mean_tpot_ms` — per-user decode speed
  (interactivity), from that run's `mean_tpot_ms`.
- `tok/s/gpu = output_throughput / num_gpus` — per-GPU output
  throughput.
- `num_gpus` is the serving world size: the product of the parallel
  sizes actually in effect (`tensor_parallel_size` ×
  `pipeline_parallel_size` × `moe_expert_parallel_size` where they
  multiply the GPU count, each defaulting to 1) read from the
  `extra_llm_api_options` YAML in effect, cross-checked against
  `nvidia-smi` (the GPUs actually holding server memory). Record the
  value **and how you determined it** next to the metrics.

In Pareto-curve mode (`benchmark.concurrency` is a list) every measuring
report must include this **curve summary table**, one row per
concurrency point in ascending order (add the workflow's target metric
as an extra column when your instructions name one other than
`output_throughput`):

| concurrency | output_throughput (tok/s) | tok/s/user | tok/s/gpu | mean TPOT (ms) | mean TTFT (ms) |
|---|---|---|---|---|---|

The Pareto curve plots **x = tok/s/user, y = tok/s/gpu**, one point per
concurrency. `tok/s/user` and `tok/s/gpu` are **reporting / Pareto
axes** — any acceptance gate or target-metric comparison runs on the
result-JSON metric named in your instructions (e.g.
`output_throughput`), never on these derived values.
"""


# --------------------------------------------------------------------------- #
# Profiling references (shared by both workflows' analyzers)
# --------------------------------------------------------------------------- #

PROFILING_KNOB_VERIFICATION = """\
## Verify the profiling knobs first (verify before asserting)

Use `rg` via `Bash` in `trtllm_repo_path` before profiling:
- Check `tensorrt_llm/_torch/pyexecutor/py_executor.py` for
  `TLLM_PROFILE_START_STOP` and its iteration-window format (e.g. `100-150`).
- Check `tensorrt_llm/serve/openai_server.py` before assuming a
  `/start_profile` endpoint exists. Without it, capture is driven by
  the server env var; the benchmark client's `--profile` is a no-op.
- If a required knob is absent, skip that profiler and record the
  reason in `analysis.md`.
"""

_PROFILING_RUNS_BEFORE_TARGETING = f"""\
## Run A — Nsight Systems (GPU timeline)

Capture the timing trace, then analyze it with the
`internal-perf-nsight-system-analysis` skill in step 5 whenever nsys runs,
without waiting to be asked. Analysis costs no extra server launch.
Use the server lifecycle below for readiness and PID/process-group
teardown on every pass, including failures.

1. Relaunch `trtllm-serve` with the Benchmarker’s flags and serving config.
   Start from this canonical invocation; change only
   placeholders, the explicit requested-point/phase overrides, and the
   compatibility/fallback options below:
   ```bash
   cd <trtllm_repo_path>
   setsid env TLLM_PROFILE_START_STOP="<profile.nsys_iter_range>" \\
   nsys profile \\
       -o <workspace>/server_nsys -f true \\
       -t 'cuda,nvtx,python-gil' \\
       -c cudaProfilerApi --capture-range-end=stop \\
       --cuda-graph-trace node \\
       -e TLLM_NVTX_DEBUG=1 \\
       --trace-fork-before-exec=true \\
       trtllm-serve <checkpoint_path> ...same trtllm-serve flags... \\
       > <workspace>/serve.log 2>&1 < /dev/null &
   echo $! > <workspace>/serve.pid
   ```
   - `TLLM_PROFILE_START_STOP` counts server forward iterations; its
     endpoints call `cudaProfilerStart/Stop`. Keep
     `--capture-range-end=stop` so collection stops without terminating
     the serving engine (`stop-shutdown` can crash it before report export).
   - Default to the configured `profile.nsys_iter_range` and a usable
     steady-state decode window. When the turn explicitly requests a
     different operating point or phase to resolve missing model evidence,
     the Profiler (or combined capture stage) may select a different
     `TLLM_PROFILE_START_STOP` window to capture that request, including
     early prefill or mixed prefill/decode iterations. Keep task and serving
     config read-only. Record the request, configured and effective windows,
     reason, exact command and observed phase in `profiler_report.md` and
     the capture manifest when present. A requested window is not proof
     that the intended phase was captured; verify its trace/phase markers.
     These overrides authorize capture only; the offline Analyzer requests
     evidence and never launches or adjusts a capture itself.
   - NVTX, Python-GIL tracing and graph-node expansion provide attribution;
     `--trace-fork-before-exec=true` follows fork-before-exec children.
     See the multi-GPU topology limits below before launching.
   - Node tracing can hang graph-heavy servers. If the first attempt
     cannot finish and finalize a report after the bounded teardown in
     step 3, retry with `--cuda-graph-trace graph`, not identical flags.
     A missing report while the server is still running does not itself
     prove a failed capture. Record the fallback’s graph-level granularity.
   - If the installed profiler rejects a flag, drop only that flag and
     record it. Skip the pass if dropping it would invalidate capture.
2. Poll readiness, then replay the canonical benchmark at the effective
   profiling operating point, with the same load and `--no-test-input`.
   The skipped test prompt would advance the iteration counter before
   concurrent load arrives. Keep the requested point's configured paired
   `num_prompts` (the scalar value when not a list).
   If that load cannot reach `<stop>`, lower and document the iteration
   window. With no explicit phase request, require a usable steady-state
   decode window or report profiling unavailable. With an explicit phase
   request, judge coverage against that requested phase and operating point;
   early-prefill evidence is not invalid solely because it lacks steady-state
   decode. Record any uncaptured request as missing evidence. Verify
   `serve.log` contains `Profiling started at iteration <start>` and
   `... stopped at iteration <stop>`.

   For per-request prefill/decode data, pass `--save-request-time-breakdown`
   and retain `*perf_metrics*.json` only when `/perf_metrics` is enabled
   through `return_perf_metrics: true` in the current server config.
   Otherwise omit the flag and note the unavailable breakdown; do not
   edit or copy the serving config to enable metrics.
3. Tear down the server so nsys flushes `server_nsys.nsys-rep`. Allow a
   bounded finalization interval before escalating signals. If the report
   has not finalized, send `SIGINT` to the recorded profiler PID and wait
   for it to write the report before killing harder.
4. Produce the kernel table:
   ```bash
   nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_trace \\
       <workspace>/server_nsys.nsys-rep > <workspace>/nsys_stats.txt
   ```
   Extract kernel names, durations and shares. Busy/idle and gap attribution
   come from step 5, not sums of the kernel table.
5. **Decompose the timeline with the `internal-perf-nsight-system-analysis`
   skill.** Load it via the `Skill` tool, using
   `trtllm-agent-toolkit:internal-perf-nsight-system-analysis` if the bare
   name is not found. Its announced base directory is `<skill_dir>`;
   follow its methodology and the findings contract below.
   ```bash
   nsys export --type sqlite \\
       -o <workspace>/server_nsys.sqlite \\
       <workspace>/server_nsys.nsys-rep
   cp {TRTLLM_TAXONOMY_PATH} <workspace>/taxonomy.json
   python <skill_dir>/scripts/run_all.py \\
       --profile 0=<workspace>/server_nsys.sqlite \\
       --taxonomy <workspace>/taxonomy.json \\
       --out <workspace>/nsys_analysis
   ```
   - Use the TRT-LLM taxonomy above, not the skill's
     `references/taxonomy_template.json`, shaped for training frameworks.
   - One captured rank runs directly with a single `--profile` and no
     `--representative`. For several ranks, this campaign has no user to
     ask: survey every captured rank with no `--representative`; it
     stops with exit 2, which is the expected outcome. For example:
     ```bash
     python <skill_dir>/scripts/run_all.py \\
         --profile 0=<workspace>/server_nsys_rank0.sqlite \\
         --profile 4=<workspace>/server_nsys_rank4.sqlite \\
         --taxonomy <workspace>/taxonomy.json --out <workspace>/nsys_analysis
     ```
     The survey provides a group index shared by ranks with identical
     kernel fingerprints. Build `--part` from those groups rather than
     from a rank-layout convention you assumed, and select one
     representative per part. Cross-check against the tuning config:
     TP × PP is the world size; EP reuses TP ranks. Record disagreements
     and unobserved ranks instead of inferring their fingerprints.
     Re-run with the surveyed membership and original rank ids, never
     renumbered; for an eight-rank survey that established two groups:
     ```bash
     python <skill_dir>/scripts/run_all.py \\
         --profile 0=... --profile 1=... --profile 2=... --profile 3=... \\
         --profile 4=... --profile 5=... --profile 6=... --profile 7=... \\
         --representative 0 --representative 4 \\
         --part stage-0=0,1,2,3 --part stage-1=4,5,6,7 \\
         --taxonomy <workspace>/taxonomy.json --out <workspace>/nsys_analysis
     ```
   - If anchor detection lists candidates and exits, rerun with
     `--anchor <pattern>` for the densest recurring decode kernel in
     `nsys_stats.txt`. Record the choice; `--n-iters` cannot fix detection.
   - **Verify the taxonomy before quoting a single category number.**
     Inspect `cat_full.json`’s `classified_by` and
     `uncategorized_above_threshold`. Extend regexes/categories for every
     uncategorized name above 1% of iteration time and rerun until none
     remains significant. This re-reads files only, no server, no GPU.
     Save the final `taxonomy.json`. Keep `gemm`, `mha` and `nccl`:
     the pipeline hard-codes them as the Step 4 anchors and collective class.
   - Read `summary.json`, and per representative `windows.json`,
     `busy.json`, `gap.json`, `cat_*.json`, `comm.json`, and
     `opgroup.json` / `module_slice.json`; read `part-<name>/jitter.json`
     when parts were declared. Report their metrics per the findings
     contract, including classification coverage, reconciliation, and
     graph-granularity limitations.
   - Author `<workspace>/nsys_analysis/items.json` per the skill’s Output
     contract: one entry per opportunity with `id`, `title`, `claim`,
     `evidence`, `magnitudeMs`, `status`; utilization-derived entries also
     need `boundingResource`, `headroomVerdict`, `iters`. An omitted
     opportunity reads as one that does not exist.
     Write `{{"items": []}}` when the analysis genuinely found nothing.
   - Analyze Run A’s timing capture now; do not wait for A2. If the skill
     is unavailable, export fails, or the pipeline errors, retain the
     kernel table and use the findings contract’s unavailable marker.
     Never hand-derive busy/idle rungs or a compute-absent split that the
     pipeline did not produce.

### Multi-GPU: where the nsys wrap has to go

- Bare `trtllm-serve` at world size >1 uses `MPI.COMM_SELF.Spawn`
  (`tensorrt_llm/llmapi/mpi_session.py`, `MpiPoolSession`).
  `--trace-fork-before-exec=true` does not follow them: the trace may
  contain only the parent, which has no model rank. Treat absent model
  kernels as a topology limitation; `profile.profile_ranks` beyond `[0]`
  cannot be honored through this launch shape.
- With one task per rank (`srun --ntasks-per-node=<world_size> ...
  trtllm-llmapi-launch trtllm-serve ...`), the wrap goes **inside** the
  step, once per task, with rank ids in output names, as in
  `examples/disaggregated/slurm/benchmark/start_worker.sh`:
  ```bash
  if echo ",<profile.profile_ranks, comma-separated>," \\
       | grep -q ",${{SLURM_PROCID}},"; then
      wrap="nsys profile -o <workspace>/server_nsys_rank${{SLURM_PROCID}} \\
            ...the same flags as the single-rank command above..."
  else
      wrap=""
  fi
  ${{wrap}} trtllm-llmapi-launch trtllm-serve <ckpt> ...same trtllm-serve flags...
  ```
  Wrap only `profile.profile_ranks`: tracing overhead can appear as rank
  jitter. Unnecessary wrapping makes the straggler verdict describe nsys
  rather than the model; record the captured ranks and this limitation.

Every rank arms `cudaProfilerStart/Stop` on the same iteration counter;
`TLLM_PROFILE_START_STOP` needs no per-rank handling.
`TLLM_PROFILE_LOG_RANKS` selects which ranks print the step log line and
changes no capture. Never generalize a single-rank trace to the whole job.

## Run A2 — Nsight Systems utilization + call-stack passes

These two **additional** captures are **additive, never blocking**: run
only after Run A produced a usable timing report. Sampling and backtraces
perturb timing, so every iteration/busy/idle/gap number must come from
Run A. Reuse its server config, replay, iteration window, capture gate and
working graph granularity; change only the output name and flags below.
If a pass fails or a required flag is unsupported, record the unavailable
marker from the findings contract and retain Run A’s analysis.

### Pass A2a — per-operator utilization (`--gpu-metrics-devices`)

```bash
   -o <workspace>/server_nsys_metrics -f true \\
   --gpu-metrics-devices=all \\
   --gpu-metrics-frequency=100000 \\
```

Use 100 kHz: the default 10 kHz misses short kernels, while 200 kHz has
not delivered proportionate sampling and enlarges artifacts. For selected
GPUs, replace `all` with ids such as `0,1`. If `ERR_NVGPUCTRPERM` occurs,
record unavailable metrics and continue; do not change the system owner’s
`NVreg_RestrictProfilingToAdminUsers` setting. Interpret throughput and
unsampled kernels under the findings contract.

### Pass A2b — the call sites behind the kernels (backtraces)

```bash
   -o <workspace>/server_nsys_stacks -f true \\
   -s process-tree -b dwarf --sampling-frequency=2000 \\
   --python-backtrace=cuda \\
   --python-sampling=true --python-sampling-frequency=2000 \\
   --cudabacktrace=kernel:5000,sync:10000 \\
```

- `--python-backtrace` captures Python CUDA call sites;
  `--cudabacktrace` captures C/C++ chains and requires CPU sampling, so
  retain `-s`/`-b`. Verify version-specific accepted values with
  `nsys profile --help 2>&1 | grep -A3 python-backtrace`.
  Frequency limits are 100–8000 Hz for CPU and 1–2000 Hz for Python.
- Measure the graph-kernel share in the exported sqlite:
  `SELECT COUNT(*) FILTER (WHERE graphId IS NOT NULL) * 100.0 / COUNT(*)
  FROM CUPTI_ACTIVITY_KIND_KERNEL;`. At >~90%, expect eager-prologue and
  host-loop attribution: a `cudaGraphLaunch` stack names the graph launch
  site, not its individual kernels.
- Check `SAMPLING_CALLCHAINS`: rows carrying `unresolved = 1` throughout
  indicate missing symbols, not absent stacks. Record that limitation.

### Consume the additional captures

Export each successful capture:
```bash
nsys export --type sqlite -o <workspace>/<name>.sqlite \\
    <workspace>/<name>.nsys-rep
```
Then rerun Run A step 5 with
`--metrics-profile 0=<workspace>/server_nsys_metrics.sqlite` added, using
actual rank ids for multi-rank inputs. For utilization, it reads the
sampling capture, never the timing one; timing still comes from Run A.
This only rereads files.
Read A2b from its own sqlite: the pipeline’s `--launch-sequences` /
`--call-stack-trace` options require a different tool’s exports, not A2b.

## Run B — Nsight Compute (ncu, per-kernel deep dive)

Run B runs last: nsys locates the time, ncu measures the kernel’s bound
class, occupancy and stalls. Select targets and bounded passes in step 2.

1. Check `ncu --version` via `Bash`. If the tool is absent or profiling
   yields `ERR_NVGPUCTRPERM`, skip it using the findings contract’s
   unavailable marker. Load `perf-nsight-compute-analysis` via the `Skill`
   tool (`trtllm-agent-toolkit:perf-nsight-compute-analysis` if needed).
   It owns classification thresholds and escalation interpretation.
   If the skill is unavailable, retain raw metrics and mark ncu-derived
   classification unavailable. A required ledger bound may instead use
   source/timeline-supported inference with explicit provenance; do not
   guess the missing skill’s thresholds.
"""


_DEFAULT_NCU_TARGETING = """\
2. **Pick the targets from the timeline decomposition, not the kernel sum.**
   This pass targets the top kernels Run A surfaced: choose 3–6 stems
   covering most in-window time from `cat_full.json`’s `per_category`
   (`median_ms_per_iter`) and `opgroup.json` / `module_slice.json`.
   Use `matched_kernels` to build family filters, then one
   `--kernel-name "regex:<stem1|stem2|...>"`; record stem → full name.

   `cuda_gpu_kern_sum` is a sum across overlapping streams over the whole
   capture; the decomposition is a union clipped to the iteration window.
   Use the table only as fallback: if nsys ran but the skill's pipeline
   did not, rank on `kern_sum` and disclose it. If nsys was not run,
   omit `--kernel-name`, retain the `--launch-count 40` cap, and disclose
   the untargeted sample. After a partial capture, retry a missing dominant
   family only if walltime allows; report achieved coverage.
"""


_NCU_CAPTURE_TEMPLATE = """\
3. Relaunch with the same server flags and steady-state window. Start
   from this canonical invocation — do not improvise the ncu flags.
   Use step 2's targets and pass policy with these capture settings:
   ```bash
   cd <trtllm_repo_path>
   setsid env TLLM_PROFILE_START_STOP="<profile.nsys_iter_range>" \\
   ncu --target-processes all \\
       --profile-from-start off \\
       -o <workspace>/server_ncu{artifact_suffix} -f \\
       --section SpeedOfLight --section LaunchStats --section Occupancy \\
       --section WarpStateStats --section MemoryWorkloadAnalysis \\
       --section ComputeWorkloadAnalysis \\
       --kernel-name "regex:<target stems from step 2>" \\
       --launch-count {launch_count} \\
       trtllm-serve <checkpoint_path> ...same trtllm-serve flags... \\
       > <workspace>/serve.log 2>&1 < /dev/null &
   echo $! > <workspace>/serve.pid
   ```
   - `--profile-from-start off` uses the server’s profiler start/stop gate.
     `--launch-count` bounds kernel replay, which can slow launches 10–100×.
   - Multi-rank graph replay may trigger the executor’s hang watchdog
     (`tensorrt_llm/_torch/pyexecutor/hang_detector.py`; 300 s in this
     checkout, without a `hang_detection_timeout` config/env knob).
     Import surviving launches from the incrementally written report;
     use step 2’s bounded targeting policy for any follow-up pass.
     Exclude cross-rank allreduce/allgather stems: collective replay can
     deadlock ranks. Take their time share from nsys.
   - Collect the listed escalation sections together to avoid repeated
     checkpoint loads. `--target-processes all` follows server workers.
     Drop only rejected, nonessential flags and record the change.
4. Poll readiness and replay the canonical benchmark with
   `--no-test-input`. Allow for replay overhead. Client timings are
   **not measurements** of serving performance and must not be reported
   as performance results.
5. Tear down to finalize the report, then export details and CSV:
   ```bash
   ncu --import <workspace>/server_ncu{artifact_suffix}.ncu-rep --page details \\
       > <workspace>/ncu_details{artifact_suffix}.txt
   ncu --import <workspace>/server_ncu{artifact_suffix}.ncu-rep --page raw --csv \\
       > <workspace>/ncu_raw{artifact_suffix}.csv
   ```
6. Interpret captured kernels with the loaded skill and fill the
   `ncu kernel analysis` findings contract, including achieved coverage
   and the evidence for each bound class.
"""


def build_profiling_runs_reference(
    ncu_targeting: str | None = None,
    *,
    launch_count: str = "40",
    artifact_suffix: str = "",
) -> str:
    """Compose capture instructions with one effective ncu targeting policy.

    Args:
        ncu_targeting: Complete Run B step-2 text, including its numbered
            heading. None uses the default top-kernel targeting policy.
        launch_count: Effective launch-count value or placeholder for the
            selected targeting policy.
        artifact_suffix: Suffix shared by the capture, details and CSV files.
    """
    targeting = _DEFAULT_NCU_TARGETING if ncu_targeting is None else ncu_targeting
    return (
        "\n\n".join(
            (
                _PROFILING_RUNS_BEFORE_TARGETING.rstrip(),
                targeting.strip(),
                _NCU_CAPTURE_TEMPLATE.format(
                    launch_count=launch_count, artifact_suffix=artifact_suffix
                ).rstrip(),
            )
        )
        + "\n"
    )


PROFILING_RUNS_REFERENCE = build_profiling_runs_reference()

# --------------------------------------------------------------------------- #
# The findings contract (both workflows' analyzers): the required structure
# of analysis.md. Path-neutral — perf-analyze writes it at the
# workspace root, perf-optimize in the round's analysis/ directory; each
# workflow's instructions name the exact path.
# --------------------------------------------------------------------------- #

PROFILE_FINDINGS_CONTRACT = """\
## Current theoretical performance model and analysis report

Write `performance_model.yaml` and `analysis.md` in the assigned directory
on every turn, including replan, reuse and SOL-disabled turns. This current
model drives analysis, experiment priority and convergence. The initial
`sol_projection.md`, `sol.json`, kernel ledger and exports support it.

### Comparable basis and derivation

Cover every configured concurrency (null in scalar mode). Identify build,
hardware, precision/shapes, TP/EP, workload/context distribution, accepted
tokens and timing basis. Match work, scope and statistic; a whole-run mean
minus a short-window decode median is not measured prefill/host time.
Unmatched differences stay `unexplained`. Mark unprofiled points unknown
unless a supported scaling law and uncertainty justify an estimate; keep
estimates distinct from measurements.

Derive necessary FLOPs, bytes, resource rates, communication and dependencies,
including prefill and non-layer serving work. Account for legal elimination,
fusion and overlap once in disjoint critical-path components. Show conversion
to the target metric using batch, accepted tokens, phase frequency and queueing.
Partial kernel/decode models cannot establish complete serving ceilings;
unprofiled time and sub-threshold tails are not zero-cost removable work.

Distinguish physical constraints from implementation/scope limits. A token
cap or rebuild issue is not a physical floor; achieved bandwidth is an
empirical reference. Use collective-specific latency calibration. A failed
lever closes that experiment, not its remaining gap. Revise assumptions
from evidence, never to fit a failed attempt or an unexpected measurement.

### Machine-readable contract (`performance_model.yaml`, version 1)

```yaml
version: 1
model_id: round-1-current              # new identity when assumptions/bounds change
target_metric: output_throughput      # optimize.target_metric, else output_throughput
direction: higher                     # lower for target metrics ending in _ms, else higher
points:
  - concurrency: 64                   # null for a scalar benchmark
    operating_point:
      build: "recorded source/build identity"
      hardware: "recorded GPU and parallel mapping"
      workload: "ISL/OSL, precision, shapes, batch, context distribution"
      timing_basis: "whole-run serving throughput; matched modeled workload"
    measured_value: 10000.0            # use the actual benchmark; null if not available
    theoretical_best_value: null      # never invent a complete ceiling from a partial model
    derivation: "Decode-only evidence cannot yet price the complete serving workload."
    assumptions: []
    evidence: []                      # nonempty artifact citations for a known bound
    measurement_evidence: ["baseline/concurrency_64/result.json: output_throughput"]
    components: []                    # unknown decomposition, NOT zero work
    status: measurement_limited
    unexplained: "Representative prefill timing and frequency are missing."
    next_test: "Capture mixed prefill/decode iterations at c=64 with phase markers."
```

Use actual benchmark values. Each point requires all fields shown above;
metric values are positive numbers or null. Known predictions/measurements
need nonempty evidence/measurement_evidence. Unknowns require `unexplained`
and `next_test`; empty components mean unknown decomposition, not zero work.

Comparable timing adds paired nullable `measured_ms`/`theoretical_best_ms`
and `timing_derivation` (composition, overlap removal and metric conversion).
Components require unique `id`, nullable nonnegative `measured_ms` and
`theoretical_best_ms`, `gap_kind`, evidence and next_test for unresolved gaps.
Known component sums must match point totals. Unknown bounds cannot become
zero or a complete theoretical_best_ms. Reference supporting model/kernel IDs.

`gap_kind`: `actionable`, `physical_limit`, `scope_limited`,
`measurement_limited`, `unexplained`, `model_error`. Unknown, scope-limited,
measurement-limited and model-error components need a concrete next_test.
Physical necessary work belongs in the floor. `model_error` requires
point status `model_invalid`.

Point `status`: `open`, `converged`, `measurement_limited`, `scope_limited`,
`model_invalid`. Mark a point/component `measurement_limited` when its next
runtime measurement should trigger targeted profiling within remaining rounds.
Use `open` for supported optimizations or model work using available facts.
Measurements beating a predicted bound require `model_invalid`; retain the
measurement and investigate. Numerical bound comparisons allow 0.1% rounding.

Convergence requires known comparable metrics, reconciled numeric timing
totals and timing_derivation, complete explained components, no unexplained
residual, `convergence_tolerance_pct` (0–5), and `convergence_evidence` for
that tolerance. Residual percentage is
`abs(measured_value - theoretical_best_value) / theoretical_best_value * 100`,
not throughput uplift. All scored points must converge (focus subset when
configured). Benchmark noise, an empty roadmap, target gain attained or
exhausted budgets do not prove convergence; report stop reasons separately.

Replan/reuse preserves measurement provenance. Cite previous model_id,
changed assumption, old/new prediction, reason and evidence; keep detailed
history in linked artifacts. Old measurements do not describe a new build.
Only the Profiler or combined capture stage collects new runtime evidence.

### Required `analysis.md` structure

Use these four sections, about 1,000 words plus two tables. Link commands,
config, inventories, detailed counters/dispositions and derivations.

```
# Analysis: <model>

## Result
<One paragraph: measured state, current model_id, convergence status and
most consequential uncertainty. Link profiler_report.md for capture scope.>

## Theoretical performance model
<Operating basis and concise equations composing the best current model.>
| Concurrency | Measured target metric | Theoretical best | % of best | Remaining gain | Status |
<All configured points, scored subset marked; unknowns as —. Include a
brief evidence-backed model revision only when assumptions changed.>

## Gap analysis
| Component / point | Measured ms | Best ms | Gap ms | Constraint or next test | Evidence |
<Disjoint matching timing scopes, explicit unknown residual, floor plus
gaps reconciled to measured total. Separate physical, scope, measurement
constraints, actionable work and model error; uncertainty is not infeasibility.>

## Next actions
<Rank a short set of model-linked experiments or missing measurements.
For each: predicted target-metric effect, evidence, and falsification test.
Include only experiment outcomes that changed the model or next decision.>
```

For throughput use measured/best for % of best and best/measured - 1 for
remaining gain; for latency use best/measured and 1 - best/measured for
potential latency reduction. Name units and denominator. Do not mix mean
per-point gain with ratio of means or serving throughput with kernel-sum
utilization. Baseline and final must use the same current model when
reporting gap closure; model revisions are not implementation gains.

### Diagnostic evidence retained in supporting artifacts

Keep these diagnostics in linked artifacts, not extra report sections.
Record each unavailable reason beside its affected model once; capture
setup belongs in profiler_report.md.

### Supporting nsys timeline artifacts
- Top kernels: name and % of GPU time
- Per-iteration time: median/min/max, `n=` iterations and chosen anchor
- Three busy rungs and idle complements: device busy, non-transfer busy,
  compute busy / **compute-absent**; never "compute idle"
- Compute-absent split: launch-starved / blocking (name the producer) /
  dependency-stalled, in ms and % of iteration. Reconcile
  `iter_ms ≈ device_busy_ms + device_idle_ms` (~0.2 ms) and
  `launch_starved + blocking + dependency_stalled ≈ compute_absent`
  (~0.5 ms), both stated as pass/fail. Investigate whether a sum was used
  where a union belongs before reporting a mismatch; do not round it away
- Kernel mix: compute/tensor-core vs memory/elementwise and NCCL for
  multi-GPU; exposed collective time split into transfer vs jitter wait.
  State `classified_by` (observed / regex / uncategorized) beside the table;
  a category number whose taxonomy was not verified is not a finding.
  Graph-granularity categories are graph aggregates, not individual kernels
- Per-op breakdown: which Step 5 mode ran, `fused_share_of_residual_pct`
  and `module_slicing_recommended`; top operators from `opgroup.json`, or
  `window_labels` and every touched scope’s `share_pct` from
  `module_slice.json`. This names *what to optimize*
- Per-operator utilization (A2a): bounding resource = maximum throughput
  across compute / memory / network / bus, its `[Throughput %]`, headroom
  verdict and sample count. Never average resources or use compute alone.
  `SMs Active [Throughput %]` is residency, not throughput. Short kernels
  with no samples are `unsampled`, not 0%. If unavailable, retain
  `gpu metrics unavailable: <reason>`; cost alone is not headroom
- Call sites (A2b): observed Python/host frames with graph-kernel share
  and scope of attribution (per-kernel or eager prologue / host loop).
  Do not label a graph-launch stack as an internal kernel’s call site.
  Record unresolved symbols; if unavailable, retain
  `call stacks unavailable: <reason>`
- Rank jitter, when several ranks/parts exist: from `part-<name>/jitter.json`,
  report `jitter_cost.mean_jitter_wait_ms_per_iter`, `pct_of_iter`,
  `non_comm_busy_spread` (`spread_pct`, `slowest_rank`, `fastest_rank`),
  and `operator_spread.imbalance_operator`. Include `straggler.verdict`:
  the spread is never reported without it. The straggler verdict (`pinned` /
  `rotating`) distinguishes a repeated straggler at `threshold_pct` from
  a changing identity. Quote `arrival_lateness`
  only with its `floor_ms` beside it and valid `alignment`; otherwise
  report `lost_claims`. A part holding one rank has a `null` `jitter_cost`:
  state why it is absent and make no imbalance claim
- If jitter wait dominates comm, the cost is imbalance, not the network:
  the collective is where the cost *appears*, not where it is *caused*.
  Pursue the straggler rather than assuming transfer/overlap changes help
- Per-request prefill/decode split, if perf_metrics is available
- If timeline analysis failed or was unavailable, retain available
  `nsys stats` numbers and `timeline analysis unavailable: <reason>`

### Supporting ncu kernel analysis artifacts
- Per-kernel table: kernel, duration, Compute (SM) SOL%, Memory SOL%,
  achieved occupancy, bound class per the loaded skill’s thresholds,
  dominant warp-stall reason, occupancy/launch limiters
- Captured timeline kernels and their GPU-time coverage; explain why
  dominant kernels are slow using the measured escalation evidence
- If ncu did not run, retain `ncu unavailable: <reason>`. If only the
  methodology skill is missing, retain raw metrics with ncu-derived
  classification unavailable

For the largest gaps, reconcile **nsys timeline**, **ncu kernel analysis**,
and **SOL correlation** when enabled. Name corroborating, contradicting and
missing evidence; at comparable impact prefer agreement across pillars.
"""

# --------------------------------------------------------------------------- #
# Optimization casebook consultation (shared by benchmarker + analyzer)
# --------------------------------------------------------------------------- #

CASEBOOK_CONSULTATION = """\
## Optimization casebook

After reading `task.yaml`, load `perf-optimization-casebook` with `Skill`
(or `trtllm-agent-toolkit:perf-optimization-casebook`). Use its bottleneck
signal → candidate pattern index and relevant cases as read-only references;
do not apply changes or run extra experiments. Note unavailability once.
"""

# --------------------------------------------------------------------------- #
# Remote execution boundary (appended with task-specific values by the CLI)
# --------------------------------------------------------------------------- #

REMOTE_SLURM_EXECUTION = """\
## Remote Slurm execution

Local files are the source of truth: edit code, run Git, and write reports
locally. Use SSH for remote-only inputs. For each remote Slurm job, rsync the
required inputs and changed source into an isolated directory under the remote
run root, excluding `.git`, builds, and caches. Submit through the configured
SSH target and wait for Slurm to finish. Pull the role's required outputs and
failure logs back for local inspection, then remove that remote job directory.
Prefer one allocation for related work; retry only after a concrete correction
or when another measurement is needed.
"""


# --------------------------------------------------------------------------- #
# Slurm execution (appended only when task.yaml has a slurm-environment block)
# --------------------------------------------------------------------------- #

EXECUTION_SLURM_BOOTSTRAP = """\
## Slurm execution (`slurm-environment`)

Run the server, readiness poll, workload and profilers inside one Slurm
container. Use `slurm_partition`, `docker_image`, `trtllm_repo_path` and
`checkpoint_path` from `task.yaml` verbatim. Bind repo, checkpoint and
workspace at identical absolute host/container paths. Invalid inputs are
blockers; do not substitute paths, partitions, images or local execution.

Use one `sbatch` script for the entire stage: launch, readiness, workload,
profiles, teardown. Interactive `salloc` / `srun --pty bash` allocations
end with their `Bash` call and do not survive into subsequent calls.
Write job output (`sbatch -o`) and artifacts inside `<workspace>`, not
compute-node `/tmp`. Paths in the script are cluster paths.

### MPI launch and network namespace

Inside `srun`, bare `trtllm-serve` fails at `MPI.COMM_SELF.Spawn`
(`MPI_ERR_SPAWN`). Launch with all three required pieces:

```bash
srun --partition=<slurm_partition> \\
     --mpi=pmix \\
     --ntasks-per-node=<world_size> \\
     --container-image=<docker_image> \\
     --container-mounts=<repo>:<repo>,<ckpt>:<ckpt>,<workspace>:<workspace> \\
     --container-workdir=<repo> \\
     --gres=gpu:<num_gpus> \\
  bash -c 'trtllm-llmapi-launch trtllm-serve <ckpt> ...'
```

`<world_size>` comes from the parallel mapping: TP × PP × applicable
independent dimensions; `moe_expert_parallel_size` reuses TP ranks and
**does not** multiply world size. Recompute after config changes.
The `trtllm-llmapi-launch` wrapper adopts the ranks created by `srun`.

Server, readiness poll and benchmark client must share the **same `srun`
step**, not merely the allocation: pyxis steps have separate network
namespaces, so `127.0.0.1` does not cross steps. Follow the normal lifecycle
inside that step. If separate steps are necessary, bind the server to
`0.0.0.0`, use `$SLURMD_NODENAME` from the client, and `--overlap` to share
the allocation. Prefer the single-step script.
"""


# --------------------------------------------------------------------------- #
# SOL projection derivation (the projector role's methodology blocks — shared
# with perf-optimize, whose projector prompt composes the same fragments)
# --------------------------------------------------------------------------- #

SOL_PROJECTOR_METHODOLOGY = """\
## The methodology: `internal-perf-sol-analysis`

Load the skill and use its α-β-u model, per-op recipes, peaks calculator
and `measure_channels.py`. Report **% of SOL** (latency: SOL/measured;
throughput: measured/SOL), secondary **MFU**/**MBU**, **gap-to-SOL**, and
**bound** ∈ compute / memory / launch / comm. A ratio above 100% signals a
model/measurement mismatch; do not clamp it. This projection seeds the
Analyzer's current `performance_model.yaml`.

- Use `task.yaml`'s `sol.gpu` as the `sol_calc.py peaks --part` hint.
  The Skill load supplies the scripts' base directory.
- When local GPUs are reachable and idle (`nvidia-smi`), measure latency
  constants and merge them into the peaks file. On a Slurm login node,
  leave launch α unmeasured; derive β/u and record the limitation in
  *Projection setup* and *Open questions*. Never guess α.
- No measured per-op timings exist before profiling. Instantiate the
  skill's formulas/recipes in linked `sol_work/` derivations with actual
  inputs and units. Do not invent measured_ms rows to run `sol_calc.py analyze`;
  the Analyzer runs it against its selected profile. The report's
  *Arithmetic* entry links these derivations and explains their composition.
- Persist calculator output and any measured latencies in
  `<workspace>/sol_work/peaks.json`; record its path in *Projection setup*.
- Derive deployment quantities from checkpoint `config.json`, serving
  precision and TP/PP/EP: weight bytes per GPU (active experts for MoE),
  KV bytes per decoded token at mean context, FLOPs/token and per-layer
  collectives. Recipes do not supply these inputs.
- If the skill is unavailable, ground a labeled coarse ceiling in
  `config.json` and sourced internal knowledge, or write *Projection
  unavailable* when no defensible bound exists.

Recipes model **kernel execution plus per-launch latency only**. Compose
scheduler/host prep, queueing and dynamic-batching terms on a matching basis
before claiming an end-to-end ceiling. Unmatched residuals remain unexplained;
a large gap alone does not establish host overhead. Name missing phase
measurements or derivations in *Open questions*.
"""


# Appended to the projector's prompt only when the workflow resolved
# ``perf-analysis`` instead of the skill above (``sol_methodology``). The
# methodology's own last bullet already says what to do without the
# calculator; this names the skill that replaces it and the one artifact
# that stops being writable.
SOL_METHODOLOGY_FALLBACK = """\
## Fallback: `internal-perf-sol-analysis` unavailable

Load the supplied `perf-analysis` skill and use its bottleneck classification.
Ground a labeled coarse ceiling in `config.json` and sourced internal
knowledge; state that peaks are not calculator-resolved. There is no peaks
calculator or `measure_channels.py`: skip `sol_work/peaks.json`. If no
defensible bound exists, write *Projection unavailable*. Never fabricate.
"""


SOL_PROJECTOR_INTERNAL_KNOWLEDGE = """\
## Internal knowledge (reference only)

For unresolved architecture/GPU specs, consult `internal-glean-search` or
`internal-glean-specialist` if available. Every projected value must remain
reproducible from named sources and recorded arithmetic. Stop searching
once those sources support the derivation.
"""


# --------------------------------------------------------------------------- #
# SOL projection consumption (appended only when the projector stage is enabled)
# --------------------------------------------------------------------------- #

# Role-neutral correlation recipe shared by both workflows' analyzers —
# each workflow's SOL analyzer extension composes it and names the
# artifact directory the regions/sol JSONs land in.
SOL_CORRELATION_METHOD = """\
### Correlate the fresh profile against the ceiling (`sol_calc.py analyze`)

The selected profile supplies measured per-op times. Join structural work
with the skill's calculator to support the current model. Kernel-sum ratios
are diagnostics, not end-to-end % of theoretical best.

1. **Load the `internal-perf-sol-analysis` skill** (via the `Skill`
   tool; fully-qualified
   `trtllm-agent-toolkit:internal-perf-sol-analysis` if the bare name
   is not found). The load announces the skill's base directory — the
   calculator is `<skill_dir>/scripts/sol_calc.py`. Its correlation
   contract and `regions.json` schema are the method; what follows is
   only what the skill cannot know about this stage.
2. **Recover the Projector's peaks file** at
   `<workspace>/sol_work/peaks.json` (its path is recorded in
   `sol_projection.md`'s *Projection setup*). When it carries no
   measured `latencies`/`sms` — the Projector ran without GPU reach —
   run the skill's `measure_channels.py --launch … --merge-into <that
   peaks.json>` yourself: unlike the Projector's stage, a GPU is
   reachable here by construction (you just profiled on it).
3. **Build `regions.json` from your traces — structural facts only.**
   The rows come from the nsys per-kernel sums
   (`cuda_gpu_kern_sum`, NVTX ranges, the timeline's kernel-category
   rollup), rolled up into the skill's region keys and schema. Start with
   the shapes in `sol_projection.md`'s *Arithmetic*,
   then verify them against current source/runtime facts; correct stale
   shapes with evidence. A region whose params you cannot ground stays in
   `other` with a note — **never invent params or `measured_ms` rows**.
4. **Run the calculator** (never hand-compute a SOL number):
   ```bash
   python <skill_dir>/scripts/sol_calc.py analyze \\
       --regions <artifact dir>/regions.json \\
       --peaks <workspace>/sol_work/peaks.json \\
       [--recipes-dir <artifact dir>/sol_recipes] \\
       --out <artifact dir>/sol.json
   ```
   An op family the built-in recipes do not cover is either given a
   recipe under `sol_recipes/` (the skill's `check-recipe` route) or
   left in `other` — those are the only two options.
5. **Keep `sol.json` as a supporting artifact.** Cite its region/model IDs
   from the current model's component derivations. Preserve calculator
   columns, scope, % of SOL, MFU/MBU and corrections in the artifact; do not
   paste the full table into `analysis.md` or make its kernel-sum ratio a
   serving-performance headline. Unknown recipes are partial model coverage.


Degrade honestly: when a precondition fails (projection unavailable,
peaks file missing and the constants unmeasurable, nsys produced no
usable per-kernel table), record `Correlation unavailable: <reason>` beside the affected model and move
on — a fabricated correlation is worse than none.
"""


SOL_ANALYZER_CONTEXT = (
    """\
## Update the current model from SOL evidence

Read `sol_projection.md` as the initial theoretical model and preserve it
as provenance. It is context, not a measurement or a permanently fixed ceiling.
Use its structural derivations as a starting point, checking shapes and
runtime facts against the selected capture. Feed valid per-op results into
`performance_model.yaml`, composing a current end-to-end model only where
scope and timing allow it. Record changes in assumptions with evidence and
keep kernel-level revisions in `kernel_ledger.yaml` when present.

The model's remaining gaps drive experiment ranking. A failed optimization
alone does not weaken a bound, and a large gap alone does not identify a
host bottleneck. Missing facts stay unknown with next_test. The single
Theoretical performance model and Gap analysis sections own this account;
no extra SOL or remaining-gap section. Unavailable correlation is a one-line
limitation beside the affected model, not a reason to omit the current model.
"""
    + SOL_CORRELATION_METHOD
    + """\
Keep regions.json, sol.json and sol_recipes/ in <workspace>/sol_work/.
"""
)


SOL_REPORTER_GUIDANCE = """\
## Use the current theoretical model

Read `sol_projection.md` as initial provenance and the latest
`performance_model.yaml` + `analysis.md` as the authoritative current model.
Use the current model for the report's single per-point comparison and gap
analysis. Do not add Projection vs Measured, SOL correlation or extra
remaining-gap sections. If the initial projection has been superseded,
mention the material correction once with its evidence and model_id.
Do not mix its old ceiling with a current kernel floor or practical rate.

Use `benchmark_results.md` and its raw result JSONs for measured values.
Compare only compatible builds/workloads/timing scopes in `analysis.md`.
A capture that does not represent the benchmark workload needs an explicit
limitation and cannot establish convergence. If the current bound is unavailable, display unknown and
the required next test; do not fall back to a superseded initial bound.
Large model discrepancies do not establish host/scheduler overhead: require
phase measurements. Distinguish physical limits, campaign scope, measurement
limits and unresolved model error. Incomplete evidence cannot establish convergence.
"""


# --------------------------------------------------------------------------- #
# Bottleneck taxonomy + HTML companion (reporter)
# --------------------------------------------------------------------------- #

BOTTLENECK_TAXONOMY = """\
## Bottleneck taxonomy

Choose one dominant category from measured evidence; rank close secondary
factors. If evidence cannot distinguish causes, say so.

- **Compute-bound:** high GPU busy, near-roofline FLOPs and GEMM/attention
  dominance; throughput scales with batch to a compute ceiling.
- **Memory-bandwidth-bound:** high DRAM throughput at modest FLOPs;
  elementwise/norm/KV/dequant kernels or weight/KV reads dominate TPOT.
- **KV-cache-capacity-bound:** low free-block headroom, queued/preempted
  requests and concurrency below target; more KV capacity raises throughput.
- **Kernel-launch / host-overhead-bound:** distinguish two mechanisms:
  - *Kernel-launch overhead:* many tiny eager kernels, launch-heavy CUDA
    API time and short idle gaps. CUDA graphs / overlap scheduler amortize
    forward launches.
  - *Host-prep / scheduler exposed:* a named host phase such as
    `_prepare_inputs`, `.item()`/`cudaStreamSynchronize`, or >100 µs gaps
    rivals GPU time. CUDA graphs do not remove host prep outside replay;
    reduce host work/synchronization. Low GPU busy alone cannot distinguish
    these two causes.
- **Communication-bound (multi-GPU):** exposed NCCL/all-reduce/all-gather
  and waiting dominate; scaling efficiency falls with TP/PP/EP.
"""


HTML_COMPANION = """\
## HTML companion (`performance_report.html`)

Write one self-contained offline HTML file with the same four sections,
tables, numbers, model_id and conclusions as the Markdown. Use inline CSS,
no external CDN/fonts/assets, accessible headings and real tables, readable
light/dark styles, and print-friendly layout. Keep navigation small; do not
add mandatory interactive widgets or extra report sections.

The current model comparison is the primary visual. An optional inline SVG
chart may plot its exact per-point values, distinguishing measured values
from theoretical estimates and omitting unknown points. In curve mode keep
one compact Pareto chart (x = tok/s/user, y = tok/s/gpu) when the measured
curve is available; optimization reports compare baseline and final only.
A theoretical overlay is allowed only with a supported conversion from the
CURRENT model to both axes; never substitute the initial projection. Label
concurrency and series clearly. No separate trajectory or top-kernel charts.
All charts must use exactly the table data and name the model revision.
"""


# --------------------------------------------------------------------------- #
# Shared rigor / progress rules
# --------------------------------------------------------------------------- #

EVIDENCE_DISCIPLINE = """\
## Evidence discipline

Cite the files supporting every metric, kernel and claim. Record exact
commands for reproducibility. Report failures/unavailable tools; never
fabricate numbers. Keep prose direct and free of conversational filler.
"""


def profile_ranks_note(ranks: Sequence[int]) -> str:
    """The per-run sentence naming which ranks nsys must capture.

    Interpolated into the analyzer's driving instructions by both
    workflows, so the stateless agent does not have to re-derive
    ``profile.profile_ranks`` from the spec — and so the multi-rank case
    is stated as a duty rather than left as an option the agent may or
    may not notice it has.
    """
    if len(ranks) <= 1:
        only = ranks[0] if ranks else 0
        return (
            f"Capture nsys on **rank {only} only** (`profile.profile_ranks`). "
            f"A single `--profile {only}=<sqlite>` runs the skill's pipeline "
            f"straight through with no representative to choose, and the "
            f"rank-jitter step does not apply — say so in one line rather "
            f"than reporting an imbalance you did not measure."
        )
    listed = ", ".join(str(rank) for rank in ranks)
    return (
        f"Capture nsys on **ranks {listed}** (`profile.profile_ranks`), one "
        f"trace per rank, per the *Multi-GPU* section of your system prompt: "
        f"the wrap goes inside the per-rank launcher step, and only these "
        f"ranks are wrapped — a rank slowed by its own profiler becomes "
        f"jitter the others wait on, which is the very quantity the "
        f"rank-jitter step measures. Then run the skill twice as that "
        f"section shows: once with every `--profile` and no "
        f"`--representative` to print the Step 0 survey, then again with "
        f"the representatives and `--part`s you derive from the survey's "
        f"fingerprint groups. Report the straggler verdict with the spread, "
        f"always. If the launch topology cannot give you per-rank traces "
        f"(a spawn-launched `trtllm-serve` cannot), capture what you can, "
        f"record the reason under *Caveats*, and make no imbalance claim."
    )
