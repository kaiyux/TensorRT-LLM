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

You drive the whole server lifecycle yourself with `Bash`. One server at
a time on the GPU(s) — the previous stage is *expected* to have torn its
server down before you start, but never assume it did: step 1 verifies
it, because an interrupted run leaves a detached server behind and the
port is fixed.

**You get a single turn — finish the work in it.** Launch, readiness
poll, benchmark/profile, teardown, and writing your output `.md` file
all happen in this one turn. Wait for slow steps (a checkpoint can take
many minutes to load) with **foreground** blocking shell loops; if one
poll window is not long enough, issue another blocking poll and stay in
your turn. **Do not end your turn to wait for a background poll to wake
you** — nothing re-invokes you, so the stage would advance with your
output file still empty and the whole run is wasted.

1. **Assert port 8000 is free — before launching anything.** The port is
   fixed, and a `trtllm-serve` from an earlier stage or an interrupted run
   is `setsid`-detached, so it *survives* a Ctrl-C and keeps answering on
   :8000. If you skip this check, your own server dies with "address
   already in use" while the **stale** one — serving an older config, or a
   different checkpoint entirely — answers every health poll, and the whole
   stage silently measures the wrong server:
   ```bash
   ss -ltn 'sport = :8000' | grep -q LISTEN && {
     echo "FATAL: port 8000 already in use — a stale server is still up"
     ss -ltnp 'sport = :8000'   # shows the owning pid/cmd
     exit 1
   }
   ```
   Do **not** work around a busy port by picking another one — the
   benchmark and profiling commands all target :8000. Reap the stale
   server by its PID with the teardown recipe in step 4 (identify it from
   the `ss -ltnp` output above), confirm the port is free, then launch.
   If `ss` is unavailable, `lsof -i :8000` or
   `curl -fsS http://127.0.0.1:8000/health` works the same way — a health
   response *before* you have launched anything is a stale server, not a
   ready one.

2. **Launch in the background, fully detached**, redirecting logs to a
   file in the workspace, and capture the PID. `setsid` + `< /dev/null`
   puts the server in its own session/process group so it survives across
   your separate `Bash` calls (a plain `&` job can be reaped when the
   launching shell exits):
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

3. **Poll readiness in the foreground** before sending any load. Large
   checkpoints can take many minutes to load; poll on an interval with a
   generous timeout, bail out early if the process dies, and `tail`
   `serve.log` if it stalls. Check **liveness first and ownership before
   declaring READY** — a `/health` response only proves *some* server is on
   :8000, never that it is yours:
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

   ⚠️ **A `FATAL: … not owned by PID` result is never something to work
   around.** It means a foreign server holds :8000, so every number you
   would produce belongs to a different config or checkpoint. Reap it
   (step 4), confirm :8000 is free, relaunch, and only then benchmark.
   The same applies when `owns_port` cannot resolve an owner at all: treat
   "unverified" as "not ours" and resolve it by hand (`ss -ltnp` /
   `lsof -i :8000` may need more privilege than the current shell has)
   rather than benchmarking on the assumption it is yours.

4. **Tear the server down — always**, even when a step failed, so the
   GPU is free for the next stage. Kill by the **recorded PID and its
   process group** (never by a name pattern), escalating to SIGKILL:
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
   Then confirm the GPU memory actually freed with `nvidia-smi`, **and
   that :8000 is free again** (`ss -ltn 'sport = :8000'` prints no
   `LISTEN` row), before launching another server. If an MPI daemon
   (`orted`) or other straggler survives, find it with a read-only
   `ps`/`pgrep` and reap it by its **exact PID**.

   ⚠️ **Never `pkill -f 'trtllm-serve'` (or any bare `trtllm-serve`
   pattern).** The string `trtllm-serve` appears in *your own* agent
   process and in the very shell running the teardown, so a name-based
   kill can terminate this agent. Always target the recorded PID /
   process group; if you must pattern-match a straggler you cannot reach
   by PID, use a precise pattern that includes the checkpoint path and
   confirm it with read-only `pgrep` before killing.

`nvidia-smi` (read-only) is a useful sanity check for how many GPUs are
visible and whether memory is free between runs.
"""


def build_server_lifecycle(active_tuning_config: bool = False) -> str:
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

The load generator lives in the TensorRT-LLM checkout at
`tensorrt_llm/serve/scripts/benchmark_serving.py`. Run it as a module
(deps are importable wherever `trtllm-serve` works).

**Start from this canonical command — do not improvise the flags.** Fill
the `<...>` placeholders from `task.yaml` (`checkpoint_path` and the
`benchmark` block) and the workspace path; keep every other flag exactly
as shown:

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
- **One run per concurrency point.** `benchmark.concurrency` is a single
  integer (one operating point) or a list of integers (Pareto-curve
  mode; the resolved spec is sorted ascending). With a single integer,
  run the command once with that value. With a list, launch the server
  **once**, run the command once per point **sequentially in ascending
  order** against the same server, and tear down after the last point —
  never relaunch the server between points, never run points in
  parallel, and never add, drop, or resize points beyond the configured
  list. ISL / OSL stay fixed across points.
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
  reason in `profile_findings.md`.
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
   placeholders and the explicit compatibility/fallback options below:
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
   concurrent load arrives. Keep the benchmark's configured `num_prompts`.
   If that load cannot reach `<stop>`, lower and document the iteration
   window; if it cannot provide a steady-state window, report profiling
   unavailable. Verify `serve.log` contains `Profiling started at iteration
   <start>` and `... stopped at iteration <stop>`.

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
# of profile_findings.md. Path-neutral — perf-analyze writes it at the
# workspace root, perf-optimize in the round's analysis/ directory; each
# workflow's instructions name the exact path.
# --------------------------------------------------------------------------- #

PROFILE_FINDINGS_CONTRACT = """\
## Required findings structure (`profile_findings.md`)

Write to the path named in your instructions. Keep these section headers
and cite artifact paths plus numbers for every signal. For each effective
profiling operating point, identify its trace/config and keep its metrics
separate; use the task’s resolved profiling points, not an assumed maximum.

```
# Profiling Findings: <model name>

## Profiling setup
- Profiled concurrency points: effective ISL/OSL/concurrency and matching
  benchmark row; serving config and profiling flags
- nsys: command, TLLM_PROFILE_START_STOP window, trace, graph granularity,
  captured/missing ranks, representatives, who chose them and survey basis
- Run A2a: GPU metric devices/frequency and capture, or reason unavailable
- Run A2b: backtrace flags, capture and graph-kernel share, or reason unavailable
- ncu: command, stem → full kernel names, launch counts and report per pass

## nsys timeline
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

## ncu kernel analysis
- Per-kernel table: kernel, duration, Compute (SM) SOL%, Memory SOL%,
  achieved occupancy, bound class per the loaded skill’s thresholds,
  dominant warp-stall reason, occupancy/launch limiters
- Captured timeline kernels and their GPU-time coverage; explain why
  dominant kernels are slow using the measured escalation evidence
- If ncu did not run, retain `ncu unavailable: <reason>`. If only the
  methodology skill is missing, retain raw metrics with ncu-derived
  classification unavailable

## SOL correlation (measured vs ceiling)
<Include when the SOL projector stage is enabled; its instructions supply
this section’s content. Omit the section entirely otherwise.>

## Ranked bottleneck hypotheses
1. <hypothesis> — taxonomy category, trace + numerical evidence, matching
   casebook bottleneck signal → candidate pattern row (if available), and
   supporting/corroborating/contradicting/missing evidence pillars
2. ...

## Caveats
<Unavailable profilers/skills, failed windows, partial coverage,
measurement changes, multi-GPU tracing and attribution limits. Repeat
unavailable reasons from the relevant evidence sections here.>
```

Synthesize the three evidence pillars: **nsys timeline**, **ncu kernel
analysis**, and **SOL correlation** when enabled. Each hypothesis names
which support it and whether the others corroborate, contradict or are
silent; a pillar that did not run is missing, never silently skipped.
At comparable impact, agreement across available pillars outranks a
single-pillar hypothesis. Use casebook precedents as read-only references
under the casebook consultation policy.
"""

# --------------------------------------------------------------------------- #
# Optimization casebook consultation (shared by benchmarker + analyzer)
# --------------------------------------------------------------------------- #

CASEBOOK_CONSULTATION = """\
## Ground your analysis in the optimization casebook (load it early)

After reading `task.yaml`, load `perf-optimization-casebook` with the
`Skill` tool (`trtllm-agent-toolkit:perf-optimization-casebook` if the bare
name is not found). Use its bottleneck signal → candidate pattern index
and relevant case files as **read-only reference material only**.
Do not apply optimizations, edit configs, or run extra experiments from
it. If it is not available in this environment, note that in one line
and proceed.
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
## Slurm execution (this task has a `slurm-environment` block)

`task.yaml` contains a `slurm-environment` section, so the server and the
benchmark must run **inside a Slurm-launched container**, not on the
login node. Read these fields and use them verbatim:

1. `slurm_partition` — the partition to submit GPU jobs to.
2. `docker_image` — the enroot/pyxis container image (typically a `.sqsh`).
3. Top-level `trtllm_repo_path` and `checkpoint_path` — bind-mount both
   into the container at the **same absolute path** they have on the host.

Do not invent a different partition, image, or path, and do not silently
fall back to a local non-Slurm run when `slurm-environment` is present. If
a value is unusable, stop and report it as a blocker.

Run the server, the readiness poll, the benchmark, and the profilers
**within one allocation** (so the load generator can reach the server),
e.g. an interactive `salloc`/`srun` shell or a single `sbatch` script:

```bash
# bind trtllm_repo_path, checkpoint_path, and workspace at identical
# host:container paths (comma-separated in --container-mounts).
srun --partition=<slurm_partition> \\
     --container-image=<docker_image> \\
     --container-mounts=<repo>:<repo>,<ckpt>:<ckpt>,<workspace>:<workspace> \\
     --gres=gpu:<num_gpus> --pty bash
# then, inside the container, follow the local launch / benchmark /
# nsys / ncu steps exactly as described, writing all artifacts to
# <workspace>.
```

The local launch, readiness-poll, teardown, and profiling steps are
otherwise identical — they just run inside the container. All artifacts
(`serve.log`, result JSON, `*.nsys-rep`, `*.ncu-rep`, the `.md`
outputs) must land in `<workspace>` so later stages and the user can read
them.

### Under Slurm, `trtllm-serve` needs `trtllm-llmapi-launch`

A bare `trtllm-serve` works from a login shell and **fails inside a Slurm
step**, so this is a rule you cannot discover by testing the command
locally. The LLM API creates its workers with `MPI.COMM_SELF.Spawn`, and
dynamic MPI process spawning is not permitted inside an `srun` step. The
symptom is not a clear rejection — the server simply never becomes ready
while `serve.log` repeats:

```
mpi4py.MPI.Exception: MPI_ERR_SPAWN: could not spawn processes
```

and your readiness poll eventually times out, which reads as "the model is
slow to load" rather than "this can never work".

Three things are required together — one alone is not enough:

```bash
srun --partition=<slurm_partition> \\
     --mpi=pmix \\                      # 1. PMIx bootstrap for the ranks
     --ntasks-per-node=<world_size> \\   # 2. one task PER RANK, not 1
     --container-image=<docker_image> \\
     --container-mounts=<repo>:<repo>,<ckpt>:<ckpt>,<workspace>:<workspace> \\
     --container-workdir=<repo> \\
     --gres=gpu:<num_gpus> \\
  bash -c 'trtllm-llmapi-launch trtllm-serve <ckpt> ...'
#          ^ 3. adopts the ranks srun already created instead of spawning
```

`<world_size>` is the product of the parallel sizes in the tuning config
(`tensor_parallel_size` × `pipeline_parallel_size` × …; `moe_expert_parallel_size`
reuses the TP ranks and does **not** multiply it). It is a property of the
config, so recompute it whenever you change one of those values — leaving
`--ntasks-per-node=1` under a `tensor_parallel_size: 2` config is the
common version of this mistake.

Use `trtllm-llmapi-launch` even at world size 1: the restriction is on
spawning inside a step, not on the number of ranks.

`--gres=gpu:<num_gpus>` may exceed `<world_size>` when the cluster's QOS
enforces a floor (this one requires `--gres=gpu:4`). Allocating more GPUs
than ranks is allowed and simply leaves the extras idle; do NOT raise the
parallel sizes just to consume them, because that changes the configuration
under measurement.

### One STEP, not merely one allocation — `127.0.0.1` is per-step

The server and whatever talks to it (readiness poll, benchmark client) must
run in the **same `srun` step**. One allocation is not enough. Under pyxis
each `srun` gets its own container with its own network namespace, so
`127.0.0.1` inside the client's step is NOT the loopback the server bound —
the connection is refused even though both steps are on the same node and
the server is perfectly healthy.

This failure is a liar. The server log shows `Application startup complete`
and `200 OK` for its own polls, `/v1/models` lists the model, the GPUs show
memory in use — and the client still reports the server as unreachable, so
the natural conclusion is "the server did not come up" when it plainly did.

    sbatch script
      └── srun (ONE step)
            server &        # background, binds 127.0.0.1
            poll /health    # same namespace — this works
            benchmark       # same namespace — this works
            teardown

If you genuinely need a separate step, then the server must bind `0.0.0.0`
and the client must address it by node name (`$SLURMD_NODENAME`), and that
step needs `--overlap` to share the allocation. Prefer the single step: it
has fewer ways to go wrong and needs no extra flags.

### Use a batch script, not an interactive allocation

Prefer the single `sbatch` script form. Do **not** use `srun --pty bash`
or `salloc` and then issue further commands expecting the allocation to
still be there.

The reason is mechanical, not stylistic: each of your `Bash` calls is a
separate process. An interactive allocation belongs to the call that
created it and is gone when that call returns, so a server started in one
call is not running in the next, `serve.pid` names a process that no
longer exists, and the readiness poll on `127.0.0.1:8000` reaches a
different machine's loopback. The failure looks like a server that would
not start.

One script that runs the whole stage — start the server, poll it, run the
benchmark, capture the profiles, tear down — keeps all of that inside one
allocation on one node, which is what the "within one allocation" rule
above actually requires.

Two consequences worth stating:

- **Every path in the script is a CLUSTER path.** `<workspace>`,
  `<trtllm_repo_path>` and `<checkpoint_path>` are already cluster paths,
  so write them verbatim; do not try to translate them.
- **Send the job's own output somewhere shared.** `sbatch -o /tmp/x.out`
  writes to the compute node's local /tmp and disappears with the
  allocation. Point `-o` inside `<workspace>`.
"""


# --------------------------------------------------------------------------- #
# SOL projection derivation (the projector role's methodology blocks — shared
# with perf-optimize, whose projector prompt composes the same fragments)
# --------------------------------------------------------------------------- #

SOL_PROJECTOR_METHODOLOGY = """\
## The methodology: the `internal-perf-sol-analysis` skill

The skill is the single source of truth for SOL modeling — its α-β-u
model, its per-op recipes, its peaks calculator and
`measure_channels.py`, and the ground rules that come with them. Load
it and follow it: the arithmetic you write down instantiates *its*
formulas, not your own, and your report speaks its vocabulary — **% of
SOL** as the headline (latencies: SOL ÷ measured; throughput: measured
÷ SOL — both ≤ 100%), **MFU** / **MBU** as secondary utilizations,
**gap-to-SOL**, and **bound** ∈ compute / memory / launch (plus comm on
multi-GPU).

What the skill cannot know is this stage's contract:

- **`task.yaml`'s `sol.gpu` (when set) is the part-name hint** for
  `sol_calc.py peaks --part`; the `Skill` load announces the base
  directory its scripts live in.
- **A GPU may not be reachable from here.** On local runs the GPUs sit
  idle between stages (confirm with `nvidia-smi`), so measure the
  latency constants and merge them into the peaks file. Under the
  workflow's Slurm mode you run on a login node — do **not** guess α:
  derive the β/u terms and record the launch-α term as unmeasured in
  *Caveats*.
- **Nothing measured exists yet.** `sol_calc.py analyze` correlates
  *measured* per-op times with their ceilings, and no profiling stage
  has run — do not invent `measured_ms` rows, and never fabricate an
  input to force a script run. The **Analyzer** runs `analyze` after
  you, against its fresh profile and the peaks file you persist. Your
  job is the predictive end-to-end ceiling: instantiate the skill's
  formulas and per-op recipes yourself, with **every formula's actual
  numbers written down** — a projection whose arithmetic cannot be
  re-checked from the report is worthless.
- **Persist the machine-readable peaks file for the Analyzer** — the
  peaks-calculator output, with the measured latency constants merged
  in whenever you measured them — to
  `<workspace>/sol_work/peaks.json`, and record that path in
  *Projection setup*. The Analyzer's measured↔SOL correlation joins
  against this exact file; a projection whose peaks live only in prose
  starves that stage.
- **The structural quantities are this deployment's, and the skill's
  recipes do not supply them:** read them off the checkpoint's
  `config.json` (a misread config silently corrupts every downstream
  number — read it carefully) at the serving precision and the
  parallel mapping (tp/pp/ep, from the config named under *Workspace*)
  — weight bytes per GPU (count only active experts per token for
  MoE), KV-cache bytes read per decoded token at the mean context
  length, FLOPs per token, and on multi-GPU the per-layer collective
  term.
- **If the skill is not available in this environment** (neither name
  resolves), say so in one line, ground what you can from `config.json`
  + internal knowledge into a clearly marked coarse ceiling — and if
  nothing defensible can be grounded, write the unavailable form.
  Never fabricate.

One limit the skill does not state and your report must: the ceiling
models **kernel execution plus per-launch latency only** — no
serving-stack scheduler/host prep, no request queueing, no
dynamic-batching effects. A measured result far below even this α-aware
ceiling therefore points at host/scheduling costs the model does not
price; say so explicitly, it is a valuable signal for the downstream
stages.
"""


# Appended to the projector's prompt only when the workflow resolved
# ``perf-analysis`` instead of the skill above (``sol_methodology``). The
# methodology's own last bullet already says what to do without the
# calculator; this names the skill that replaces it and the one artifact
# that stops being writable.
SOL_METHODOLOGY_FALLBACK = """\
## Fallback: `internal-perf-sol-analysis` is not installed here

This session does not have the skill above, so load the `perf-analysis`
skill your driving message names instead and take its
bottleneck-classification table as the methodology. There is no peaks
calculator and no `measure_channels.py`, so follow the last bullet of
*The methodology*: ground what you can from `config.json` + internal
knowledge into a clearly marked coarse ceiling, and say in one line
that the peaks are not calculator-resolved. Skip
`sol_work/peaks.json` — `sol_calc.py` ships with the missing skill, so
nothing downstream reads it. If nothing defensible can be grounded,
write the *Projection unavailable* form. Never fabricate.
"""


SOL_PROJECTOR_INTERNAL_KNOWLEDGE = """\
## Consulting internal knowledge (reference only)

When the mapping is uncertain or a spec is missing — e.g. to
characterize a model architecture or GPU part the skill's references
leave uncertain — use the `internal-glean-search` skill or the
`internal-glean-specialist` subagent for detailed internal knowledge
(if that skill/subagent exists). **It is consultative** — every
projected number in your report must be reproducible from the
arithmetic you wrote down over named sources (the skill's calculator
output, `config.json`, what you retrieved); never copy a number you
cannot derive, and don't burn the turn searching when the derivation
already stands on cited sources.
"""


# --------------------------------------------------------------------------- #
# SOL projection consumption (appended only when the projector stage is enabled)
# --------------------------------------------------------------------------- #

# Role-neutral correlation recipe shared by both workflows' analyzers —
# each workflow's SOL analyzer extension composes it and names the
# artifact directory the regions/sol JSONs land in.
SOL_CORRELATION_METHOD = """\
### Correlate the fresh profile against the ceiling (`sol_calc.py analyze`)

The projection alone is a predictive end-to-end ceiling; your profile
just produced the measured per-op times the Projector did not have.
Join the two with the skill's calculator — the correlation turns "the
workload is at N% of SOL" into a per-op table naming *where* the gap
physically sits:

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
   rollup), rolled up into the skill's region keys and schema. The
   shapes come from `sol_projection.md`'s *Arithmetic* (the Projector already
   derived them from `config.json`) — reuse them rather than
   re-deriving. A region whose params you cannot ground stays in
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
5. **Transcribe `sol.json` into the `## SOL correlation (measured vs
   ceiling)` section** of `profile_findings.md`: the joined per-op
   table verbatim (region, calls, measured ms, SOL ms, % of SOL,
   MFU %, MBU %, gap ms, bound), the workload-level % of SOL line, and
   one sentence naming the largest-`gap ms` regions — that is where
   the headroom physically sits, and it is the sharpest signal the
   downstream stages get from you.

Degrade honestly: when a precondition fails (projection unavailable,
peaks file missing and the constants unmeasurable, nsys produced no
usable per-kernel table), keep the section with a one-line
`Correlation unavailable: <reason>`, record it in *Caveats*, and move
on — a fabricated correlation is worse than none.
"""


SOL_ANALYZER_CONTEXT = (
    """\
## SOL projection as context (the projector stage ran)

The Projector stage ran before you and left `sol_projection.md` — an
analytical speed-of-light (SOL) ceiling for this model/hardware/
operating point, derived with the `internal-perf-sol-analysis` skill,
with a measured-vs-SOL gap analysis. `Read` it (or call
`read_latest_progress` with `agent: "projector"`) after
`benchmark_results.md` and use it as **context, not evidence**:

- Let the projected headroom (% of SOL) and the bound mix (compute /
  memory / launch) inform which hypotheses you probe hardest and how
  you rank them — e.g. a low % of SOL with a memory bound raises the
  prior on memory-bandwidth causes; a gap far beyond what the ceiling
  can explain points at host/scheduling overhead (the ceiling models
  kernel execution plus per-launch latency only, so serving-stack
  scheduler and queueing costs are invisible to it).
- Measured trace evidence always outranks the projection: when they
  disagree, trust the trace and note the disagreement.
- In `profile_findings.md`, say where the profile **confirms or
  contradicts** the projection (a sentence per ranked hypothesis is
  enough).
- Projected numbers are not measurements — never present a SOL number
  as a measured one. If `sol_projection.md` declares itself
  unavailable, ignore it for ranking, skip the correlation below, and
  record that in *Caveats*.

"""
    + SOL_CORRELATION_METHOD
    + """
Artifact placement in this workflow: `regions.json`, `sol.json`, and
any `sol_recipes/` go under `<workspace>/sol_work/`, next to the
Projector's `peaks.json`.
"""
)


SOL_REPORTER_GUIDANCE = """\
## Projection vs Measured (the projector stage ran)

The Projector left `sol_projection.md` — an analytical speed-of-light
(SOL) ceiling derived with the `internal-perf-sol-analysis` skill, with
% of SOL / MFU / MBU numbers and a measured-vs-SOL gap analysis. `Read`
it with the other inputs and add one section to `performance_report.md`,
placed **between "Profiling Findings" and "Main Bottleneck"** (the HTML
companion mirrors it like every other section):

```
## Projection vs Measured

<The measured-vs-SOL table lifted from sol_projection.md (throughput,
TTFT, TPOT with % of SOL, plus measured MFU/MBU), the projected bound
mix (compute / memory / launch), and what the gaps mean: % of SOL sizes
the theoretical headroom, and the bound names which side it is on.
When profile_findings.md carries a **SOL correlation (measured vs
ceiling)** section, also lift its joined per-op table (region / calls /
measured ms / SOL ms / % of SOL / gap ms / bound) — it localizes the
same headroom per op — and name the largest-gap regions; when the
correlation was unavailable, say so in one line rather than
substituting. State explicitly how this projection moves (or does not
move) the verdict.>
```

Weighing rules:
- **Weigh the projection when deciding the Main Bottleneck and when
  ranking Recommendations**: the SOL headroom sizes the win (a fix
  cannot recover more than the ceiling says is available on that side),
  and the projected bound mix corroborates or challenges the Analyzer's
  ranked hypotheses — the per-op correlation table, when present, is
  the sharpest tie-breaker (measured rows against their own ceilings).
  State in the Main Bottleneck section how the projection was weighed.
- The projection is a model, not a measurement — when it conflicts with
  trace evidence, measured evidence wins, and the conflict is worth a
  sentence.
- The ceiling models kernel execution plus per-launch latency only — a
  measured result far below it often indicates serving-stack
  scheduler/queueing costs the model does not price; treat that as
  supporting evidence for host-side bottleneck categories, not as a
  contradiction.
- If `sol_projection.md` is missing or declares itself unavailable,
  the section must honestly say **"Projection unavailable (<reason>)"**
  and the verdict falls back to measured evidence alone — never
  fabricate projected numbers.
"""


# --------------------------------------------------------------------------- #
# Bottleneck taxonomy + HTML companion (reporter)
# --------------------------------------------------------------------------- #

BOTTLENECK_TAXONOMY = """\
## Bottleneck taxonomy

Classify the **single dominant** bottleneck into exactly one primary
category (note secondary factors separately). Tie the verdict to concrete
evidence rows from the benchmark + profile findings — never assert a
category without the signal that supports it.

- **Compute-bound** — GPU math units saturated. Signal: high GPU busy %,
  GEMM/attention/tensor-core kernels dominate kernel time, near-roofline
  FLOPs, throughput scales with batch up to a compute ceiling. Common in
  prefill / large-batch decode.
- **Memory-bandwidth-bound** — HBM bandwidth saturated. Signal: memory-bound
  kernels dominate (elementwise, norms, KV gather/scatter, dequant),
  high DRAM throughput at modest FLOPs, decode-phase TPOT dominated by
  weight/KV reads. Common in low-batch decode.
- **KV-cache-capacity-bound** — serving throughput limited by how many
  requests fit in KV cache. Signal: low KV-cache free-block headroom /
  high utilization, requests queued / preempted, concurrency capped below
  the requested level, throughput rises if `kv_cache_free_gpu_memory_fraction`
  or quantization increases.
- **Kernel-launch / host-overhead-bound** — GPU starved by the host. Two
  distinct sub-causes share this bucket; identify **which one** dominates
  before prescribing a fix, because the fixes differ:
  - *Kernel-launch overhead* — launching/dispatching the model forward
    dominates. Signal: many tiny kernels, launch calls dominate CUDA-API
    time, eager (non-CUDA-graph) execution, idle made of short per-launch
    gaps. This is what **CUDA graphs / overlap scheduler** collapse: graph
    replay wraps the model forward, removing the per-kernel launch cost
    inside it.
  - *Host-prep / scheduler exposed* — a host phase (input preparation,
    block-table/index math, host-device `.item()` syncs, request
    scheduling) runs on the timeline and is not hidden by GPU work.
    Signal: a named host phase (e.g. `_prepare_inputs`) whose wall time
    rivals or exceeds the GPU forward, high `.item()` /
    `cudaStreamSynchronize` counts, long (>100 µs) idle gaps. **CUDA graphs
    do not remove this** — the host prep runs before/around the replayed
    forward, not inside it; the fix is cutting host work and removing
    host-device syncs from the hot path (and when it also blocks graph
    capture, fix it first). Low GPU busy % at low batch is common to both.
- **Communication-bound (multi-GPU)** — collectives dominate. Signal:
  NCCL/all-reduce/all-gather kernels are a large share of time, GPUs wait
  on communication, scaling efficiency drops with TP/PP/EP size.

If two categories are close, say so and rank them; the Executive Summary
still names one headline bottleneck.
"""


HTML_COMPANION = """\
## HTML companion (`performance_report.html`)

Produce a **single self-contained** HTML file alongside the markdown — all
CSS/JS inline, **no external CDN, font, or asset URLs** so it opens
offline. It presents the *same content* as `performance_report.md` (same
sections, same numbers, same verdict) in a clean, interactive form.

**Required structure (top-down):**

1. `<!DOCTYPE html>` with `<html lang="en">`, a `<title>` matching the
   report's H1, and `<meta name="viewport">`.
2. Inline `<style>`: clean readable font stack, generous line-height,
   ~800–900 px max content width, and light/dark mode via
   `@media (prefers-color-scheme: dark)`.
3. A **sticky table-of-contents nav** listing every H2, each linking to
   the section's slugified anchor id (`#executive-summary`, etc.).
4. The main `<article>` body, sections in the same order as the markdown
   (Executive Summary, Configuration, Benchmark Results, Pareto Curve —
   Pareto-curve mode only, Profiling Findings, Main Bottleneck,
   Recommendations), each heading carrying a stable id.
5. Metric tables are real HTML `<table>`s (same columns/values as the
   markdown). The **Main Bottleneck** verdict is visually prominent
   (e.g. a callout box).
6. Inline `<script>` at the end of `<body>`.

**Required charts (self-contained — no chart library, no CDN):** embed
each chart's data as a JSON array in the inline script and render it to
inline SVG with your own small renderer. Style via CSS variables so both
color schemes stay readable, and never plot a value that differs from
the section's table — the table is the source of truth.

- **Top-kernel share bars** — at the top of *Profiling Findings*: one
  horizontal bar per row of the top-kernels table (GPU-time share,
  sorted descending), each labeled with the kernel name — abbreviate
  template-heavy names to a distinctive stem — and its share, with a
  hover tooltip (an SVG `<title>` is enough) carrying the full name and
  exact value. Render it only when the findings carry a top-kernels
  table (nsys ran); with no table, omit the chart rather than plotting
  invented numbers. Further charts (e.g. top operators) are welcome
  under the same self-contained rules.
- **Pareto curve** — at the top of *Pareto Curve*, only in Pareto-curve
  mode (`benchmark.concurrency` is a list): **x = tok/s/user,
  y = tok/s/gpu**, the measured curve as one polyline with a marked
  point per concurrency, each labeled `c=<n>` and carrying an SVG
  `<title>` tooltip with the exact x/y values. Pad both axis domains
  around the data (do not force zero) and put the axis names + units on
  both axes. When the report also carries per-point SOL-projected
  values (the projector ran in curve mode), overlay the projected curve
  as a second polyline distinguished by more than hue alone (e.g.
  dashed) plus a legend. In scalar mode, or when the curve summary
  table is absent, omit the chart and the section.

**Required interactivity:**

- **TOC scroll-spy** — the entry for the section in view gets an `active`
  class as the reader scrolls.
- **Collapsible H2 sections** — clicking a heading toggles a `collapsed`
  class on its body; default expanded.
- **Print-friendly** — hide the TOC and force-expand all sections in
  `@media print`.

**Faithfulness rule:** the HTML is not a remix — same sections, same
tables, same bottleneck verdict and evidence as the markdown, and charts
that plot exactly the numbers in the tables they sit above. If you
revise the markdown, revise the HTML in the same turn.
"""


# --------------------------------------------------------------------------- #
# Shared rigor / progress rules
# --------------------------------------------------------------------------- #

EVIDENCE_DISCIPLINE = """\
## Evidence discipline

- **Never fabricate numbers.** Every metric, kernel name, or percentage
  you report must come from a file you actually produced (the benchmark
  JSON, `nsys stats` output, the ncu report, server logs). If a run
  failed or a tool was unavailable, say so plainly — do not invent
  plausible-looking results.
- **Record exact commands.** Anyone reading the workspace must be able to
  reproduce your run from the commands you wrote down.
- **No conversational filler.** Jump straight into the work.
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
