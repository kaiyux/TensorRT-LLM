"""Optimization-specific prompt contracts and shared perf-analyze recipes.

This workflow separates profiling from offline analysis. SOL guidance is
conditional on the projector stage; kernel coverage guidance is conditional
on ``profile.kernel_coverage``.
"""

from typing import Sequence

from agent_flow.workflows.perf_analyze.prompts._common import (
    BENCHMARK_FLAGS_REFERENCE,
    BOTTLENECK_TAXONOMY,
    CASEBOOK_CONSULTATION,
    DERIVED_METRICS_REFERENCE,
    EVIDENCE_DISCIPLINE,
    EXECUTION_SLURM_BOOTSTRAP,
    PROFILE_FINDINGS_CONTRACT,
    PROFILING_KNOB_VERIFICATION,
    PROFILING_RUNS_REFERENCE,
    REMOTE_SLURM_EXECUTION,
    SOL_CORRELATION_METHOD,
    SOL_METHODOLOGY_FALLBACK,
    SOL_PROJECTOR_INTERNAL_KNOWLEDGE,
    SOL_PROJECTOR_METHODOLOGY,
    build_server_lifecycle,
)
from agent_flow.workflows.perf_optimize.roadmap_schema import APPROACHES

__all__ = [
    "BENCHMARK_FLAGS_REFERENCE",
    "BOTTLENECK_TAXONOMY",
    "CASEBOOK_APPLY",
    "CASEBOOK_CONSULTATION",
    "DERIVED_METRICS_REFERENCE",
    "DISAGG_ANALYSIS_CONTEXT",
    "DORMANT_CAPABILITY_SWEEP",
    "EVIDENCE_DISCIPLINE",
    "EXECUTION_SLURM_BOOTSTRAP",
    "EXPECTATION_GATE",
    "GIT_DISCIPLINE",
    "KERNEL_COVERAGE_REPORTER_GUIDANCE",
    "KERNEL_REUSE",
    "KERNEL_REUSE_ANALYZER",
    "MEASUREMENT_PROTOCOL",
    "MEASUREMENT_METRICS",
    "MEASUREMENT_VALIDITY",
    "OPTIMIZE_HTML_COMPANION",
    "PROFILE_FINDINGS_CONTRACT",
    "PROFILING_KNOB_VERIFICATION",
    "PROFILING_RUNS_REFERENCE",
    "ROADMAP_SPEC",
    "ROADMAP_READER",
    "REMOTE_SLURM_EXECUTION",
    "SERVE_FLAGS_REFERENCE",
    "SERVER_LIFECYCLE",
    "RUNTIME_CHECKOUT",
    "SOL_ANALYZER_CONTEXT",
    "SOL_CORRELATION_METHOD",
    "SOL_METHODOLOGY_FALLBACK",
    "SOL_OPTIMIZER_CONTEXT",
    "SOL_PROFILER_CONTEXT",
    "SOL_OPTIMIZE_REPORTER_GUIDANCE",
    "SOL_PROJECTOR_INTERNAL_KNOWLEDGE",
    "SOL_PROJECTOR_METHODOLOGY",
    "TUNING_CONFIG_NOTE",
    "approach_restriction_note",
    "build_offline_analysis_reference",
    "kernel_coverage_analyzer_note",
    "kernel_coverage_ncu_targeting",
]


# Render the active-config policy directly; the perf-analyze default allows
# startup recovery to alter config, which invalidates independent measurements.
SERVER_LIFECYCLE = build_server_lifecycle(active_tuning_config=True)

SERVE_FLAGS_REFERENCE = """\
### Server configuration

Serve `checkpoint_path` with `--backend pytorch` at `127.0.0.1:8000`.
Pass `--extra_llm_api_options <active tuning config>` with the exact path
in the turn instructions. All server tuning, including parallel sizes,
comes from that YAML. Verify fields against `trtllm-serve --help`
and the LLM API reference in the active runtime checkout.
"""

RUNTIME_CHECKOUT = """\
## Verify the runtime checkout

Before editing source or launching a server, prepend the turn's active
runtime checkout to `PYTHONPATH` in the launch environment and shell. Verify:
```bash
python -c "import tensorrt_llm, os; print(os.path.realpath(tensorrt_llm.__file__))"
```
The printed path must resolve under the active runtime checkout; otherwise
stop and record a blocker without benchmarking or claiming the change ran.
Record the resolved import path and source/build identity beside the exact
serve command. `task.yaml`'s `trtllm_repo_path` is the campaign checkout;
an item, integration worktree or staged remote copy may use another path.
For remote execution, verify the staged path inside the allocated container.
For disaggregated serving, propagate the path and `PYTHONPATH` into every
worker role and verify imports in each worker's launch environment.
"""


def build_offline_analysis_reference() -> str:
    """Reuse shared interpretation recipes without server or capture instructions.

    These section boundaries select the canonical export/decomposition
    instructions; perf-analyze keeps its combined capture/analysis workflow.
    All generated files go into the analysis directory, preserving reports.
    """
    decomposition = PROFILING_RUNS_REFERENCE.split("5. **Decompose the timeline", 1)[1].split(
        "### Multi-GPU:", 1
    )[0]
    decomposition = "1. **Decompose the timeline" + decomposition
    decomposition = decomposition.replace(
        "<workspace>/server_nsys.nsys-rep", "<profile_dir>/server_nsys.nsys-rep"
    ).replace(
        "   cp ",
        "   nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_trace \\\n"
        "       <workspace>/server_nsys.sqlite > <workspace>/nsys_stats.txt\n"
        "   cp ",
        1,
    )
    decomposition = decomposition.replace(
        "   - Analyze Run A’s timing capture now; do not wait for A2. If the skill\n",
        "   - Analyze the saved timing capture even if A2 is unavailable. If the skill\n",
    )
    additional = PROFILING_RUNS_REFERENCE.split("### Consume the additional captures", 1)[1].split(
        "## Run B", 1
    )[0]
    additional = additional.replace(
        "<workspace>/<name>.nsys-rep", "<profile_dir>/<name>.nsys-rep"
    ).replace("Then rerun Run A step 5", "Then rerun the decomposition above")
    ncu_exports = PROFILING_RUNS_REFERENCE.split(
        "5. Tear down to finalize the report, then export details and CSV:", 1
    )[1]
    ncu_exports = ncu_exports.replace(
        "<workspace>/server_ncu.ncu-rep", "<profile_dir>/server_ncu.ncu-rep"
    ).replace("6. Interpret captured kernels", "Interpret captured kernels")
    return "\n\n".join(
        (
            """\
## Offline analysis of saved captures

Use the manifest's artifact paths, ranks and operating points in these
examples. `<profile_dir>` is read-only; `<workspace>` is the current
analysis directory. Analyze raw `.nsys-rep` / `.ncu-rep` reports or their
`.sqlite` / raw CSV exports. Treat preliminary taxonomy/decomposition as
hints. If a raw report is missing, copy an existing export into the analysis
directory. Never run a workload to fill missing evidence; analysis needs no GPU.
""",
            decomposition.strip(),
            "### Interpret saved utilization and call-stack captures\n" + additional.strip(),
            """\
For call stacks, inspect `SAMPLING_CALLCHAINS` and report unresolved
symbols. A `cudaGraphLaunch` stack identifies the graph launch site,
not the internal kernels. Keep A2 timing separate from the timing trace.

### Interpret saved ncu reports

Load `perf-nsight-compute-analysis` via the `Skill` tool, using
`trtllm-agent-toolkit:perf-nsight-compute-analysis` if needed. Its
thresholds and escalation interpretation determine bound class,
occupancy, stalls and utilization; never invent its thresholds. If the
skill is unavailable, retain raw metrics and mark ncu-derived
classification unavailable. A required ledger bound may instead use
source/timeline-supported inference with explicit provenance.
Export each saved report and pass into this analysis directory:
"""
            + ncu_exports.strip(),
        )
    )


# --------------------------------------------------------------------------- #
# The roadmap contract (analyzer / optimizer / evaluator / qa / reporter)
# --------------------------------------------------------------------------- #

ROADMAP_SPEC = """\
## The roadmap contract (`roadmap.yaml`)

```yaml
version: 1
target_metric: output_throughput      # key in the benchmark_serving result JSON
baseline:                             # analyzer writes this once in round 1; frozen afterward
  value: 1234.5                       # curve mode: the MEAN of curve[].value
  source: baseline/benchmark_results.md
  curve:                              # curve mode only (benchmark.concurrency is a list);
    - {concurrency: 8, value: 812.0, tok_s_user: 21.4, tok_s_gpu: 101.5}   # ascending,
    - {concurrency: 32, value: 1657.0, tok_s_user: 12.9, tok_s_gpu: 207.1} # one entry per point
current_best:                         # the last ACCEPTED measurement; seeded equal to baseline
  value: 1298.7                       # curve mode: mean across points; carries curve too
  source: rounds/round_1/attempt_1/evaluation.md
items:                                # pending items ordered by expected benefit, descending
  - id: opt-001                       # stable slug; never renumbered or reused across rounds
    title: Enable CUDA graphs for decode
    category: launch-host             # compute | memory-bw | kv-capacity | launch-host | communication
    approach: config                  # config (tuning YAML only) | code (source edit)
    evidence:                         # >= 1 entry; cite trace files with numbers
      - "nsys: 31% GPU idle from per-launch gaps (rounds/round_1/analysis/nsys_stats.txt)"
    casebook_ref: "launch storm at decode -> cuda-graph capture"   # optional; the matched casebook row
    expected_gain_pct: 12.0
    expected_gain_rationale: "idle share x casebook-typical recovery for this pattern"
    how_to_apply: |
      Add cuda_graph_config to tuning/extra_llm_api_options.yaml; no source edit.
    status: pending                   # pending | in_progress | accepted | failed | obsolete
    attempts: 0
    measured_gain_pct: null           # filled from the evaluator's measurement
nsys_items:                           # coverage of nsys_analysis/items.json; one row per id there
  - id: nsys-01                       # the id verbatim from items.json
    disposition: item                 # item | dismissed
    ref: opt-001                      # the roadmap item id, or the evidence for dismissing it
  - id: nsys-02
    disposition: dismissed
    ref: "0.2 ms/iter is below the noise floor at this operating point"
```

Rules:

- **Enum fields are exact.** Use only the categories, approaches and
  statuses above (`memory-bw`, never `memory`). Invalid enums abort the run.
- **List order is priority order.** Sort pending items by descending
  `expected_gain_pct`; the orchestrator selects from the front.
- **Evidence:** cite artifacts and recovery arithmetic, following the
  analyzer's evidence contract and dormant-capability exception.
- **`nsys_items` covers exactly the ids in `nsys_analysis/items.json`.**
  Omit it when that file is unavailable and state why. An `item` ref names
  a real roadmap id in any status; a `dismissed` ref supplies evidence.
  Full analyses rebuild this block from their own file; replan-only rounds
  use the standing analysis. `nsys-NN` ids restart at `nsys-01` in each
  analysis. Reassess reused judgments against the current ids, citing
  earlier evidence where applicable. Missing or extra ids fail validation.
- **Initialization and ids.** In round 1, populate `baseline` from
  `baseline/benchmark_results.md` and seed `current_best` identically,
  including any curve. Thereafter freeze `baseline` and
  preserve all accepted / failed / in_progress items. Never renumber or
  reuse ids; allocate fresh ids continuing the sequence.
- **Ownership.** Only the **analyzer** writes item content and ordering,
  and may mark pending items `obsolete` when new evidence warrants it.
  The **orchestrator** owns lifecycle fields: `in_progress` / `accepted` /
  `failed` status, `attempts`, `measured_gain_pct` and `current_best`.
  Agents never edit those fields or rewrite accepted/failed history.
- **Curve mode** (`benchmark.concurrency` is a list): include one ascending
  `{concurrency, value, tok_s_user, tok_s_gpu}` row per configured point.
  Absolute `value` is the mean of per-point values; `measured_gain_pct`
  is the **mean of per-point gains**.
  The orchestrator advances `current_best` from validated serial promotions
  or accepted parallel integration measurements. Scalar runs omit `curve`.
- **Focus scoring** (`optimize.focus_concurrencies`, optional, curve mode
  only): scalar values and gains average **only those points**; otherwise
  they average all points. Every curve and regression check still covers
  **all** configured points. Estimate expected gains on the scored regime.
"""

ROADMAP_READER = """\
## The roadmap contract (`roadmap.yaml`)

Read the target metric, frozen `baseline`, `current_best`, and any item
named in the turn instructions: its approach (`config` or `code`), evidence,
`expected_gain_pct`, `expected_gain_rationale`, `how_to_apply` and history.
**List order is priority order**, not execution chronology.

Treat the roadmap as read-only. The **orchestrator** owns status, attempts,
measured_gain_pct and current_best; only the Analyzer authors or replans
item content.

`current_best` is the accepted campaign measurement. A parallel batch's
candidates share a frozen reference: evaluator APPROVE makes an item
candidate-ready. Only promotion of the integrator's measured state accepts
items and advances current_best.
Standalone candidate gains are not successive campaign improvements.

In curve mode, every `curve` contains all configured concurrency points as
`{concurrency, value, tok_s_user, tok_s_gpu}` rows. Absolute scalar values
are means over `optimize.focus_concurrencies` when set, else all points;
gain is the mean of direction-normalized per-point gains over the same
scored subset. All points remain subject to the regression check.
"""


# --------------------------------------------------------------------------- #
# Git discipline (optimizer / evaluator)
# --------------------------------------------------------------------------- #

GIT_DISCIPLINE = (
    """\
## Git discipline (the TRT-LLM checkout)

The orchestrator owns the optimization branch in `trtllm_repo_path`,
committing accepted items and reverting rejected attempts with
`git reset --hard` + `git clean -fd`:

- **Never run `git commit`, `git reset`, `git checkout`/`git switch`,
  `git stash`, or `git push`.** Use read-only `git diff`, `git status`
  and `git log` to inspect the change.
- Keep **only the current roadmap item's changes** in the worktree;
  the whole worktree is committed on accept and wiped on reject.
- **The code must stand on its own in the TRT-LLM repo.** Comments,
  docstrings and names must not reference roadmap ids (`opt-008`),
  rounds/attempts, workspace files, benchmark results or this workflow.
  Explain non-obvious constraints in the repo's terms; put experiment
  provenance in `optimization_summary.md`.
- Review uncommitted changes with `git -C <active runtime checkout> diff`
  and `--stat`. List added files from `git status --porcelain` in your output.
"""
    + RUNTIME_CHECKOUT
)


# --------------------------------------------------------------------------- #
# Kernel reuse (analyzer / optimizer / evaluator)
# --------------------------------------------------------------------------- #

KERNEL_REUSE_ANALYZER = """\
## Prefer existing kernels over writing new ones

Before planning kernel work, search in priority order:
1. The TRT-LLM checkout's custom ops, kernels and gated paths.
2. The installed **flashinfer** version and its existing TRT-LLM call sites.
3. Other integrated providers: CUTLASS / cuBLAS / cuDNN, DeepGEMM and
   vendored Triton ops.

Name the suitable existing kernel/op and source location in `how_to_apply`.
If none fits, plan a scoped **new kernel**, name its required convention/API,
and record the search. Missing reuse alone does not dismiss an opportunity.
"""

KERNEL_REUSE = (
    KERNEL_REUSE_ANALYZER
    + """
- **Optimizer** — repeat the search before implementing. If no suitable
  implementation exists, write a scoped kernel. Explain why nothing fit
  in the summary's *Mapping to the roadmap item* section.
- **Evaluator** — a new kernel fails code quality when a suitable
  existing kernel exists, regardless of gain: PUSH_BACK
  with `reason_category: code_quality`, directing the optimizer to wire
  up the existing kernel (REJECT only when out of retries). When the
  recorded search confirms none exists, apply the normal scoped-diff,
  correctness, targeted-test and measured-gain checks.
"""
)


# --------------------------------------------------------------------------- #
# Dormant-capability sweep (analyzer)
# --------------------------------------------------------------------------- #

DORMANT_CAPABILITY_SWEEP = """\
## Dormant-capability sweep (round 1)

Before writing the round-1 roadmap, inspect disabled capabilities absent
from traces:

1. **Checkpoint config** (`config.json` under `checkpoint_path`):
   speculative-decode / multi-token-prediction heads
   (`mtp_num_hidden_layers`, `num_nextn_predict_layers`, eagle/draft
   blocks) and cache/precision hints. Confirm matching tensors in the
   checkpoint's weight index
   (e.g. `mtp.*` entries in `model.safetensors.index.json`).
2. **Serving config**: knobs the live tuning YAML leaves unset whose
   default disables a capability the checkpoint ships (e.g. no
   `speculative_config` while an MTP layer sits in the weights), and
   documented backend/strategy selectors still on their generic default.
3. **Model code**: env-gated or condition-gated paths in the model's
   modeling file(s) in the checkout that default OFF for this deployment
   shape — `grep -n "environ" <modeling files>` via `Bash`, then read
   each gate's condition against the live config (TP/EP/attention-DP,
   quant mode) to check applicability.

For each surface, produce exactly one of:

- a **roadmap item** — when its approach is allowed and the mechanism
  plausibly helps this workload. Ground `expected_gain_pct` in the current
  theoretical model, casebook or published behavior. In `evidence`, identify
  the dormant lever and verification (config key, weight names, gate);
- or a one-line **dismissal with evidence** — wrong hardware, a gate
  that is provably correct to keep off, a capability incompatible with
  the workload or the campaign's accuracy scope. Never dismiss for "no
  trace evidence".

Keep the sweep outcome in `dormant_capabilities.md`: one line per surface
with its disposition (item id or evidence-backed dismissal), or "none found".
Link material opportunities from `analysis.md`'s Next actions. Revisit in
later rounds only when an accepted item changes what is reachable.
"""


# --------------------------------------------------------------------------- #
# Acceptance gate (evaluator)
# --------------------------------------------------------------------------- #

EXPECTATION_GATE = """\
## The acceptance gate — APPROVE, PUSH_BACK, or REJECT

APPROVE only when all three axes pass. Both negative verdicts revert the
attempt and require one `reason_category`:

- **PUSH_BACK** — name a concrete fix. Retries are bounded by
  `optimize.max_attempts_per_item`; on the **final attempt**,
  decide APPROVE or REJECT (PUSH_BACK is treated as REJECT).
- **REJECT** — no useful retry remains: broken premise, unresolvable
  blocker, inapplicable knob/mechanism or fundamental regression.
  The item becomes `failed` and the loop moves on.

1. **Code quality** (`reason_category: code_quality`) — minimal scoped
   diff, surrounding style, no obvious bugs, dead code or debug leftovers;
   follow *Git discipline* and *Prefer existing kernels*. For
   `approach: config`, change only documented tuning YAML keys.
2. **Functionality** (`reason_category: functionality`) — the server
   starts and serves coherent completions with the change applied. For
   `approach: code` items, also run the narrowest relevant tests in the
   TRT-LLM checkout when a targeted test exists; a server crash, garbage
   output, or a failed targeted test always fails this axis.
3. **Perf expectation** (`reason_category: perf_shortfall`) — use the
   measurement protocol **against the frozen reference named in your turn
   instructions** (`current_best` at the item's base, not the original
   baseline unless still current best). Read `accept_fraction` and
   `noise_floor_pct` from `task.yaml`'s `optimize` block, and
   `expected_gain_pct` from the roadmap item.

   **Single operating point** (`benchmark.concurrency` is an integer) —
   compute the gain against `current_best.value`; the attempt passes iff:

   ```
   measured_gain_pct >= accept_fraction × expected_gain_pct
   AND measured_gain_pct >= noise_floor_pct
   ```

   **Curve mode** (`benchmark.concurrency` is a list) — apply the
   **Pareto gate** to signed, direction-normalized per-point gains:

   ```
   gain_i          = per-point gain vs current_best.curve[concurrency = c_i]
   mean_gain_pct   = arithmetic mean of gain_i over the SCORED points
   regression_bar  = optimize.max_regression_pct when task.yaml sets it,
                     else noise_floor_pct

   PASS iff  mean_gain_pct >= accept_fraction × expected_gain_pct
         AND mean_gain_pct >= noise_floor_pct
         AND every gain_i >= -regression_bar   # no point (scored or not) regresses beyond the bar
   ```

   The measurement protocol defines scored points and curve validity.
   In `evaluation.md`, show all per-point rows, the scored mean and all
   three conditions. With focus scoring, show the all-points mean too
   and state that the focus mean gated. If an accepted point uses an
   explicit regression budget beyond the noise floor, name the point,
   signed regression and budget in the Verdict and progress `summary`.

   In both modes, show reference/measured values and threshold arithmetic.

On APPROVE, `reason_category` is `"none"`. Report `measured_gain_pct`,
`measured_value` and any `curve` exactly as scored by the protocol; never
round up a gain. The orchestrator records them on promotion. In parallel
mode APPROVE makes a candidate ready for integration; it does not advance
`current_best` or accept the item.
"""


# --------------------------------------------------------------------------- #
# Measurement protocol (benchmarker / evaluator / qa)
# --------------------------------------------------------------------------- #

MEASUREMENT_METRICS = """\
## Measurement protocol

- Metric keys in the result JSON: `output_throughput` (output tok/s — the
  default target metric), `total_token_throughput`, `request_throughput`,
  and latency keys `mean_ttft_ms` / `median_ttft_ms` / `p99_ttft_ms`
  (likewise `*_tpot_ms`, `*_itl_ms`, `*_e2el_ms`). The active target is
  `optimize.target_metric` in `task.yaml`.
- **Direction rule:** throughput metrics are better when higher; `*_ms`
  latency metrics are better when lower. Always report `gain_pct`
  normalized so **positive = improvement**:
  - throughput: `gain_pct = (new − reference) / reference × 100`
  - latency (`*_ms`): `gain_pct = (reference − new) / reference × 100`
- Name the reference beside every gain. In curve mode, compare each point
  with its same-concurrency reference. Average gains over
  `optimize.focus_concurrencies` when set (the scored subset), else all
  points. Profiling replays alone do not provide a scored curve measurement.
"""

MEASUREMENT_VALIDITY = """\
## Valid measurement evidence

Use completed benchmark result JSON from an unprofiled run. The target
metric and reference must be finite positive numbers; null, strings,
NaN, infinity, zero and failed/empty runs cannot establish a passing gate.
In curve mode measure every configured concurrency exactly once, with no
missing, duplicate or extra points. Both the measured and reference curves
must be complete. Never replace a missing curve with a scalar comparison.

`measured_gain_pct` is the arithmetic mean of per-point `gain_i` over the
scored points; `measured_value` is the mean absolute value over those same
points. Keep all `{concurrency, value, tok_s_user, tok_s_gpu}` rows in the
structured `curve`, ascending. Show scored and all-points means with focus scoring.

For an acceptance verdict, the scored gain must reach the turn's required
gain threshold and `optimize.noise_floor_pct`. In curve mode every point,
including unscored points, must satisfy `gain_i >= -regression_bar`, where
`regression_bar = optimize.max_regression_pct` when configured, else
`optimize.noise_floor_pct`. Report any point that spends this explicit
regression budget. An invalid or missing measurement never passes.
"""


MEASUREMENT_PROTOCOL = (
    MEASUREMENT_METRICS
    + "\n"
    + MEASUREMENT_VALIDITY
    + """

Curve example (`output_throughput`, all points scored): reference at
c=8/32/128 is 812.0/1657.0/2210.0; measured is 846.1/1755.2/2201.2.
Gains: +4.20%, +5.93%, −0.40%; mean +3.24%. With expected gain 5%,
accept_fraction 0.5, noise floor 1% and no explicit regression budget:
3.24 ≥ 0.5×5 = 2.5; 3.24 ≥ 1; worst point −0.40 ≥ −1. All gates pass.
"""
)


# --------------------------------------------------------------------------- #
# The live tuning config (all server-launching roles)
# --------------------------------------------------------------------------- #

TUNING_CONFIG_NOTE = """\
## The active tuning config

Server tuning is **owned by the workspace**. The turn's **active tuning
config** path supersedes `tuning/extra_llm_api_options.yaml` shorthand.
Always serve with that exact path, including when its content is `{}`.
- The **optimizer** may edit its item's active tuning config. The
  **integrator** may combine candidate configs and make minimal combination
  fixes only in its isolated integration config. Every other role treats
  the active tuning config as read-only and serves with it as-is.
- Never edit the orchestrator-managed accepted config snapshot named in
  the turn; it restores the active tuning config after a rejected attempt.
"""


# --------------------------------------------------------------------------- #
# Disaggregated serving (every role that launches or measures a server).
#
# Composed LAST in each role prompt so its overrides win: it supersedes
# the single-server lifecycle, the tuning-config note, and the profiling
# runs above it, the same way TUNING_CONFIG_NOTE supersedes the
# extra_llm_api_options guidance. It is conditional on `task.yaml` — an
# aggregate campaign reads it, finds no `disagg:` block, and ignores it.
# --------------------------------------------------------------------------- #

DISAGG_ANALYSIS_CONTEXT = """\
## Disaggregated capture interpretation

This campaign has context and generation worker groups. Read the source
manifest and disaggregated harness/config snapshots as evidence; never
submit the harness or launch workers. Preserve each trace's worker role,
instance and original rank IDs when decomposing it. Context and generation
iteration windows use different clocks; compare matching operating points
and do not treat the two roles as interchangeable ranks. KV-cache transfer
from context to generation is a first-class `communication` cost. The
harness supports nsys only: record `not available in a disagg campaign`
for ncu and reason from saved nsys plus source evidence.
"""


DISAGG_CAMPAIGN = """\
## Disaggregated serving (supersedes the server-lifecycle, tuning, and profiling guidance above)

**This campaign is disaggregated.** A Slurm job owns launch, readiness and
teardown; do not launch `trtllm-serve` or poll `:8000` yourself. The harness
runs the canonical benchmark command above with the same flags.

### Inputs

- **harness config** (`task.yaml`'s `disagg.config`) — cluster,
  environment, measurement conditions. Read-only.
- **`task.yaml`** — the campaign knobs (`optimize`). Read-only.
- **active tuning config named in the turn instructions** — the harness
  config's `worker_config`, i.e. `{ctx: {...}, gen: {...}}`. The optimizer
  edits its item's config; the integrator combines candidates only in its
  isolated integration config. Other roles keep it read-only.

### Per launch

1. Write `<your artifact dir>/disagg_config.yaml` from the harness config,
   replacing `worker_config` with the active tuning file's contents.
   For parallel items add `--exclusive` to this run-local copy's
   `slurm.extra_args`, preserving existing arguments. Each item needs its
   own allocation, exclusive node set and job directory; never reuse a
   sibling's allocation. Keep the original harness config read-only and
   worker topology frozen.
2. Submit and poll in the foreground until the job leaves the queue:
   ```bash
   cd <trtllm_repo_path>/examples/disaggregated/slurm/benchmark
   python submit.py -c <your artifact dir>/disagg_config.yaml \\
       --log-dir <your artifact dir>/bench      # --dry-run to check the node math first
   squeue -j <id>                               # blocking loop; never yield your turn
   ```
3. Read every metric from `<log-dir>/concurrency_<c>/result.json`. On
   failure read `<log-dir>/slurm-<id>.{out,err}` and the per-role worker
   logs before resubmitting — a retry costs another allocation.

The job tears its own cluster down; never kill workers by PID. To abandon
a run, `scancel <id>`.

### Traps that cost an allocation

- `--log-dir` must be a **fresh** path: `submit.py` wipes it if it exists
  without a `trtllm_config.yaml`.
- `environment.work_dir` is where `slurm.script_file` is resolved, **not**
  an output dir. Leave it as the campaign set it.

### `num_gpus` and the frozen topology

`num_gpus` = **sum over roles** =
`num_ctx_servers x (ctx tp x pp x cp) + num_gen_servers x (gen tp x pp x cp)`.

Worker counts and per-role parallel sizes are **frozen**: any change is a
REJECT regardless of gain. If `num_gpus` differs from baseline, stop and
report an invalid comparison. Other role settings (batch sizes,
token limits, KV-cache, MoE, `cache_transceiver_config`, scheduling,
speculative decoding) are normal `approach: config` work.

### Profiling

nsys only, and the harness owns it — in the config you synthesize:

```yaml
profiling:
  nsys_on: true
  gen_profile_range: <iteration window>   # generation workers
  ctx_profile_range: <iteration window>   # context workers
```

- It wraps **workers only** (never the router or the benchmark client),
  writing `<log-dir>/nsys_worker_proc_<ROLE>_<instance>_<procid>.nsys-rep`.
- Profile generation workers by default for steady-state decode. Label
  each trace's role.
- Choose each window from the operating point — the roles count
  iterations on different clocks and `profile.nsys_iter_range` is only a
  default. State the windows you used.
- ncu is unsupported: record `not available in a disagg campaign` and
  plan from nsys.
- Classify KV-cache transfer (ctx to gen) as `communication`.
"""


# --------------------------------------------------------------------------- #
# Actionable casebook variant (optimizer)
# --------------------------------------------------------------------------- #

CASEBOOK_APPLY = """\
## Apply from the optimization casebook (load it early)

After reading the roadmap item, load `perf-optimization-casebook` via
`Skill`, or `trtllm-agent-toolkit:perf-optimization-casebook` if needed.

Use the item's `casebook_ref` or matching bottleneck signal to select a case.
Follow its application, accuracy, verification and rollback guidance,
adapting it to this config. The item's `how_to_apply` overrides conflicting
casebook guidance; note the divergence in your summary.

If unavailable, note it once and proceed from `how_to_apply`.
"""


# --------------------------------------------------------------------------- #
# HTML companion (reporter) — adapted from perf-analyze's HTML_COMPANION
# --------------------------------------------------------------------------- #

OPTIMIZE_HTML_COMPANION = """\
## HTML companion (`optimization_report.html`)

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
# SOL projection consumption (appended only when the projector stage is enabled)
# --------------------------------------------------------------------------- #

SOL_PROFILER_CONTEXT = """\
## Capture measured SOL constants when needed

Use `sol_projection.md` to locate the peaks file and missing measurements.
If `<campaign_workspace>/sol_work/peaks.json`
exists but lacks measured `latencies`/`sms`, load
`internal-perf-sol-analysis` (fully-qualified
`trtllm-agent-toolkit:internal-perf-sol-analysis` if needed) and, in the
profiling GPU environment with servers stopped and the GPU idle, run its
`measure_channels.py --launch … --merge-into <campaign peaks.json>`.
Record the command and outcome; unavailable skill/GPU measurements are
manifest limitations. This capture-stage calibration may update missing
constants in the campaign peaks file; never change existing measurements
because an optimization failed. Keep other campaign analysis/planning
artifacts read-only. Save the peaks used as `sol_peaks.json` in the profile
directory and list it under the manifest's top-level `artifacts`.
The offline Analyzer performs correlation, not measurement.
"""

_OFFLINE_SOL_CORRELATION_METHOD = (
    SOL_CORRELATION_METHOD.replace("the fresh profile", "the selected capture")
    .replace(
        "your profile\njust produced the measured per-op times",
        "the selected capture\nsupplies the measured per-op times",
    )
    .replace(
        "   `sol_projection.md`'s *Projection setup*). When it carries no\n"
        "   measured `latencies`/`sms` — the Projector ran without GPU reach —\n"
        "   run the skill's `measure_channels.py --launch … --merge-into <that\n"
        "   peaks.json>` yourself: unlike the Projector's stage, a GPU is\n"
        "   reachable here by construction (you just profiled on it).",
        "   `sol_projection.md`'s *Projection setup*). Use saved measured\n"
        "   `latencies`/`sms`, checking the profile's `sol_peaks.json` snapshot\n"
        "   when present. If constants are missing, label them unmeasured\n"
        "   and record the limit; when correlation requires them, report\n"
        "   `Correlation unavailable: missing measured constants`. Request\n"
        "   any necessary measurement from the Profiler in your summary.\n"
        "   Never run a GPU microbenchmark or invent missing constants.",
    )
    .replace("<workspace>/sol_work/peaks.json", "<campaign_workspace>/sol_work/peaks.json")
    .replace(
        "5. **Transcribe `sol.json` into the `## SOL correlation (measured vs\n"
        "   ceiling)` section** of `analysis.md`: the joined per-op\n",
        "5. **Present `sol.json` in the report's comparison section** (selected\n"
        "   by the findings structure in your instructions): the joined per-op\n",
    )
)

SOL_ANALYZER_CONTEXT = (
    """\
## Update the current model from SOL evidence

Check `sol_projection.md`'s structural derivations against the selected
capture, then feed valid per-op results into `performance_model.yaml` under
the shared model contract. Keep kernel-level revisions in `kernel_ledger.yaml`
when present. State unavailable correlation beside the affected model.
"""
    + _OFFLINE_SOL_CORRELATION_METHOD
    + """\
Keep `regions.json`, `sol.json` and `sol_recipes/` in the current analysis/
directory, with peaks at <campaign_workspace>/sol_work/peaks.json.
Re-run correlation in full analysis/re-analysis using selected captures;
replan-only turns preserve standing measurements and may revise predictions
from new facts without collecting runtime evidence.
"""
)

SOL_OPTIMIZER_CONTEXT = """\
## Current model alignment (the projector stage ran)

Read the current `performance_model.yaml` and `analysis.md` alongside the
roadmap item. `sol_projection.md` is initial provenance only; use the
current model's binding resource, operating point and exposed excess to
choose among variants allowed by `how_to_apply`.

Match the mechanism to the binding resource: fewer bytes/better layouts for
memory, amortized launches for launch overhead, efficient math for compute.
Do not infer end-to-end gain from a raw kernel-time share or unmatched ceiling.
Add one `Model alignment:` line in `optimization_summary.md` naming the
current model_id/component, predicted effect on the scored metric and the
mechanism tested. Retain actual contrary evidence for the Analyzer.

Stay within the roadmap item; leave new headroom for the Analyzer to plan.
Current measured evidence takes precedence over older predictions. If the
model is unavailable, say so and use the item's supported evidence without
inventing a ceiling or declaring convergence.
"""


SOL_OPTIMIZE_REPORTER_GUIDANCE = """\
## Use the current theoretical model

Read `sol_projection.md` as initial provenance and the latest
`performance_model.yaml` + `analysis.md` as the authoritative current model.
Use the current model for the report's single per-point comparison and gap
analysis. Do not add Projection vs Measured, SOL correlation or extra
remaining-gap sections. If the initial projection has been superseded,
mention the material correction once with its evidence and model_id.
Do not mix its old ceiling with a current kernel floor or practical rate.

Use final verification measurements when available, else the accepted
measurement (say which). Compare only compatible builds/workloads/timing
scopes. A newer unmodeled state needs an explicit limitation, not an inherited
convergence claim. If the current bound is unavailable, display unknown and
the required next test; do not fall back to a superseded initial bound.
Large model discrepancies do not establish host/scheduler overhead: require
phase measurements. Distinguish physical limits, campaign scope, measurement
limits and unresolved model error. A drained roadmap is not convergence.
"""


# --------------------------------------------------------------------------- #
# Per-kernel coverage contract (analyzer / reporter) — built per run, only
# when task.yaml declares profile.kernel_coverage
# --------------------------------------------------------------------------- #


def kernel_coverage_ncu_targeting(min_share_pct: float, coverage_target_pct: float) -> str:
    """Return Run B step 2 with the task's kernel coverage thresholds.

    Args:
        min_share_pct: Minimum in-window GPU share requiring a ledger row.
        coverage_target_pct: Minimum combined GPU share of enumerated rows.
    """
    return f"""\
2. **Select kernels by coverage and capture in bounded passes.** Enumerate
   all kernels at/above **{min_share_pct}%** of in-window GPU time, then add
   next-largest kernels to reach **{coverage_target_pct}%** combined share.
   Record the tail as `other`. Rank from `nsys_analysis/`:
   `cat_full.json`'s `per_category` and
   `matched_kernels`, with `opgroup.json` / `module_slice.json` for the
   residual. This decomposition clips the union of GPU activity to the
   iteration window. Whole-capture `cuda_gpu_kern_sum` is only a fallback
   when the pipeline cannot run; record it in the manifest. Record GPU
   busy vs idle and selected kernels' full names/shares. The Analyzer
   independently authors the ledger and dispositions.

   Run the canonical command below for up to **3 passes**, excluding
   collectives from ncu replay. Pass 1 targets the hottest 3–6 stems.
   Inspect each report with `ncu --import ... --page raw --csv`; later
   passes target **only still-missing stems**. Use `--launch-count` ≈
   8 × the pass's stem count (cap ~300), since per-layer kernels can
   exhaust a launch-order budget before once-per-step kernels appear.
   Each pass relaunches the server with the same iteration gate. Name
   artifacts `server_ncu_pass<k>.ncu-rep`, `ncu_details_pass<k>.txt`, and
   `ncu_raw_pass<k>.csv`. Keep uncaptured kernels in manifest coverage
   notes with `ncu: "unavailable: <reason>"`; the Analyzer uses nsys plus
   source to account for missing evidence.
"""


def kernel_coverage_analyzer_note(min_share_pct: float, coverage_target_pct: float) -> str:
    """Return the per-kernel ledger contract for the task's coverage bars.

    Args:
        min_share_pct: Minimum in-window GPU share requiring a ledger row.
        coverage_target_pct: Minimum combined GPU share of enumerated rows.
    """
    return f"""\
## Per-kernel coverage contract (this task declares `profile.kernel_coverage`)

Every optimization-round Analyzer, including re-analysis and replan turns,
writes `analysis/kernel_ledger.yaml` to support
`performance_model.yaml`. Final reconciliation reads the last ledger;
it updates only the aggregate model and analysis. Enumerate all kernels
at/above {min_share_pct}% and add rows to cover
{coverage_target_pct}% of GPU time in the selected capture's timeline.
A missing row, question or model, invalid roadmap reference, or insufficient
coverage aborts the stage. Re-analysis rebuilds from saved evidence,
independently of the profiler's ncu targets. Replans copy standing
measurements, coverage and provenance, updating dispositions and models
from new facts without claiming a new capture. Preserve prior
`model_revisions`. Imported ledgers are prior art: start a new version-2
ledger with local roadmap references and `model_revisions: []`, retaining
source operating conditions and measurement citations.

### Materiality and shared disposition rules

`share_pct` and `coverage.gpu_busy_pct` are percentages (0–100):

```
wall_clock_share_pct = share_pct x gpu_busy_pct / 100
time_saved_pct      = wall_clock_share_pct x recovery_fraction
latency_gain_pct    = time_saved_pct
throughput_gain_pct = 100 x time_saved_pct / (100 - time_saved_pct)
```

For fixed work, set `best_case_gain_pct` in the target metric:
latency uses `latency_gain_pct`, throughput uses
`throughput_gain_pct`. Require `0 <= time_saved_pct < 100`; recovering
all elapsed time does not justify a finite throughput estimate. This
assumes recovered wall-clock time affects the target metric proportionally;
state and bound that assumption for TTFT, TPOT, percentiles or capacity
changes. Do not equate unrelated times.
Use the target-metric estimate in `expected_gain_rationale` and compare
it with `optimize.noise_floor_pct` for every `below-materiality` dismissal.
For example, 8% GPU share at 60% busy with half recoverable saves 2.4%
elapsed time, implying about 2.46% throughput gain; 50% less elapsed
time implies 100% more throughput for the same work.
Choose the affected time and recovery bound for the question:

| Question | Maximum recoverable time |
|----------|--------------------------|
| Eliminate | Whole row; measured padded fraction for skipping wasted work |
| Faster | Row share x best-case recovery fraction |
| Fuse | Whole affected chain x best-case saving fraction |
| Overlap | `min(wall_clock_share_pct_A, wall_clock_share_pct_B)`, reduced by contention |

Record low GPU utilization as a separate host/launch finding and an item
when evidence, materiality, and allowed approaches support it.
All kernel work follows *Prefer existing kernels*. Shared dismissal tags:

- `below-materiality` — show the applicable wall-clock arithmetic above.
- `needs-rebuild: <artifact>` — cite why the artifact cannot be rebuilt
  **and** why a replacement kernel cannot help: no Python-reachable
  dispatch to reroute, or no credible headroom over the tuned incumbent
  near its bound-class ceiling. Otherwise plan the replacement, including
  a newly written fused kernel where appropriate.
- `approach-restricted` / `accuracy-scope` — cite the disallowed approach
  or forbidden lossy change as `scope_limited` in Gap analysis/Next actions.

Answer **all four questions for every row**, even when elimination is an
item. Prioritize elimination over that row's alternative implementations;
do not add their expected gains together. When fusion and overlap recover
the same time, identify them as alternatives in the
second item's `expected_gain_rationale` and count the saving once.

### Question 1 per kernel — can it be eliminated?

Check source and the NVTX timeline for:

- **Redundant:** duplicate computation, removable cast/copy, or an undone
  layout transform.
- **Wasted:** padded/masked tokens, inactive experts, or dummy graph slots;
  measure the fraction that cannot affect the output.
- **Hoistable:** invariant preprocessing, scales, indices, or tables that
  can move to load/warmup or a cache.
- **Accidental slow path:** a fast-path flag, backend selector,
  or shape/dtype guard whose inactive path creates the kernel. Enabling
  that path is elimination; a faster implementation of necessary work
  belongs to question 2.

Lead dismissal `ref` with the applicable tag and evidence:

- `mandatory-math: <what>` — necessary per-step work on real data, neither
  duplicated nor invariant; cite its consumer in `why_it_runs`.
- `padding-minimal: <n>%` — measured wasted fraction is below materiality.
- `already-hoisted` — cite where the invariant part runs at load/warmup.
- `fast-path-active` — cite the live selector/flag/config.
- `fast-path-blocked: <guard>` — quote the unsatisfiable hardware, shape,
  or dtype guard; a satisfiable guard instead supports an item.
- `below-materiality`, `approach-restricted`, or `accuracy-scope` as above.

### Question 2 per kernel — can it be made faster?

Classify using the `perf-nsight-compute-analysis` skill's thresholds and
bottleneck guide. For memory-bound work, inspect removable round
trips (question 3); for compute-bound work, inspect backend/kernel choices
and permitted precision changes. For latency-bound work, distinguish
inter-launch gaps (graph/launch amortization) from a kernel underfilling
the device inside a graph replay (question 4).

Dismissal tags:

- `at-sol-floor: <side> SOL <n>%` — binding-side utilization ~≥85%, with
  no material faster-execution headroom. Still evaluate fusion/overlap.
- `below-materiality`, `needs-rebuild`, `approach-restricted`, or
  `accuracy-scope` under the shared rules.

### Question 3 per kernel — can it be fused with its neighbors?

Derive predecessor/successor launches from `cuda_gpu_trace` or the
steady-state timeline, and producer/consumer tensors from NVTX plus
source. Record `fusion.neighbors`. Check elementwise/cast/activation
chains, norm + quantization, RoPE + KV-cache write, dequant + GEMM
prologue/epilogue, and attention-adjacent glue. Use the whole chain's
materiality bound.

Dismissal tags:

- `multi-consumer-pinned` — cite consumers preventing removal of the
  intermediate round trip.
- `already-fused` — no adjacent work remains to absorb.
- `phase-boundary` — name the capture, stream, or prefill/decode boundary
  this fusion cannot cross.
- `neighbors-at-bandwidth-floor` — show mandatory bytes on both sides;
  fusion removes no traffic.
- `below-materiality` or `needs-rebuild` under the shared rules.

### Question 4 per kernel — can it be overlapped with independent work?

Inspect low SOL on both sides/low occupancy, lopsided SOL, and small grids
or partial waves. A faster-execution dismissal does not settle overlap.
From the same timeline and source used for fusion, prove the partner's
data independence: neither reads what the other writes, with disjoint
outputs and step state. In `overlap.concurrent_with`, name the partner
and evidence that they are serialized today. Sum demand on the binding
resource and show it remains under ~100%; account for contention in the
pair's recovery bound.

Use `maybe_execute_in_parallel(fn0, fn1, event0, event1, aux_stream)` in
`tensorrt_llm/_torch/modules/multi_stream_utils.py`. Name the wrapped call
site and existing `AuxStreamType` slot (`tensorrt_llm/_torch/utils.py`,
allocated in model `__init__`) in `how_to_apply`; do not hand-roll a
parallel stream/event mechanism. The casebook's *Overlap, launch &
scheduling knobs* covers shared/routed-expert and MLA RoPE/uk-BGEMM
precedents for `casebook_ref`.

**Check the live CUDA-graph config first.** The runner activates
`with_multi_stream(True)` during capture; otherwise the helper executes
`fn0(); fn1()` sequentially. With graphs disabled, dismiss overlap as
`graph-disabled`. Plan graph enablement only if the task permits it;
otherwise record that prerequisite as `scope_limited` in Gap analysis/Next actions.

Dismissal tags:

- `graph-disabled` — cite the live config.
- `no-independent-partner` — cite the tensors creating dependencies.
- `resource-saturated` — show the pair's summed binding-resource SOL.
- `already-concurrent` — cite stream ids and the observed overlapped span.
- `below-materiality` — use the pair's bound in the materiality table.
- `phase-boundary` — name the capture, stream, or prefill/decode boundary
  preventing the pairing.

### Maintain theoretical models from evidence

Each kernel references one current `models` entry, scoped to a kernel,
shared logical region or iteration. A shared model appears once. Record
derivation, assumptions, hardware constraints, predicted/measured milliseconds
and evidence. Its `operating_point` identifies hardware, shapes/dtypes,
concurrency, rank, capture/build and timing aggregation for a matched comparison.

Every round, including replans, review new traces, counters, source facts
and experiment outcomes. Correct invalid assumptions, omitted necessary
work or wrong hardware constraints; retain valid models. Append each
changed field's old/new value, reason, round and evidence to `model_revisions`,
preserving earlier revisions. A retired model uses `changes.removed`, with
`from` equal to its complete previous mapping and `to: null`. Keep measured
values and provenance unchanged unless new measurements exist.

Apply the shared model's evidence and convergence rules to these derivations.
Unknown predictions/measurements are `null`, with `unexplained` and `next_test`.
For failed experiments, establish whether the intended mechanism executed;
failure alone neither relaxes a bound nor closes an opportunity. Investigate
measurements below predicted bounds without clamping or fitting them.

### Connect kernel models to the current performance model

Map repeated logical layers to `models` IDs, counts, shapes and precision;
label per-layer versus repeated-total costs. These derivations support the
shared `performance_model.yaml` contract. Put full layer/op tables and
calculator output in linked artifacts, following the four-section
`analysis.md` contract on full, replan and reused turns.

### The kernel ledger contract (`kernel_ledger.yaml`)

The two example rows total 27.6%. Add measured rows to reach the task's
coverage target, then recompute both coverage shares from the inventory.

```yaml
version: 2
source: rounds/round_<n>/analysis/nsys_analysis   # the decomposition you enumerated
coverage:
  enumerated_share_pct: 27.6    # sum of the two example kernels[].share_pct
  other_share_pct: 72.4         # unenumerated share; shrink by adding measured rows
  min_share_pct: {min_share_pct}
  gpu_busy_pct: 82.4            # GPU busy %; wall-clock share = share_pct x gpu_busy_pct / 100
kernels:                        # descending share_pct; one row per kernel/group
  - kernel: gdn_bf16_state              # distinctive stem or group label (unique)
    full_name: "void tensorrt_llm::..." # representative full name(s); group members
    share_pct: 18.4                     # % of in-window GPU time (nsys_analysis)
    model: state-update                # references models[].id
    ncu:                                # metrics mapping (or the string below)
      duration_us: 41.2
      sm_sol_pct: 12.1
      mem_sol_pct: 78.5
      occupancy_pct: null               # a metric the capture did not yield is null
      bound: memory                     # compute | memory | latency | balanced | comm
      note: "occupancy section empty: replay stalled"   # required by that null
    elimination:
      disposition: dismissed            # item | dismissed
      why_it_runs: "state update consumed by the next layer's gate (NVTX + source);
        selected by the fused path (is_fused=True, modeling_x.py:412)"
      ref: "mandatory-math: per-step recurrence, no padded or invariant part"
    faster:
      disposition: item                 # item | dismissed
      ref: opt-003                      # roadmap item id | evidence-backed dismissal
    fusion:
      disposition: dismissed
      neighbors: "rmsnorm -> THIS -> fp8_quant (cuda_gpu_trace, step 120)"
      ref: "multi-consumer-pinned: intermediate feeds residual add + next norm (cuda_gpu_trace)"
    overlap:
      disposition: item                 # item | dismissed
      concurrent_with: "moe_gemm: data-independent (disjoint outputs, per the NVTX
        ranges + source); serialized back-to-back on stream 7 today
        (cuda_gpu_trace, step 120)"
      ref: opt-004
  - kernel: allreduce_fusion            # collective: never replay under ncu
    full_name: "void tensorrt_llm::kernels::ar_fusion::..."
    share_pct: 9.2
    model: allreduce
    ncu: "unavailable: collective — kernel replay deadlocks the ranks"
    bound: comm                         # with the string form, `bound` sits here
    elimination:
      disposition: dismissed
      why_it_runs: "TP-sharded partials summed for the next layer's norm (source)"
      ref: "mandatory-math: the parallelism, not the kernel, requires the sum"
    faster:
      disposition: dismissed
      ref: "approach-restricted: strategy A/B falsified in a prior round; no NVLS here"
    fusion:
      disposition: dismissed
      neighbors: "sigmoid_gate_mul_add -> THIS -> scaleMatrixPerTensorVec (step 120)"
      ref: "already-fused: this IS the AR + residual/norm/quant fused epilogue"
    overlap:
      disposition: dismissed
      concurrent_with: "nothing independent in reach: every rank blocks here before
        the next layer (cuda_gpu_trace, step 120)"
      ref: "no-independent-partner: the collective is the layer's barrier"
models:
  - id: state-update
    scope: kernel                      # kernel | region | iteration
    operating_point:
      hardware: H100-SXM
      concurrency: 32
      dtype: bf16
      shape: "state [32, 64, 128]"
      rank: 0
      capture: round-2-capture
      build: "profile_manifest.json runtime.build"
      timing: "mean per invocation in steady-state steps 100-150"
    derivation: "mandatory 8 MB / measured sustainable 2 TB/s = 0.004 ms"
    assumptions: ["Each state element is read and written once; no reuse across steps"]
    evidence: ["modeling_x.py:412 mandatory traffic; calibration/bandwidth.md"]
    predicted_ms: 0.004
    measured_ms: 0.0412
    measurement_evidence: ["server_ncu.csv state-update duration, profile_manifest.json"]
    unexplained: "0.0372 ms above mandatory traffic bound; replay counters suggest extra traffic"
    next_test: "Count actual bytes and cache misses for the state-update launch"
  - id: allreduce
    scope: region
    operating_point:
      capture: round-2-capture
      hardware: "8 H100-SXM, NVLink"
      concurrency: 32
      dtype: bf16
      shape: "TP partials [32, 8192]"
      rank: all
      build: "profile_manifest.json runtime.build"
      timing: "critical-path collective per iteration, steps 100-150"
    derivation: "Need actual collective algorithm and channel bandwidth before bounding latency"
    assumptions: ["TP reduction is necessary; fused epilogue shares this region"]
    evidence: ["source collective call and cuda_gpu_trace step 120"]
    predicted_ms: null
    measured_ms: null
    measurement_evidence: []
    unexplained: "Collective timing and channel calibration unavailable at this operating point"
    next_test: "Request rank-aligned collective timing and channel calibration"
model_revisions: []                    # preserved and appended each round
# Revision entry shape (actual changes only, with artifact citations):
# - round: 3
#   model: state-update
#   reason: "Source confirms a second mandatory state read omitted by the model"
#   evidence: ["modeling_x.py:416 and trace/traffic.csv"]
#   changes:
#     predicted_ms:
#       from: 0.004
#       to: 0.006
#     derivation:
#       from: "mandatory 8 MB / measured sustainable 2 TB/s = 0.004 ms"
#       to: "mandatory 12 MB / measured sustainable 2 TB/s = 0.006 ms"
```

- `disposition: item` references an existing or newly authored roadmap id;
  accepted/failed items are valid historical references. Multiple rows
  may share an item: cite fusion items from every affected kernel and
  overlap items from both partners. `dismissed` refs use the tags above
  and cite an ncu row, timeline, source file, or failed evaluation.
- Collectives never go under ncu replay because it deadlocks ranks. Use
  `ncu: "unavailable: <reason>"` for them or other uncaptured kernels,
  with `bound` on the row (`comm` for collectives). For captured kernels,
  `bound` lives inside the `ncu` mapping; an absent metric is `null` with
  an explanatory `note`. Keep measured metrics and never invent others.
  `bound` is required in either shape.
- A fusion/overlap dismissal requires `neighbors`/`concurrent_with`;
  an item may carry that evidence in its referenced roadmap entry.
- Items below `optimize.noise_floor_pct` are not actionable; use
  `below-materiality` rather than a roadmap reference to such an item.
- **Subsequent full analyses:** author a fresh ledger. Carry a
  dismissal only if share changed by no more than ~20% relative, bound
  class is unchanged, and no accepted item touched the kernel. Cite its
  original evidence plus `carried from round <k>`. Re-derive changed
  rows, include newly qualifying kernels, and re-derive fusion/overlap
  when the neighbor/partner changed. Recompute materiality whenever
  `gpu_busy_pct` changes. Replan-only rounds keep standing measurements
  in their new ledger and refresh model/disposition reasoning from new evidence.
- Keep the complete kernel disposition table in `kernel_ledger.yaml`.
  `analysis.md` links to it and selects only evidence explaining material
  gaps in `performance_model.yaml`; do not duplicate every row in Markdown.
"""


KERNEL_COVERAGE_REPORTER_GUIDANCE = """\
## Supporting kernel evidence

Read the final round's `kernel_ledger.yaml` and link to it from Gap analysis.
It supports `performance_model.yaml`; it does not supply a second headline
ceiling or a separate Kernel Coverage section. Disclose coverage
and how many rows ncu actually measured when that limits a conclusion.
Unavailable counters, nulls and contaminated samples remain visible.

Explain the largest residuals using model-relevant kernel findings,
with ledger/model IDs and evaluation evidence. Resolve item outcomes from
the roadmap, without treating rejected or dismissed items as physical limits.
Keep detailed four-question dispositions and per-kernel counters in the YAML.
Do not reproduce all kernels, add a theoretical headroom summary, or repeat
next actions in a separate remaining-roadmap/durable-facts section.
"""


# --------------------------------------------------------------------------- #
# Approach restriction (analyzer / optimizer / evaluator) — built per run
# --------------------------------------------------------------------------- #

_APPROACH_GUARDS = {
    "config": """\
  - `tuning/extra_llm_api_options.yaml` is **read-only for every role**
    this run. After every optimizer attempt, the orchestrator compares it
    with the accepted snapshot and **auto-rejects the attempt without any
    evaluation** if it changed. Implementing a config knob through source
    edits to a default or env-var fallback is also prohibited.\
""",
    "code": """\
  - The TRT-LLM checkout is **read-only for every role** this run. After
    every optimizer attempt, the orchestrator checks `git status --porcelain`
    and **auto-rejects the attempt without any evaluation** if the worktree
    is dirty.\
""",
}


def approach_restriction_note(allowed: Sequence[str], *, analyzer_only: bool = False) -> str:
    """The prompt block for a run restricted to a subset of ``APPROACHES``.

    Returns ``""`` when every approach is allowed (nothing to say). The
    block is appended to the analyzer / optimizer / evaluator prompts —
    ``analyzer_only`` emits only planning constraints; otherwise
    each role gets its consequence of ``optimize.approaches`` spelled
    out, mirroring the deterministic enforcement in the orchestrator
    (item-selection filter + post-optimizer auto-reject).
    """
    allowed = tuple(a for a in APPROACHES if a in allowed)
    disallowed = tuple(a for a in APPROACHES if a not in allowed)
    if not disallowed or not allowed:
        return ""
    allowed_str = ", ".join(f"`{a}`" for a in allowed)
    disallowed_str = ", ".join(f"`{a}`" for a in disallowed)
    if analyzer_only:
        return f"""\
## Approach restriction (`optimize.approaches`)

Allowed roadmap approaches: {allowed_str}. {disallowed_str} is off-limits.
Record disallowed opportunities with evidence as `scope_limited` in the
current model and `analysis.md`'s Gap analysis/Next actions. Do not disguise
a config change as code by editing a default or env-var fallback. If no
allowed approach can affect the active runtime, report the blocker and
leave no unactionable item.
"""
    guards = "\n".join(_APPROACH_GUARDS[a] for a in disallowed)
    return f"""\
## Approach restriction (`optimize.approaches`)

`task.yaml` sets `optimize.approaches: [{", ".join(allowed)}]`.
Only {allowed_str} roadmap items may be planned, applied, or accepted;
{disallowed_str} is off-limits.

- **Analyzer** — use only allowed `approach` values in `roadmap.yaml`.
  Record disallowed opportunities as `scope_limited` in
  `performance_model.yaml`, with evidence in `analysis.md`'s Gap analysis/Next actions,
  without planning unactionable work.
- **Optimizer** — implement only allowed approaches; do not bypass the restriction:
{guards}
- **Evaluator** — disallowed approaches fail code quality regardless of
  measured gain. PUSH_BACK with `reason_category: code_quality` toward an
  allowed approach, or REJECT if no allowed approach can implement the item.
"""
