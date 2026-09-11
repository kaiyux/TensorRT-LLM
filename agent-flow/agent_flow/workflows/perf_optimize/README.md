# perf-optimize

Iteratively **applies** TensorRT-LLM serving optimizations — the acting
counterpart to `perf-analyze` (which only diagnoses). Measures a
`trtllm-serve` baseline, plans evidence-grounded optimizations into a
machine-readable `roadmap.yaml`, evaluates a round's selected items in
serial or parallel worktrees, integrates parallel candidate-ready results, gates every
change on code quality / functionality / measured gain, runs the configured number of rounds,
independently verifies the final state, and reports measured performance
against the best current theoretical performance model.

`--reuse-analysis <dir>` imports a previous perf-analyze / perf-optimize
run's baseline, SOL projection and findings, starting the campaign at the
optimize stage (round 1's analyzer then plans from imported findings).
Add `--reanalyze` to reinterpret that run's saved captures before planning,
without profiling again in round 1.

```
benchmarker → (projector) → [round loop × max_rounds]
                                     ↓
                         qa → final_analyzer → reporter
                     (verification) (reconciliation)

round loop:
  (profiler) → analyzer → optimizer ⇄ evaluator item pairs
                         (serial or parallel worktrees;
                          ≤ max_attempts_per_item)
                                      ↓ (parallel only)
                                  integrator
                          (combine, benchmark, verdict)
```

- **benchmarker** — measures every configured operating point with the
  canonical `benchmark_serving.py` command and writes
  `baseline/benchmark_results.md`. These measurements anchor
  `roadmap.yaml`'s `baseline` and its per-point curve.
- **projector** — runs once between the baseline and round 1 unless
  `sol.enabled: false`. It derives an initial SOL estimate from model
  architecture, hardware peaks and measured latency constants where
  available, writing `sol_projection.md` and `sol_work/peaks.json`.
  This initial projection is provenance for the evolving model.
- **profiler** — owns server launch and cleanup, nsys/ncu captures,
  exports, targeting, coverage and runtime provenance. It writes
  `rounds/round_<n>/profile/profiler_report.md` and
  `profile_manifest.json`. The report summarizes capture quality and
  missing evidence; the manifest retains capture identity and the artifact
  inventory. A successful capture is checkpointed before analysis, so an
  analyzer retry does not repeat GPU work.
- **analyzer** — interprets saved evidence offline, writes
  `rounds/round_<n>/analysis/analysis.md`, updates that directory's
  `performance_model.yaml`, and ranks `roadmap.yaml` items by expected
  measured benefit. Every turn updates the model, including reuse,
  re-analysis and replan-only turns. It links kernel and region evidence
  to the end-to-end model, accounts for nsys opportunities, and maintains
  `kernel_ledger.yaml` when kernel coverage is enabled. It launches no
  server or GPU profiler. A completed full analysis has an
  `analysis_manifest.yaml` linking it to its source capture.
- **optimizer** — implements a selected item in an isolated worktree,
  using the current model to choose among allowed implementations.
  `approach: config` edits the live tuning YAML; `approach: code` edits
  the TRT-LLM source after verifying the installed package. It
  smoke-checks its change and leaves benchmarking and commits to the
  evaluator and orchestrator. Each item's retries are sequential.
- **evaluator** — checks code quality and functionality, measures the
  candidate, and emits `APPROVE`, `PUSH_BACK` or `REJECT`. Acceptance
  uses measured-versus-measured gains, never a theoretical prediction.
  Failure evidence distinguishes an invalid mechanism, no measured gain,
  a scope constraint and insufficient measurement sensitivity. Approved
  candidates receive a diagnostic nsys replay when configured; replay
  throughput never enters the acceptance gate.
- **integrator** — in parallel mode, combines candidate-ready changes in
  roadmap order, benchmarks the combined state and emits
  `APPROVE`, `FALLBACK_BEST` or `REJECT`. The orchestrator validates the
  selected ids, gain arithmetic and regression limits before promotion.
- **qa** — independently benchmarks the final accepted state once,
  checks completions and runs a configured accuracy evaluation. It is
  skipped when no item was accepted.
- **final_analyzer** — after QA, the same offline analyzer reconciles
  the final accepted runtime and independent measurements with current
  theoretical bounds. It writes `final_verification/analysis/analysis.md`
  and `performance_model.yaml` without editing the roadmap or using GPUs.
  This is a separately checkpointed stage, skipped when no change was
  accepted; retrying it does not repeat QA. Both output files must be
  nonempty and the model must pass validation before the stage advances
  to the reporter. Missing final-state evidence remains explicit in the model.
- **reporter** — writes `optimization_report.md` and a self-contained
  `optimization_report.html` with the same content. It prefers the
  reconciled final model over round models and states remaining headroom,
  its explanation, and unresolved evidence.

In serial mode each item starts from the latest accepted state. Parallel
items start from one frozen round base and are combined by the integrator.
On a shared local runtime, complete build/GPU/server sessions use
`flock` on `<workspace>/.local_runtime.lock`, including cleanup; coding
and offline analysis can overlap. This is a cooperative agent protocol.
The orchestrator owns roadmap lifecycle fields and checkout git state.

## The current theoretical performance model

`performance_model.yaml` is the central analysis and convergence artifact,
required independently of `sol.enabled` and `profile.kernel_coverage`.
Each round writes it in `rounds/round_<n>/analysis/`; final reconciliation
writes it in `final_verification/analysis/`. The latest validated model
represents the current analysis, and the reconciled final model takes
precedence for reporting. The analyzer updates it from the best available facts;
`sol_projection.md` preserves the initial estimate and its assumptions.
A failed optimization cannot by itself justify raising the theoretical
floor or declaring the remaining gap unavoidable.

For every benchmark concurrency, the current model connects measured
performance to theoretical best performance on the same workload and
timing basis. It identifies the runtime, model, hardware, parallelism,
request counts, token lengths, timing window and statistic. Kernel-sum
milliseconds, decode critical-path milliseconds and whole-run serving
latency are separate observations until an explicit derivation connects
them. A profile at one concurrency does not measure another concurrency.

The model must make the following accounting reviewable:

- Current measured performance, the current theoretical ceiling, the
  distance between them and the assumptions behind the conversion.
- Unavoidable physical costs, backed by a bound and evidence.
- Recoverable implementation costs, tied to experiments or roadmap items.
- Campaign scope constraints and measurement limitations, stated
  separately from physical limits. A noise-floor rejection is not a
  hardware impossibility.
- The unresolved residual, including unmeasured phases and invalid or
  incomplete comparisons, with the next measurement needed to resolve it.

Region and kernel models support this accounting; detailed derivations
remain linked artifacts. Account for overlap and dependencies before
adding region savings. Do not subtract a short decode-window median from
a whole-run mean and label the difference removable prefill overhead.
Unavailable quantities stay explicit rather than acquiring guessed values.
Evidence-backed model revisions retain the prior assumption and explain
what changed. Imported evidence keeps its source measurement provenance.

Convergence requires a consistent model and evidence explaining the
remaining residual at every focus point (all configured points when no
focus subset is set). Every benchmark point remains recorded, with
unknown theory and a next measurement when no bound can be grounded.
Model status is `open`, `converged`, `measurement_limited`, `scope_limited`
or `model_invalid`; see the
[shared model contract](../perf_analyze/performance_model.py).
An empty roadmap, an exhausted budget, a met improvement target, or an unmeasured residual alone cannot
establish convergence. Report the workflow stop reason separately.

## Report format

The analyzer's `analysis.md`, including final reconciliation, has four
concise sections:

1. **Result** — measured outcome, source/runtime identity and convergence
   status; final reconciliation uses the independent QA measurements.
2. **Theoretical performance model** — one per-point table comparing
   current measurements with the current theoretical best, plus the
   essential assumptions and model revisions.
3. **Gap analysis** — the largest remaining costs, evidence for each
   explanation, unresolved residual and missing measurements.
4. **Next actions** — the few highest-value optimizations or measurements
   needed to close the remaining gap.

The final `optimization_report.md` and HTML companion use the same first
three sections, then **Changes and next actions** for consequential
accepted changes, failed experiments and follow-up work. Their **Result**
includes the verified baseline-to-final gain.

Keep commands, full configurations, per-kernel tables, attempt histories
and detailed arithmetic in linked artifacts. Do not repeat the same gap
in separate SOL, kernel-coverage, projection and future-work sections.
`profiler_report.md` is the profiler's brief capture handoff: what was
captured, whether it matches the requested runtime/window, coverage gaps,
and links to evidence. It accompanies the machine-readable manifest.

## The nsys opportunity-coverage gate

The `internal-perf-nsight-system-analysis` skill closes every run by writing
`nsys_analysis/items.json` — the performance opportunities the timeline
found, each with a stable `id`, the step table behind it and a
`magnitudeMs`. Without a consumer that list is prose: the analyzer reads
the numbers, writes findings, and whether an opportunity ever reached
the plan is invisible.

So `roadmap.yaml` carries a top-level `nsys_items` block accounting for
every id in that file — the same `disposition` / `ref` vocabulary as
`kernel_ledger.yaml`, for the same reason:

```yaml
nsys_items:
  - {id: nsys-01, disposition: item, ref: opt-003}
  - {id: nsys-02, disposition: dismissed, ref: "0.2 ms/iter is below the noise floor"}
```

An `item` ref must name a real roadmap id (any status — an opportunity
whose fix was already tried *was* considered); a `dismissed` ref is the
evidence for dismissing it. The orchestrator validates the block the
moment the analyzer's turn ends, so an opportunity that was neither
planned nor dismissed parks the campaign at the analyzer instead of
quietly evaporating. Dismissing is a first-class answer — below the
noise floor, mechanism already present, no allowed approach reaches it —
and is always cheaper than an unfounded item a full benchmark has to
disprove.

**Self-gating on the artifact**, so it needs no task knob: enforced
exactly when the round produced an `items.json`. A replan-only round
runs no profiler and writes none; a round whose skill was unavailable or
whose pipeline errored writes none either and records the reason under
*Caveats*. Neither owes the block anything.

## The acceptance gate

An attempt is **APPROVEd** only when all three hold:

1. code quality OK (scoped, clean diff), and
2. functionality OK (server serves coherent completions; targeted tests
   pass for code items), and
3. `measured_gain_pct ≥ accept_fraction × expected_gain_pct` **and**
   `measured_gain_pct ≥ noise_floor_pct`, measured on
   `optimize.target_metric` against the **last accepted** measurement
   (`current_best` in `roadmap.yaml`), so gains accumulate.

The orchestrator checks every APPROVE before committing a candidate,
in either execution mode. It recomputes the gain against the item's
frozen reference measurement, checks the reported arithmetic, requires
finite positive measurements and complete curves, and enforces the
threshold and regression budget. Integration uses the same measurement
validation with its combined-candidate threshold. Frozen reference
measurements also make promotion retries independent of later ledger updates.

Fresh and imported baselines must contain successful benchmark results
with a finite positive target metric at every configured concurrency
point. An unusable imported baseline is measured again.

When any axis fails, the evaluator chooses between two negative
verdicts: **PUSH_BACK** (a concrete, actionable fix exists — the
optimizer retries with the feedback, bounded by
`max_attempts_per_item`; on the final attempt PUSH_BACK coerces to
REJECT) or **REJECT** (the item's premise is broken — no retry would
help; the item is failed immediately and the loop moves on). Either way
the orchestrator reverts every change.

**Pareto-curve mode** (`benchmark.concurrency` is a list): every
measurement runs once per concurrency point (over one server launch) and
gains are computed per point against the same-concurrency
`current_best.curve` entry. Rule 3 becomes the **Pareto gate**: the
**mean** per-point gain must pass both thresholds **and** no individual
point may regress by more than `noise_floor_pct` — a trade that helps
one regime by hurting another is rejected. The ledger
(`baseline`/`current_best`) then carries a `curve` of per-point
`{concurrency, value, tok_s_user, tok_s_gpu}` rows (tok/s/user =
`1000/mean_tpot_ms`, tok/s/gpu = `output_throughput/num_gpus`), and the
report includes the per-point comparison in **Result**; a compact
Pareto chart (x = tok/s/user, y = tok/s/gpu) can clarify it. Note the benchmark cost multiplies by
the point count on **every** measurement (baseline, each evaluator
attempt, the final verification) — keep the list short (~3–5 points).

## When the loop stops

The orchestrator bounds optimization work by `optimize.max_rounds` and
selects up to `max_items_per_round` candidates each round. The loop can
stop at the round budget, at the optional measured improvement target,
or after an analyzer turn leaves the roadmap exhausted on an unchanged
build with no outstanding measurement request.

A `measurement_limited` model requesting a targeted capture keeps the
loop moving while rounds remain, even when it has no pending optimization
item. Measurement requests also persist when pending items are attempted
and rejected; those verdicts do not cancel the need for evidence. The
next profiler turn receives the requested operating point, phase or
kernel instead of simply repeating the standing capture. A roadmap that
runs dry after accepted changes still needs analysis of that runtime,
subject to the remaining round budget.

When changes were accepted, the campaign proceeds through independent QA,
the checkpointed `final_analyzer` reconciliation, and the reporter.
Without accepted changes, it skips QA and final reconciliation and
reports from the latest round model. Workflow stopping is separate from
convergence: unfulfilled measurement requests remain visible when a
budget or target ends the loop.

## What a round costs: profile, re-analyze, or replan

Every round reaches an analyzer turn; the orchestrator invokes the
profiler first only when fresh captures are needed. Rejected attempts run
in isolated worktrees and restore their item configuration, leaving the
campaign source and accepted configuration unchanged. Their worktrees are
removed after their outcome is durably recorded.

A round selects up to `optimize.max_items_per_round` pending roadmap
items. Their optimizer/evaluator loops run serially or concurrently
according to `optimize.item_execution`; parallel candidates are combined
and measured by the Integrator, while serial candidates are accepted
directly.

- **Profile and analyze** — round 1 unless evidence is imported; any round
  opening after an accept; or a round fulfilling an outstanding targeted
  measurement request. The profiler captures the current runtime (nsys + ncu per
  `profile.methods`), then the analyzer interprets those captures and
  re-ranks the roadmap. An older checkpoint with no profile-currency
  marker also buys one
  conservative profile on resume.
- **Re-analyze saved captures** — round 1 of a fresh campaign started with
  `--reuse-analysis <dir> --reanalyze`. The profiler is skipped and the
  analyzer regenerates findings, derivations, ledgers, and the roadmap
  from imported raw evidence. This supports changes to analysis
  methodology or taxonomy without another GPU capture. Later rounds
  follow the normal profiling rules; the option does not make the
  entire optimization campaign offline.
- **Replan-only round** — opens when the standing profile is known to be
  current and there is no outstanding request for new evidence. The
  predecessor accepted nothing. The analyzer launches no
  server and runs no profiler; it plans from the standing analysis plus
  the round's evaluator verdicts, marking
  disproven items obsolete, bounding the gains the measurements cap, and
  adding what the failures imply. Those verdicts are the round's real
  yield — an item measured dead is evidence about *this* build — and
  converting them into roadmap edits and model updates is what the turn
  is for. The analyzer updates `performance_model.yaml`, writes this round's
  `analysis.md`, and refreshes the optional kernel ledger using standing
  measurements and the latest evidence. Prior revisions and measurement
  provenance remain intact; no new capture is needed.

The orchestrator selects the mode. A replan turn updates the model and
can either produce an optimization item or request targeted evidence.
Only an exhausted roadmap with no outstanding measurement request can
trigger the unchanged-build stop.

Capture completion and analysis completion have separate checkpoints.
If the analyzer fails after a successful capture, rerunning the command
resumes analysis from the preserved `profile/` directory. An unfinished
capture remains the profiler's responsibility. Re-analysis imported from
another workspace is not proof that the local runtime is current, so it
does not earn the local profile-currency marker used by replan-only rounds.

That break only fires at the top of a round, never mid-round. A roadmap
that runs dry between items ran dry against a plan written *before* the
round's measurements existed, so the loop spends one more round — free
when no capture is needed, a profile after accepted changes or a targeted
request — subject to the round budget. It then evaluates the updated
plan and model against the new evidence.

## Reusing a previous run's analysis

`--reuse-analysis <dir>` seeds a fresh workspace from a previous
`perf-analyze` run or `perf-optimize` campaign, so the campaign starts at
the optimize stage instead of re-deriving what that run already measured:

| imported | from a perf-analyze workspace | from a perf-optimize workspace | replaces |
| --- | --- | --- | --- |
| baseline report + result JSONs | `benchmark_results.md` | `baseline/benchmark_results.md` | the benchmarker |
| Initial SOL projection + `sol_work/` | `sol_projection.md` | `sol_projection.md` | the projector |
| analysis (+ optional `kernel_ledger.yaml`) | `analysis.md` and companion artifacts | newest `rounds/round_<n>/analysis/` | round 1's analysis, unless `--reanalyze` |
| Current performance model | `performance_model.yaml` | newest valid `analysis/performance_model.yaml`, preferring final reconciliation | read-only `reused_analysis/prior_performance_model.yaml` |
| raw profile captures | workspace trace files | `rounds/round_<n>/profile/` with `profile_manifest.json` (legacy `analysis/` layouts also supported) | round 1's profiler |
| roadmap (as read-only prior art) | — | `roadmap.yaml` | nothing |

Legacy `profile_findings.md` sources remain readable; new analyzer output
is always `analysis.md`.

Round 1's analyzer then runs **plan-only**: it reads the imported
evidence, checks that it actually describes this task (same model,
parallel mapping, operating point), runs the dormant-capability sweep,
and writes `analysis.md`, the current `performance_model.yaml`, and
`roadmap.yaml` without a new server, profiler or benchmark. Round 2
profiles normally: the imported traces describe another run's build, so they never stand in for one this campaign
made, and the replan rule only ever plans from a profile of this
campaign's own checkout.

For a new interpretation instead of a plan based on existing findings,
add `--reanalyze`:

```bash
perf-optimize --task task.yaml --workspace workspace/reanalysis \
    --reuse-analysis workspace/previous-run --reanalyze
```

This imports raw captures into round 1's `profile/` directory and runs
the analyzer offline, writing fresh outputs in round 1's `analysis/`
directory. The capture manifest records the source and available
artifacts; discovery can find a completed capture even when its original
analyzer never finished. Legacy perf-analyze and perf-optimize captures
are supported too. When importing findings, `analysis_manifest.yaml`
links them to their own source capture; discovery does not substitute
a newer round's unrelated capture. Re-analysis needs usable raw evidence
or offline exports: a findings-only
source can support plain `--reuse-analysis`, but cannot supply a new trace
analysis. The analyzer checks coverage and records missing evidence
without silently launching another capture.

Two deliberate limits:

- **Roadmap state is never imported.** A source `roadmap.yaml` lands in
  `reused_analysis/prior_roadmap.yaml` as reference material only; its
  statuses, gains and current best describe the source checkout. Imported
  models are saved as `reused_analysis/prior_performance_model.yaml`.
  This preserves source corrections and measurement provenance; it never
  stands in for this campaign's current model. Together with
  `kernel_ledger.yaml`, `sol.json` and `regions.json`, it provides evidence
  for the analyzer, which writes this campaign's current model
  and roadmap with local item references, source measurement conditions
  and citations. Local revision history starts anew with imported
  revisions retained as provenance. With `--reanalyze`, it regenerates
  derived artifacts from the saved captures.
- **The baseline is inherited, not re-measured.** Every gain this
  campaign reports is computed against numbers measured by the source
  run, so the two must describe the same system. The import writes
  `reused_analysis/manifest.md` recording what came from where, the
  analyzer owes a fit check against it, and the report says so in
  **Result**. If the hardware or checkpoint differs, don't reuse the
  baseline — the import is per-artifact, so a source without one simply
  gets benchmarked normally.

These options seed fresh runs. `--reanalyze` requires `--reuse-analysis`
and a fresh workspace (or `--clean`); it is rejected against an existing
checkpoint. To resume a campaign already started in re-analysis mode,
rerun without `--reanalyze`: the checkpoint preserves the choice. Plain
`--reuse-analysis` on resume is ignored with a warning.

Resumed roles are composed from the saved `workspace/task.yaml`, matching
the orchestrator's task rather than a changed command-line input file.
Split-layout analysis imports require a matching completion manifest;
an interrupted analyzer's partial findings cannot replace an earlier
completed analysis. Capture-only imports remain available with `--reanalyze`.

Every analyzer turn, including plan-only reuse and replan-only rounds,
updates and validates `performance_model.yaml`. With
`profile.kernel_coverage`, this round's kernel ledger is validated too.
Both preserve measurement provenance while incorporating new evidence.

## Usage

```bash
# from the repo root
pip install -e .

perf-optimize --task path/to/task.yaml --workspace workspace/perf-optimize/my-model

# resume after a crash / Ctrl-C: just re-run the same command.
# start over from scratch:
perf-optimize --task path/to/task.yaml --workspace workspace/perf-optimize/my-model --clean
# override the round budget on a fresh run:
perf-optimize --task ... --workspace ... --max-rounds 5
# start at the optimize stage, reusing a previous run's analysis:
perf-optimize --task ... --workspace ... --reuse-analysis workspace/perf-analyze/my-model
# reinterpret saved captures before round 1's plan, in a fresh workspace:
perf-optimize --task ... --workspace ... --reuse-analysis workspace/perf-analyze/my-model --reanalyze
```

## task.yaml

See `task.example.yaml`. Fields beyond the perf-analyze base spec
(`checkpoint_path`, `trtllm_repo_path`, optional `extra_llm_api_options`
/ `benchmark` / `profile` / `slurm-environment` / `sol` — see the
perf-analyze README for the base fields, including
`benchmark.concurrency`, an int or
a list of ints, and `benchmark.num_prompts`, an int or — curve mode
only — a list paired index-by-index with the concurrency list so
low-concurrency points can run fewer prompts; the pre-rename
`max_concurrency` spelling fails validation with a pointer to the new
name. The optional `sol` block — every field optional, `enabled` the
stage gate (default `true`) and `gpu` the part-name hint for the SOL
skill's peaks calculator — gates the one-shot SOL projector stage here
exactly as in perf-analyze):

| field | required | default | meaning |
| --- | --- | --- | --- |
| `optimize.max_rounds` | no | `5` | The number of rounds the loop **runs** (not just a cap — only the two deterministic breaks above end it earlier); each round is an optional profiler turn, one analyzer turn, and up to `max_items_per_round` items, so `max_rounds × max_items_per_round` bounds total items attempted. Only rounds with a stale or unproven runtime profile pay to refresh it (see *What a round costs*), so this bounds items far more tightly than GPU hours. |
| `optimize.max_items_per_round` | no | `3` | Maximum optimizer/evaluator pairs selected per round. Every pair owns an isolated worktree, tuning copy, progress file, and bounded attempt loop. |
| `optimize.item_execution` | no | `parallel` | `parallel` evaluates selected pairs from one frozen round base and runs the Integrator; pairs overlap only with isolated Slurm allocations and run one at a time on a shared local runtime. `serial` starts each pair from the latest accepted state and promotes it directly, without an Integrator. |
| `optimize.max_attempts_per_item` | no | `3` | Total optimizer attempts per item: PUSH_BACK verdicts retry until this bound, then the item is marked `failed` and reverted (an explicit REJECT fails it immediately). |
| `optimize.approaches` | no | `[config, code]` | Which optimization approaches the run may plan/apply: `config` edits the live tuning YAML, `code` edits the TRT-LLM source. Restrict to `[code]` for a code-only campaign (no knob tuning) or `[config]` to leave the checkout untouched. Enforced in three layers: the analyzer only plans allowed items, the orchestrator never dispatches a disallowed pending item, and any attempt that edits through a disallowed approach (tuning file differs from the accepted snapshot / dirty worktree) is auto-rejected before the evaluator benchmarks it. |
| `optimize.accept_fraction` | no | `0.5` | Fraction of an item's `expected_gain_pct` the measured gain must reach. |
| `optimize.noise_floor_pct` | no | `1.0` | Minimum measured gain (%); also the actionability floor for pending items. |
| `optimize.target_metric` | no | `output_throughput` | Result-JSON key gains are computed on (see below). |
| `optimize.target_improvement_pct` | no | — | Optional early-stop: the orchestrator concludes the loop once the roadmap ledger's cumulative improvement reaches this. |
| `accuracy.command` | with `accuracy` | — | Accuracy eval command the final verification runs verbatim against the live server (e.g. `trtllm-eval ...`). Omit the whole block to skip accuracy checks. |
| `accuracy.baseline_score` | no | — | Reference score to compare against. |
| `accuracy.max_drop_pct` | no | `1.0` | Allowed relative score drop vs `baseline_score`. |

**Target metric keys** (from the `benchmark_serving.py` result JSON):
`output_throughput` (tok/s, default), `total_token_throughput`,
`request_throughput`, and latency keys `mean_ttft_ms` / `median_ttft_ms`
/ `p99_ttft_ms` (likewise `*_tpot_ms`, `*_itl_ms`, `*_e2el_ms`). Gains
are always normalized so positive = improvement (throughput up, latency
down).

When `slurm-environment.cluster_ssh` is set, paths belong to these machines:

| Field | Location |
| --- | --- |
| `trtllm_repo_path` | Local checkout edited and managed by perf-optimize. |
| `extra_llm_api_options` | Local input copied into the workflow's live tuning YAML. |
| `checkpoint_path` | Remote path visible to the Slurm job/container. |
| `slurm-environment.cluster_ssh` | SSH target; setting it enables remote execution. |
| `slurm-environment.docker_image` | Remote SQSH path visible to Slurm/Pyxis. |
| `slurm-environment.remote_run_root` | Remote absolute temporary root; defaults to `~/agent_flow_workspace/<workspace-name>`. |
| `slurm-environment.slurm_partition`, `account`, `qos` | Settings of the selected remote Slurm cluster. |

The workflow workspace and Git stay local. Agents copy required inputs and
changed source to isolated directories below the remote run root, then pull
required outputs back. Without `cluster_ssh`, all paths refer to the machine
running the CLI.

## Git requirements (read before running)

`perf-optimize` **mutates the TRT-LLM checkout** at `trtllm_repo_path`:

- The checkout must be a clean git repository on startup and resume.
  Commit or stash uncommitted changes first; the workflow refuses to
  compare a dirty campaign checkout with candidates created from HEAD.
  Rejected attempts are reverted in their isolated worktrees with
  `git reset --hard` + `git clean -fd`.
- Work happens on a dedicated branch `perf-optimize/<workspace>-<ts>`
  created from the current HEAD. Candidate code is committed before
  promotion; parallel integration combines those commits. Pushed-back
  and rejected attempts are reverted before the
  next attempt/item starts. `--clean` never touches the checkout —
  abandoned branches are left for inspection.
- Every serving role, including baseline measurement and final QA,
  binds the active checkout (or its remote staged copy) to `PYTHONPATH`
  and verifies the resolved package path inside the execution container.
  A mismatched installed package is a blocker, not a valid measurement.
- `optimize.approaches` only restricts what the *loop* may change; the
  task's `extra_llm_api_options` seed still applies as the baseline
  config in every mode.

## Workspace layout

```
<workspace>/
├── task.yaml                        # resolved spec (defaults filled in)
├── prompts/<role>.md                # composed system prompt per role, snapshotted at launch
├── roadmap.yaml                     # the ranked plan; statuses/gains updated as the loop runs
├── sol_projection.md                # initial projection and assumptions (blank when disabled)
├── sol_work/peaks.json              # projector's machine-readable peaks (analyzer's correlation joins against it)
├── baseline/
│   ├── benchmark_results.md         # benchmarker's baseline report
│   └── serve.log, *.json            # baseline run artifacts
├── tuning/
│   ├── extra_llm_api_options.yaml           # live server tuning (optimizer edits this)
│   └── extra_llm_api_options.accepted.yaml  # last accepted snapshot (orchestrator-managed)
├── reused_analysis/                 # --reuse-analysis only
│   ├── manifest.md                  #   what was imported, and from where
│   ├── prior_performance_model.yaml #   corrected source model — read-only provenance
│   └── prior_roadmap.yaml           #   source campaign's roadmap — read-only prior art
├── rounds/round_<n>/
│   ├── profile/                     # profiler: raw nsys/ncu captures, exports, logs, targeting derivations
│   │   ├── profiler_report.md       # concise human-readable capture handoff
│   │   └── profile_manifest.json    # capture identity, provenance, and artifact inventory
│   ├── analysis/                    # analyzer: analysis.md and linked derived evidence
│   │                                #   kernel_ledger.yaml when kernel coverage is enabled
│   │   ├── performance_model.yaml   # current theoretical-best model and remaining gap
│   │   └── analysis_manifest.yaml   # orchestrator: successful full analysis identity and source capture link
│   └── item_<j>_<id>/attempt_<k>/   # per item: optimization_summary.md, evaluation.md, result *.json
│       └── profile/                 # accept-evidence nsys capture (APPROVEd attempts only)
├── final_verification/
│   ├── verification_report.md       # QA's independent verification (+ its artifacts)
│   └── analysis/                    # final_analyzer: offline reconciliation after QA
│       ├── analysis.md              # final measured state, model and unresolved gap
│       └── performance_model.yaml   # authoritative final model for the reporter
├── optimization_report.md           # reporter deliverable
├── optimization_report.html         # self-contained interactive companion (1:1)
├── progress.yaml                    # structured audit log (agents write via MCP tools)
└── .perf_optimize_state.json        # resume checkpoint
```

## Notes

- **Agent operator guide.** A `perf-optimize` project skill ships alongside
  this workflow, at
  [`agent-flow/.claude/skills/perf-optimize/SKILL.md`](../../../.claude/skills/perf-optimize/SKILL.md).
  It teaches Claude Code to reach for this workflow instead of hand-rolling
  a tune loop, and covers preflight, authoring `task.yaml`, driving the
  campaign and reading the result. Bringing up the node and container is
  left to your own site's recipe. To drive the CLI yourself instead, this
  README and [`task.example.yaml`](task.example.yaml) are the operator
  guide.
- **Session scoping.** Agent sessions match each role's unit of work:
  the profiler runs stateless for each capture; the analyzer keeps one
  session across the whole campaign (it must remember the roadmap it
  authored), the optimizer's session spans one
  item's retry attempts and is reset between items, and the evaluator /
  qa run stateless — the judges always get fresh eyes, and no role drags
  a long campaign's stale context into later decisions. (The
  benchmarker, projector, qa, and reporter run once each. After QA, the
  existing analyzer runs once more in final reconciliation mode.)
- **Prompt extensions.** `PromptBundle.with_extensions(profiler=...)`
  customizes capture instructions independently of
  `with_extensions(analyzer=...)`, which customizes offline analysis and
  planning. The composed prompts are snapshotted separately as
  `prompts/profiler.md` and `prompts/analyzer.md`. Put server launch,
  capture tooling, and export guidance in the profiler extension; put
  interpretation, taxonomy, findings, and roadmap guidance in the
  analyzer extension.
- **Serve tuning lives in the workspace.** Every `trtllm-serve` launch in
  this workflow passes
  `--extra_llm_api_options <workspace>/tuning/extra_llm_api_options.yaml`
  (seeded from the task's `extra_llm_api_options` file, else `{}`), so
  config optimizations are applied by editing that one file. The
  perf-analyze convention (flag only when task.yaml sets it) does not
  apply here.
- **Canonical command templates.** All measurements reuse perf-analyze's
  canonical `benchmark_serving.py` / `nsys profile` / `ncu` templates at
  the configured operating point(s) — one run per `benchmark.concurrency`
  point in Pareto-curve mode — so numbers stay comparable across the
  whole campaign. Every nsys capture — the profiler's round profile and
  the evaluator's accept-evidence capture alike — is exported to
  `.sqlite` and decomposed with the `internal-perf-nsight-system-analysis` skill
  into `nsys_analysis/`, so "the launch gaps shrunk" is a measured
  per-iteration budget on both sides rather than an eyeballed kernel
  table. The accept-evidence capture runs that pipeline **comparative**
  against the previous capture of the accepted state, so the mechanism
  check reads signed deltas out of `difference/rank-0/` instead of
  comparing two trees by eye. Kernels are classified with the
  checked-in TRT-LLM taxonomy
  (`perf_analyze/assets/taxonomy_trtllm.json`). The profiler's decomposition
  supports capture targeting; the analyzer regenerates its own derived
  outputs and extends the taxonomy per workload before quoting any
  category number. The profiler's ncu deep dive is bounded
  (`--launch-count`, kernel filter from the top decomposition kernels)
  and the analyzer interprets its saved results with the
  `perf-nsight-compute-analysis` skill; both
  degrade gracefully when the tool or the skill is unavailable.
- **Multi-rank capture (`profile.profile_ranks`, default `[0]`).**
  Listing several ranks wraps each of them separately inside the
  launcher step and unlocks the skill's rank-jitter step, whose
  `pinned`-vs-`rotating` straggler verdict is the only evidence that
  separates "waiting on a slow rank" from "waiting on the network".
  Above world size 1 a bare `trtllm-serve` cannot deliver it — its
  workers are `MPI.COMM_SELF.Spawn`ed and nsys does not follow them — so
  per-rank traces need the Slurm `trtllm-llmapi-launch` shape. An
  imbalance item is filed under the category of the imbalanced *work*
  (`compute` for uneven expert load, and so on), never `communication`:
  the jitter wait shows up inside a collective but bucketing, overlap
  and interconnect levers cannot recover another rank's lateness.
- **Per-kernel coverage contract (optional).** A
  `profile.kernel_coverage` block (empty mapping selects defaults:
  `min_share_pct: 0.5`, `coverage_target_pct: 95`) requires nsys+ncu
  coverage of every kernel above the share threshold, extending capture
  until the target is reached. The profiler owns targeting and capture
  coverage. Every analyzer turn writes
  `rounds/round_<n>/analysis/kernel_ledger.yaml`, answering whether each
  kernel can be eliminated, made faster, fused or overlapped. Each answer
  references a roadmap item or an evidence-backed dismissal, with
  consumers, guards, neighbors or independent partners where relevant.
  Materiality converts GPU-time shares through `coverage.gpu_busy_pct`
  before comparing against wall-clock gains. Validation rejects missing
  questions, invalid item references or insufficient coverage.

  Kernel and shared-region models provide supporting derivations for
  `performance_model.yaml`; they do not create another report headline or
  replace end-to-end accounting. They preserve measurements and model
  revisions across replan turns. Full kernel tables remain in the ledger,
  linked from **Gap analysis** only where they explain a material gap.

- **Optimization casebook.** The benchmarker/analyzer load the
  `trtllm-agent-toolkit:perf-optimization-casebook` skill as read-only
  reference; the optimizer uses it *actionably* (how-to-apply /
  verification / rollback guidance). All roles degrade gracefully when
  the skill is not installed.
- **SOL projection (default-on stage).** Unless `sol.enabled: false`,
  the projector follows `internal-perf-sol-analysis`. The workflow resolves
  skill availability before launch; without that internal skill it falls
  back to `perf-analysis` and named hardware sources. Unavailable peaks or
  latency measurements remain explicit, and an ungrounded projection is
  recorded as unavailable. `sol_projection.md` and `sol_work/peaks.json`
  preserve the initial derivation. The analyzer revises the current model
  from later evidence, using measured-to-SOL correlation when its inputs
  are available. The evaluator and QA retain measured-versus-measured gates.
- **Cost.** The profiler runs every round that follows an accept (nsys plus
  the bounded ncu deep dive by default); set `profile.methods: [nsys]`
  to trim it. When the standing runtime profile is still current, the
  next round opens replan-only and pays no GPU time at all — see *What a
  round costs* above. Each evaluator attempt runs a
  full benchmark, each **accepted** attempt additionally pays one
  profiled replay (the accept-evidence capture), and the final
  verification runs one more benchmark at campaign end. Final model
  reconciliation uses saved evidence without additional GPU work. The per-item
  evaluator benchmark is the irreducible price of per-item attribution;
  raising `max_items_per_round` amortizes a round's capture across
  more serial items or widens a parallel batch. Parallel execution trades extra
  isolation/integration work for concurrency. Across the campaign,
  `max_rounds` remains the primary round budget.
- **Local vs Slurm.** With a `slurm-environment` block, every
  server-launching role (benchmarker, profiler, optimizer, evaluator,
  integrator, and qa) is
  augmented with the Slurm container-bootstrap guidance, exactly like
  perf-analyze. The analyzer works from local saved artifacts and needs
  no server bootstrap. The projector launches no servers; under Slurm
  it runs on the login node and records the latency constants as unmeasured.
