# perf-optimize

Iteratively **applies** TensorRT-LLM serving optimizations — the acting
counterpart to `perf-analyze` (which only diagnoses). Measures a
`trtllm-serve` baseline, plans evidence-grounded optimizations into a
machine-readable `roadmap.yaml`, evaluates a round's selected items in
serial or parallel worktrees, integrates parallel candidate-ready results, gates every
change on code quality / functionality / measured gain, runs the configured number of rounds,
independently verifies the final state, and reports expected-vs-measured
results.

`--reuse-analysis <dir>` imports a previous perf-analyze / perf-optimize
run's baseline, SOL projection and findings, starting the campaign at the
optimize stage (round 1's analyzer then plans from imported findings).
Add `--reanalyze` to reinterpret that run's saved captures before planning,
without profiling again in round 1.

```
benchmarker → (projector) → [round loop × max_rounds] → qa → reporter
 (baseline)  (SOL ceiling)                        (final verification)

round loop:
  (profiler) → analyzer → optimizer ⇄ evaluator item pairs
                         (serial or parallel worktrees;
                          ≤ max_attempts_per_item)
                                      ↓ (parallel only)
                                  integrator
                          (combine, benchmark, verdict)
```

- **benchmarker** — serves the checkpoint, runs the canonical
  `benchmark_serving.py` at the configured operating point(s) (a
  `benchmark.concurrency` list means one run per point — Pareto-curve
  mode), writes `baseline/benchmark_results.md`. This anchors
  `roadmap.yaml`'s `baseline.value` (and `baseline.curve` in curve
  mode).
- **projector** *(on by default — skipped only when `task.yaml` sets
  `sol.enabled: false`)* — runs
  **once per campaign**, between the baseline and round 1: derives the
  analytical speed-of-light (SOL) ceiling for this model/hardware/
  operating point per the `internal-perf-sol-analysis` skill (hardware
  peaks from the skill's calculator, latency constants measured when a
  GPU is reachable, the α-β-u arithmetic written out, the model
  architecture read from the checkpoint's `config.json`), and writes
  `sol_projection.md` with a
  baseline-vs-SOL gap analysis (curve mode: per point), plus the
  machine-readable `sol_work/peaks.json` the analyzer's per-round
  correlation joins against. The ceiling is
  a property of the hardware + model + operating point — later rounds
  compare against the same projection rather than re-deriving it.
- **profiler** — runs before the analyzer when the current runtime needs
  fresh evidence. Owns server launch and cleanup, nsys/ncu captures,
  exports, capture quality and coverage checks, and runtime/configuration
  provenance. It preprocesses the nsys timeline with the
  `internal-perf-nsight-system-analysis` skill to choose ncu targets
  (ranked by in-window union time, not capture-wide `kern_sum`) over the
  same iteration window. Raw captures and capture-local targeting
  derivations live in `rounds/round_<n>/profile/`, identified by
  `profile_manifest.json`. The profiler authors no roadmap or findings.
  A separate successful checkpoint preserves these captures if the
  analyzer fails; retrying analysis does not repeat the GPU capture.
- **analyzer** — runs once per round, interpreting saved evidence and
  writing analysis artifacts under `rounds/round_<n>/analysis/`. It
  leaves the capture unchanged; after a successful full analysis, the
  orchestrator writes `analysis_manifest.yaml` linking that analysis to
  the source `profile_dir` and `capture_id`. It
  launches no server or GPU profiler. It can regenerate the nsys
  decomposition, classify per-iteration busy/idle and compute-absent
  time (launch-starved / blocking / dependency-stalled), and interpret
  saved ncu results with `perf-nsight-compute-analysis` (per-kernel SOL%,
  occupancy, warp stalls → bound class). With `--reuse-analysis` it
  plans from imported findings; with `--reuse-analysis --reanalyze` it
  first recomputes the analysis from imported captures. After a round
  that accepted nothing, it runs **replan-only** from the standing findings and
  evaluator verdicts — see *What a round costs*. It writes/updates
  `roadmap.yaml` — items ordered by `expected_gain_pct` (bottleneck share
  removed, casebook-grounded), never by fix ease, each item's evidence
  drawn across the analyses (nsys timeline, ncu kernel analysis, SOL
  correlation when the projector ran) rather than the timeline alone,
  and each kernel-speedup item bounded by the operator's measured
  headroom rather than its cost. The timeline analysis's own
  `items.json` is accounted for row by row in `roadmap.yaml`'s
  `nsys_items` block — every opportunity it found becomes an item or a
  dismissal with evidence, and the orchestrator validates that the
  moment the turn ends (see *The nsys opportunity-coverage gate*).
  After fresh captures, the analyzer updates statuses /
  ordering without rewriting history — a round following one that left
  the standing runtime profile current has no shift to find and re-plans
  instead. When the projector ran, it reads
  `sol_projection.md` as context (not evidence): the projected headroom
  and bound mix inform the ranking and sanity-bound each item's
  `expected_gain_pct`; measured trace evidence always outranks the
  projection. Each fresh analysis also runs the skill's **measured↔SOL
  correlation** (`sol_calc.py analyze`): the round's traces roll up
  into `analysis/regions.json`, join against the projector's
  `sol_work/peaks.json`, and the resulting per-op table (% of SOL,
  gap, bound per region) lands in the findings' *SOL correlation*
  section — the sharpest re-ranking signal the roadmap gets. And it
  never exhausts the roadmap silently: leaving no
  actionable pending item while projected headroom remains obliges a
  *Remaining-gap attribution* section in `profile_findings.md` — every
  part of the gap gets a new item or an evidence-backed reason it
  cannot be closed in this campaign (unexplained parts stay labeled
  unexplained). With `profile.kernel_coverage`, it maintains a single
  **kernel ledger** containing all four kernel questions and the best
  current theoretical model, revised every round from facts about the
  workload, hardware and actual implementation.
- **optimizer** — one independent persistent optimizer is created for each
  dispatched item. With `item_execution: parallel`, up to
  `max_items_per_round` pairs start from the same frozen round base.
  They run concurrently in separate exclusive Slurm allocations. On a
  shared local runtime they run one at a time, including smoke tests,
  so candidates cannot compete for GPUs or port 8000. Both schedules
  retain the frozen base and integration stage. With `serial`, each
  worktree starts from the latest accepted
  campaign state. In both modes, approved and rejected terminal items
  consume the shared `max_items_per_round` budget. Retries for one item
  are always sequential:
  `approach: config` edits `tuning/extra_llm_api_options.yaml`;
  `approach: code` edits the TRT-LLM source (installed-package check
  first). Smoke-checks the server, never benchmarks, never commits.
  When the projector ran, it reads `sol_projection.md` as context (not
  spec): where the item leaves a choice of realization variants or knob
  values, it aims at the binding ceiling and records an `SOL alignment`
  line in its summary — the projection never expands the item.
- **evaluator** — reviews the diff, verifies functionality (sanity
  completions; targeted tests for code items), re-measures with the
  canonical benchmark, and applies the acceptance gate (below). Emits a
  structured three-way verdict with a reason category: **APPROVE** marks the
  isolated result `candidate_ready`, **PUSH_BACK** loops back to the optimizer with
  actionable feedback (up to `max_attempts_per_item` attempts total),
  and **REJECT** fails the item terminally — the judge's call that no
  retry would help, saving the benchmarks a doomed retry would burn.
  Stateless — every attempt is judged with fresh eyes. Negative
  verdicts close with a one-line *Gap implication*
  (mechanism-already-present / mechanism-inapplicable /
  applied-but-no-gain / blocked-by-constraint) — the projection-free
  evidence the analyzer's re-planning and the report's remaining-gap
  accountability are built from.
- **integrator** *(parallel only)* — after every item worker reaches `candidate_ready` or
  `failed`, combines candidate commits/configs in roadmap order in a separate
  integration worktree, resolves only conflicts and minimal combination
  defects, benchmarks the combined state, and emits the authoritative
  `APPROVE | FALLBACK_BEST | REJECT` verdict. It may diagnose/remediate twice;
  after that it validates only the best standalone candidate, or rejects all.
  Before applying the structured verdict, the Python orchestrator verifies
  that included ids are non-empty candidate-ready items and cross-checks the
  reported threshold, measured gain, and Pareto-curve regression budget.
- **candidate evidence capture** — inside the evaluator's APPROVE turns,
  not a separate stage: after the clean measurement, the evaluator
  relaunches the server under the canonical nsys wrap, replays the load
  once, and saves `attempt_<k>/profile/` (trace, `nsys_stats.txt`,
  `nsys_analysis/`, replay log), then writes a *Kernel evidence* section
  comparing the capture against the frozen accepted state — verifying
  the item's claimed mechanism is actually visible in the trace. Because
  rejected attempts are hard-reverted. A final-state integration
  capture, when produced, becomes the reporter's newest accepted-state
  profile.
  The capture is diagnostic, never a measurement (fresh relaunch; the
  verdict comes from the un-profiled run); no capture is made when
  `nsys` is not in `profile.methods`, and a failed capture never flips
  a verdict. An accepted runtime change without a successful matching
  capture invalidates the current-evidence pointer; older traces remain
  historical evidence and are not presented as the final accepted state.
- **qa** — the campaign's **final verification**, run once after the
  round loop (and skipped when no item was accepted): stateless
  fresh-eyes benchmark + sanity completions (+ an accuracy eval iff
  `task.yaml` configures one), computing the verified cumulative
  improvement vs baseline that headlines the report. It makes no
  loop decision — the loop is already over.
- **reporter** — synthesizes `optimization_report.md` + a 1:1
  `optimization_report.html`: verified cumulative improvement, the
  optimization trajectory (baseline → each accepted serial item or parallel integration → final
  verification, rendered as a line chart in the HTML),
  expected-vs-measured per applied item, a kernel-level before/after
  comparison from the round profiles and accept-evidence captures,
  failed attempts with reasons, final config/code diff, and the
  remaining roadmap as future work. When the projector ran, the report
  gains a *Projection vs Measured* section (baseline vs final % of SOL
  — how much of the projected headroom the campaign captured; the
  curve-mode Pareto chart overlays the SOL-projected ceiling as a
  dotted third polyline), closed by a **remaining-gap accountability**
  breakdown: every part of the remaining gap-to-SOL is verdicted
  `closed` / `infeasible: <constraint>` / `untried` / `unexplained`,
  each verdict citing an artifact (a failed item's *Gap implication*,
  a round's *Remaining-gap attribution*, the projection's caveats) —
  a campaign may end short of the ceiling, but never without saying
  why. With `profile.kernel_coverage`, the **Kernel Coverage**
  section renders both the four questions and theoretical-model-versus-silicon
  comparison from the latest kernel ledger, including evidence for model
  revisions and explicitly unexplained discrepancies.

The orchestrator — not the agents — owns the roadmap lifecycle fields and
the git state of the TRT-LLM checkout, driven by the evaluator's
structured decisions in `progress.yaml`.

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
report gains a *Pareto Improvement* section + chart (x = tok/s/user,
y = tok/s/gpu, baseline vs final). Note the benchmark cost multiplies by
the point count on **every** measurement (baseline, each evaluator
attempt, the final verification) — keep the list short (~3–5 points).

## When the loop stops

No agent decides when to stop. The loop runs exactly
`optimize.max_rounds` rounds unless one of two deterministic,
orchestrator-enforced breaks fires first:

- **Round budget spent** — `optimize.max_rounds` rounds have closed, each
  selecting up to `max_items_per_round` candidates.
- **Roadmap exhausted on an unchanged build** — no pending item
  promises at least `noise_floor_pct` through an allowed approach
  (checked after every analyzer turn and after every item's terminal
  outcome) *and* nothing has been accepted since the analysis that
  planned it. A fresh profile would then find the same nothing, at full
  profile cost. When the roadmap runs dry with accepts outstanding the
  loop does **not** close: the build those accepts produced has never
  been analyzed, so it spends one more round profiling what they exposed
  (budget permitting) and closes on *that* verdict.
- **Target met** — the optional `optimize.target_improvement_pct` is
  reached by the roadmap ledger's cumulative gain (`current_best` vs
  `baseline`; curve mode: the mean of per-point gains), checked after
  every accepted item.

Either way the campaign proceeds to the one-shot final verification
(skipped when nothing was accepted) and the reporter.

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
  opening after an accept. The profiler captures the current runtime (nsys + ncu per
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
  current: the predecessor accepted nothing. The analyzer launches no
  server and runs no profiler; it plans from the standing analysis plus
  the round's evaluator verdicts, marking
  disproven items obsolete, bounding the gains the measurements cap, and
  adding what the failures imply. Those verdicts are the round's real
  yield — an item measured dead is evidence about *this* build — and
  converting them into roadmap edits and model updates is what the turn
  is for. The analyzer writes a new kernel/model ledger using standing
  measurements and the latest evidence, preserving prior revisions and
  recording any justified change to the model. The same ledger validation
  applies without a new ncu capture.

The orchestrator selects the mode, and a replan round is not a skipped
round: if it leaves nothing actionable, that is the roadmap-exhausted
break and the campaign closes.

Capture completion and analysis completion have separate checkpoints.
If the analyzer fails after a successful capture, rerunning the command
resumes analysis from the preserved `profile/` directory. An unfinished
capture remains the profiler's responsibility. Re-analysis imported from
another workspace is not proof that the local runtime is current, so it
does not earn the local profile-currency marker used by replan-only rounds.

That break only fires at the top of a round, never mid-round. A roadmap
that runs dry between items ran dry against a plan written *before* the
round's measurements existed, so the loop spends one more round — free
when nothing was accepted, a profile when something was — and closes on
the plan the analyzer makes against them. What ends a campaign is an
analyzer turn that has seen the evidence and still finds nothing, not
the plan simply running out.

## Reusing a previous run's analysis

`--reuse-analysis <dir>` seeds a fresh workspace from a previous
`perf-analyze` run or `perf-optimize` campaign, so the campaign starts at
the optimize stage instead of re-deriving what that run already measured:

| imported | from a perf-analyze workspace | from a perf-optimize workspace | replaces |
| --- | --- | --- | --- |
| baseline report + result JSONs | `benchmark_results.md` | `baseline/benchmark_results.md` | the benchmarker |
| SOL projection + `sol_work/` | `sol_projection.md` | `sol_projection.md` | the projector |
| profile findings (+ `kernel_ledger.yaml`) | `profile_findings.md` and companion artifacts | newest `rounds/round_<n>/analysis/` | round 1's analysis, unless `--reanalyze` |
| raw profile captures | workspace trace files | `rounds/round_<n>/profile/` with `profile_manifest.json` (legacy `analysis/` layouts also supported) | round 1's profiler |
| roadmap (as read-only prior art) | — | `roadmap.yaml` | nothing |

Round 1's analyzer then runs **plan-only**: it reads the imported
evidence, checks that it actually describes this task (same model,
parallel mapping, operating point), runs the dormant-capability sweep,
and writes `roadmap.yaml` — launching no server, no profiler, and no
benchmark. Round 2 profiles normally: the imported traces describe
*another* run's build, so they never stand in for one this campaign
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
  `kernel_ledger.yaml`, `sol.json` and `regions.json` provide evidence for
  the analyzer, which writes this round's own kernel/model ledger and
  roadmap with local item references, source measurement conditions and
  citations, and an initially empty local model revision history. With `--reanalyze`, it regenerates derived artifacts from the
  saved captures.
- **The baseline is inherited, not re-measured.** Every gain this
  campaign reports is computed against numbers measured by the source
  run, so the two must describe the same system. The import writes
  `reused_analysis/manifest.md` recording what came from where, the
  analyzer owes a fit check against it, and the report says so in
  Configuration. If the hardware or checkpoint differs, don't reuse the
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

A run with `profile.kernel_coverage` validates this round's kernel/model
ledger after every analyzer turn, including plan-only reuse and replan-only
rounds. These rounds preserve source measurement provenance while updating
the model and four-question reasoning from available facts; they need no
new capture.

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
├── sol_projection.md                # projector's SOL ceiling + baseline-vs-SOL gap (blank when sol.enabled: false)
├── sol_work/peaks.json              # projector's machine-readable peaks (analyzer's correlation joins against it)
├── baseline/
│   ├── benchmark_results.md         # benchmarker's baseline report
│   └── serve.log, *.json            # baseline run artifacts
├── tuning/
│   ├── extra_llm_api_options.yaml           # live server tuning (optimizer edits this)
│   └── extra_llm_api_options.accepted.yaml  # last accepted snapshot (orchestrator-managed)
├── reused_analysis/                 # --reuse-analysis only
│   ├── manifest.md                  #   what was imported, and from where
│   └── prior_roadmap.yaml           #   source campaign's roadmap — read-only prior art
├── rounds/round_<n>/
│   ├── profile/                     # profiler: raw nsys/ncu captures, exports, logs, targeting derivations
│   │   └── profile_manifest.json    # capture identity, provenance, and artifact inventory
│   ├── analysis/                    # analyzer: profile_findings.md, regenerated nsys_analysis/ (+ regions.json / sol.json when the projector ran;
│   │                                #   + kernel_ledger.yaml with a profile.kernel_coverage block)
│   │   └── analysis_manifest.yaml   # orchestrator: successful full analysis identity and source capture link
│   └── item_<j>_<id>/attempt_<k>/   # per item: optimization_summary.md, evaluation.md, result *.json
│       └── profile/                 # accept-evidence nsys capture (APPROVEd attempts only)
├── final_verification/
│   └── verification_report.md       # QA's one-shot independent verification (+ its artifacts)
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
  benchmarker, projector, qa, and reporter run once each.)
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
  `profile.kernel_coverage` block in `task.yaml` (empty mapping =
  defaults: `min_share_pct: 0.5`, `coverage_target_pct: 95`) upgrades
  the ncu dive from "top nsys kernels" to **every kernel above the
  share bar** (enumerated from the fresh kern_sum, extended until the
  coverage target is reached, captured over up to 3 bounded ncu passes
  that re-filter on still-missing stems so once-per-step kernels are
  not starved by per-layer hot ones). The profiler owns targeting and
  capture coverage; on every turn the analyzer must then
  answer four questions per enumerated kernel — *can it be eliminated?*
  *can it be made faster?* *can it be fused with its neighbors?* *can it
  be overlapped with independent work on another stream?* — in
  `rounds/round_<n>/analysis/kernel_ledger.yaml`: every row carries the
  kernel's ncu SOL metrics/bound class plus an `elimination`, a
  `faster`, a `fusion` and an `overlap` disposition, each either a
  roadmap item id or an evidence-backed dismissal (`mandatory-math`,
  `padding-minimal`, `already-hoisted`, `fast-path-active`,
  `fast-path-blocked`, `at-sol-floor`, `below-materiality` with
  arithmetic, `multi-consumer-pinned`, `already-fused`,
  `phase-boundary`, `needs-rebuild`, `graph-disabled`,
  `no-independent-partner`, `resource-saturated`, ...);
  `needs-rebuild` is valid only when a written-from-scratch replacement
  kernel routed from the Python call site is also ruled out, not merely
  because the incumbent ships compiled; elimination rows record what
  consumes the output (or the guard that selected this path), fusion
  rows the observed neighbors from the trace, and overlap rows the
  candidate partner plus the evidence the two are serialized today.

  The four are ordered by how much they presuppose, each asking less
  than the last. **Elimination** presupposes only that the kernel runs
  today, and is first because a `yes` moots the rest and recovers the
  row's *whole* share rather than a fraction — it covers redundant work,
  work over padded/masked data, per-step recompute of something
  invariant, and the accidental slow path (an `is_fused=False` fallback
  firing because a gated fast path did not), which is the per-kernel
  per-round teeth on round 1's dormant-capability sweep. **Faster** and
  **fusion** presuppose the work is necessary *and* that the kernel must
  run alone. **Overlap** drops the alone assumption: a kernel at its
  bound-class ceiling (`at-sol-floor`) whose neighbors move only
  mandatory bytes (`neighbors-at-bandwidth-floor`) is legitimately
  closed on both and can still give back most of its share by running
  concurrently with independent work — realized through the checkout's
  own `maybe_execute_in_parallel` / `AuxStreamType` idiom, and gated on
  CUDA graphs being enabled (multi-stream no-ops without them). The
  ledger also carries `coverage.gpu_busy_pct`, the busy share of the
  profiled window: `share_pct` is a share of *GPU time* while
  `noise_floor_pct` and `expected_gain_pct` are shares of *wall clock*,
  so every materiality claim converts through it rather than
  overstating candidates by `1/busy` on a host-bound deployment.

  The orchestrator schema-validates the ledger after every analyzer
  turn (all four dispositions per row, `item` refs resolving to real
  roadmap ids, coverage ≥ target, `gpu_busy_pct` present) and **aborts
  the stage on an incomplete ledger**, so the campaign cannot conclude
  while a hot kernel's elimination, optimization, fusion, or overlap
  possibility was never considered; the reporter's *Kernel Coverage* section resolves
  the final ledger's dispositions to campaign outcomes and itemizes the
  untried tail. Requires `nsys` + `ncu` in `profile.methods`; costs
  extra profiling wall-clock in rounds that collect fresh captures.

  The same ledger also contains the analyzer's **best theoretical
  performance model**. Every kernel references a model, and several
  kernels may share a logical-region or iteration model so fusion,
  elimination and overlap do not force artificial kernel boundaries or
  double-count savings. Each model states its operating point, derivation,
  assumptions, hardware constraints, predicted milliseconds, matching
  measured milliseconds and evidence, unexplained discrepancy, and next
  discriminating experiment. Missing predictions or measurements are
  explicit `null` values with an explanation and next test.

  On every turn, including replan-only rounds, the analyzer writes a new
  `analysis/kernel_ledger.yaml` using standing measurements or a new
  capture as appropriate. It preserves previous models and appends
  evidence-backed changes in `model_revisions`, including the old and new
  value of each changed field. Experiments improve the implementation when
  they expose inefficiency and improve the model when they expose missing
  costs or invalid assumptions. A failed optimization alone justifies
  neither relaxing the prediction nor claiming the gap is unavoidable.
  A measurement below a theoretical lower bound calls for investigating
  model or measurement compatibility, not clamping the result. Convergence
  means facts explain the residual under matching conditions; an
  unexplained gap remains open even when the campaign ends.

  The reporter renders both kernel dispositions and model-versus-silicon
  evidence in one *Kernel Coverage* section. The evaluator and QA retain
  their measured-versus-measured gates. No separate accounting ledger,
  implementation-target layer, partition buckets or mandatory attribution
  machinery is required, and the SOL projector remains optional context.

- **Optimization casebook.** The benchmarker/analyzer load the
  `trtllm-agent-toolkit:perf-optimization-casebook` skill as read-only
  reference; the optimizer uses it *actionably* (how-to-apply /
  verification / rollback guidance). All roles degrade gracefully when
  the skill is not installed.
- **SOL projection (default-on stage).** The projector runs unless
  `task.yaml` sets `sol.enabled: false`; it follows the
  `trtllm-agent-toolkit:internal-perf-sol-analysis` skill as its
  methodology. That skill is `internal-` prefixed, so open-source
  toolkit builds strip it while keeping `perf-analysis`; which of the two
  this session has is resolved **in Python** before the campaign starts
  (`perf_analyze.sol_methodology`, one ~1 s probe — a session
  connection, no model call — failing open to the SOL skill if it cannot
  run), so the projector is told to load a skill that is actually there.
  Without the SOL skill it loads `perf-analysis` instead and works the
  same methodology without a calculator: the peaks come from named
  sources, marked as not calculator-resolved, and no
  `sol_work/peaks.json` is written — so the analyzer's per-round
  correlation degrades to its honest "Correlation unavailable" line. It
  degrades to a "Projection unavailable" file when no ceiling can be
  grounded at all, and never fabricates one. It
  runs once per campaign: the ceiling depends only on the hardware +
  model + operating point, so every later round compares against the
  initial `sol_projection.md` as provenance, while the analyzer updates its
  current model from new facts. Consumers: the analyzer (roadmap ranking
  context; each round it also joins its fresh per-op measurements
  against the projector's `sol_work/peaks.json` with the skill's
  `sol_calc.py analyze` and reports the joined table in
  `profile_findings.md`'s *SOL correlation* section; plus the
  remaining-gap attribution owed whenever it leaves
  the roadmap exhausted with headroom remaining), the optimizer
  (aiming each item's realization at the binding ceiling — context,
  never an expansion of the item), and the reporter (the *Projection
  vs Measured* headroom-captured section with its remaining-gap
  accountability breakdown); the evaluator and qa deliberately see
  nothing of it — their gates stay measured-vs-measured so an
  analytical model can never anchor a verdict. For a spec or mapping
  that stays uncertain, the projector is pointed at the
  `internal-glean-search` skill / `internal-glean-specialist` subagent
  as read-only reference, used only if it is installed in the
  session.
- **Cost.** The profiler runs every round that follows an accept (nsys plus
  the bounded ncu deep dive by default); set `profile.methods: [nsys]`
  to trim it. When the standing runtime profile is still current, the
  next round opens replan-only and pays no GPU time at all — see *What a
  round costs* above. Each evaluator attempt runs a
  full benchmark, each **accepted** attempt additionally pays one
  profiled replay (the accept-evidence capture), and the final
  verification runs one more benchmark at campaign end. The per-item
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
