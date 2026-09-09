"""Schema for the perf-optimize ``headroom_ledger.yaml`` contract.

The headroom ledger is the campaign's **accounting layer** over the
measured↔SOL correlation: one row per *part* of the deployment, each
carrying where that part's gap-to-SOL sits, what every round proved about
it, and — the **target layer** — what a named, buildable implementation
would achieve. It answers the question the roadmap cannot: an
optimization that is proposed but not accepted leaves behind a
machine-readable fact instead of a paragraph.

Without it, a failed item's entire durable payload is ``status: failed``
plus ``measured_gain_pct``; the per-part attribution its evaluator wrote
("the remaining gap is a mapping/layout problem, not a launch-geometry
one — no launch-tuning item should be authored against these kernels
again") survives only as prose inside a round directory, and the next
round is free to re-propose exactly what was disproved.

Three tiers, and the gap between each pair has a different owner::

    sol_ms      <=   target_ms    <=   measured_ms
    (physics)      (best known)       (today)

    measured - target  = the ENGINEERING gap: a better implementation is
                         known and buildable. This is where roadmap items
                         come from, and what `expected_gain_pct` is sized
                         against.
    target   - sol     = the STRUCTURAL gap: no known implementation
                         reaches the floor. Research it, or record it.

``sol_ms`` is a bound, not a description: it never drifts toward what was
measured. It moves only through a ``model_revisions`` entry proving the
*model* was wrong. ``measured_ms`` moves only through
``measurement_revisions``; ``target_ms`` only through ``target_revisions``
(the plan was wrong — the ceiling still is not).

Shape::

    version: 1

    operating_point:            # the model is valid only for these points
      concurrency: [64, 512]    # the bracketing focus concurrencies; every
      isl: 1024                 # part is measured at BOTH. An isl/osl/build
      osl: 1024                 # change invalidates every part and forces a
      build_sha: 1418149a84     # re-derivation, not a revision.
      node: b200-node-a
      capture_state: gpu-bound  # gpu-bound | host-bound | mixed

    timing:
      step_ms: 18.372           # anchored median iteration
      kernel_ms: 17.335         # per-rank-step kernel time

    coverage:                   # MANDATORY, and must reconcile to step_ms.
      modeled_kernel_ms: 10.210 #   analytic parts — a sol_ms exists
      modeled_pct: 58.9
      empirical_kernel_ms: 6.342  # ncu bound class only, no analytic ceiling
      unmodeled_kernel_ms: 0.784  # accounted for but not bounded (the
                                  #   below-bar `other` tail)
      non_kernel_ms: 1.037        # host/scheduler/idle, outside the model
      residual_ms: 0.001          # what the buckets miss; recorded, never
                                  #   absorbed into a neighbour

    parts:
      - id: gdn_state:linear_attn:bf16   # == sol.json per_op[].region
        source: analytic                 # analytic | empirical | unmodeled
        at:                              # one entry per bracketing point
          64:  {measured_ms: 0.412, sol_ms: 0.267, gap_ms: 0.145}
          512: {measured_ms: 3.1321, sol_ms: 2.1386, gap_ms: 0.9935}
        sensitivity: flat                # flat | steep; null when only one
                                         #   point was captured
        bound: memory
        kernels:                         # the join — analyzer-authored,
          - _cached_replay_kernel        #   arithmetically validated
        partition:                       # sums to gap_ms at the PRIMARY point
          closed_ms: 0.0
          attributed_ms: 0.0
          open_ms: 0.0
          unexplained_ms: 0.9935
        history:                         # orchestrator-owned
          - {round: 4, measured_ms: 3.1321, gap_ms: 0.9935}
        dispositions:                    # orchestrator-owned; one per item
          - item: opt-004                #   that touched this part. Each
            round: 3                     #   closes a LEVER, not the part.
            outcome: failed
            gap_implication: mechanism-inapplicable
            lever: gdn-launch-geometry-tuning
            note: >-
              Tuned launch mapping is gated on T == 4; this MTP draft-2
              deployment runs T == 3, so it is dormant and unreachable.
            directive: no-launch-tuning-item-against-these-kernels
            evidence: rounds/round_3/item_1_opt-004/attempt_1/evaluation.md
        attribution:                     # the ONLY thing that forecloses
          attributed_ms: 0.0
          basis: null                    # kernel-ledger-exhaustive |
          note: null                     #   convergent-levers
        target:                          # optional; see the target layer
          target_ms: 2.62
          structure: "One persistent kernel per layer: load S and the conv
            window once into registers/SMEM, ... store S once."
          basis: derived                 # existing-impl | published |
          basis_ref: "same recipe arithmetic as sol_ms (qwen35_hybrid.py)"
          today: "5 kernels: _cached_replay, _causal_conv1d_update, ..."
          achieved_efficiency:
            value: 0.78                  # MEASURED, with a named source
            source: "moe_gemm_fc1 demonstrates 78% MBU in this trace"
          delta:                         # sums to measured_ms - target_ms
            - {cause: state-round-trip, ms: 0.42, evidence: cuda_gpu_trace step 120}
            - {cause: unattributed, ms: 0.09}
          falsifier: >-
            if the delta rule's S^T k reduction cannot be held in registers
            at heads_local=16, the fusion is unbuildable.

    part_lifecycle:
      - {round: 2, part: logits_upcast:bf16_to_fp32, event: eliminated,
         by: opt-001, measured_ms: 0.9792, sol_ms: 0.2871}

    model_revisions:        # the ONLY way sol_ms moves
      - {round: 3, part: <id>, sol_ms: {from: 1.0, to: 2.0},
         cause: missing-factor, detail: "...", evidence: "<recipe>@<sha>"}
    measurement_revisions:  # the ONLY way a recorded measured_ms moves
      - {round: 4, part: comm:allreduce:tp2,
         measured_ms: {from: 3.791, to: 1.350},
         cause: mixed-state-capture, detail: "...", evidence: "..."}
    target_revisions:       # the ONLY way target_ms moves
      - {round: 3, part: <id>, target_ms: {from: 2.62, to: 2.94},
         cause: multi-consumer-pinned, detail: "...", found_by: "opt-009 / 2",
         evidence: "...", falsifier_fired: true}

Ownership mirrors ``roadmap.yaml`` (:mod:`.roadmap_schema`):

- The **analyzer** authors part content — ids, sources, the ``at`` rows,
  the kernel join, ``partition``, ``attribution`` claims, ``target``
  blocks, ``part_lifecycle`` and every revision list.
- The **orchestrator** owns ``dispositions`` and ``history``, appended by
  :func:`append_dispositions` / :func:`record_history` from the
  evaluator's **structured** progress fields — never by regex over prose,
  which is exactly the failure this artifact exists to fix (the
  ``Gap implication:`` line has been written four different ways inside a
  single campaign).
- The **orchestrator** validates everything: :func:`load_ledger` for
  shape, :func:`cross_validate` for the context the file does not carry.
- The **evaluator and QA never see this file**, nor SOL, nor the target
  layer. Their gates stay measured-vs-measured; an analytical model must
  never anchor an accept decision.

Two timing facts a reader has to hold:

- **The update is two-phase.** A batch-close update pairs round N's
  ``sol.json`` — measured *before* this round's accepts landed — with
  round N's verdicts. The measured refresh arrives with round N+1's
  analyzer. ``history`` is keyed by round and never back-dated.
- **``partition``, ``target`` and ``history`` are stated at the
  *primary* point** (the highest bracketing concurrency). A part flagged
  ``sensitivity: steep`` may not hold that partition at the low endpoint
  and must be ranked per scored point from ``at``.

What this validator can and cannot do, stated plainly: it enforces that a
reason of the right shape exists, that its enum value is legal, and that
its citation resolves. It **cannot** judge whether the reason is true.
That judgement stays with the analyzer, and the value of these rules is
that a wrong reason becomes attributable and re-checkable rather than
absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .kernel_ledger import DISMISSAL_TAGS, QUESTIONS, dismissal_tag

HEADROOM_LEDGER_VERSION = 1
HEADROOM_LEDGER_FILENAME = "headroom_ledger.yaml"

# Where a part's time is accounted, and how tightly it is bounded (§6).
# `unmodeled` is not a failure state — it is the honest label for time
# that is counted but carries no ceiling, so "no modeled gap" can never
# be read as "no headroom".
PART_SOURCES = ("analytic", "empirical", "unmodeled")

# Whether a part's gap holds across the concurrency bracket. A `steep`
# part ranked once, campaign-wide, is the exact mistake that made a real
# host-work reduction look worth +1.4% at every point when it was worth
# nothing at three of four.
SENSITIVITIES = ("flat", "steep")

# The capture state the numbers were read in. A mixed capture inflates
# exposed communication (compute is absent, so the collective stands
# exposed), which is a `measurement-artifact`, not a finding.
CAPTURE_STATES = ("gpu-bound", "host-bound", "mixed")

# What a terminal item proved about the part it targeted. `change-not-live`
# is the value the four-value vocabulary lacked: a change that was applied
# but never executed in the measured binary (a config key silently
# ignored, a dead code path, a flag with no read site on the model's
# path). Without it that outcome is indistinguishable from
# `applied-but-no-gain`, which wrongly bounds the part's headroom — the
# mechanism was never tested.
GAP_IMPLICATIONS = (
    "mechanism-already-present",
    "mechanism-inapplicable",
    "applied-but-no-gain",
    "change-not-live",
    "blocked-by-constraint",
)

# Only these two say anything about the part itself, so only these may
# count toward a `convergent-levers` attribution. `change-not-live` and
# `blocked-by-constraint` never do: in both, the mechanism was never
# actually tested against the part.
CONVERGENT_GAP_IMPLICATIONS = ("applied-but-no-gain", "mechanism-already-present")

DISPOSITION_OUTCOMES = ("accepted", "failed")

# An attribution is the only operation that permanently removes headroom
# from the work queue, so a single failed item can never establish one.
ATTRIBUTION_BASES = ("kernel-ledger-exhaustive", "convergent-levers")

# `none-known` is legal and expected. When the analyzer cannot name a
# real implementation it must say so; then target_ms == measured_ms and
# the whole gap is structural. "I don't know how to build this" is a
# result, not a failure to fill the form.
TARGET_BASES = ("existing-impl", "published", "derived", "none-known")

# A target with `basis_ref` that must resolve to something a reader can
# open — a kernel that exists, or a paper/report.
_REF_REQUIRED_TARGET_BASES = ("existing-impl", "published")

# Why a ceiling was wrong. Closed, because free-text causes are how "the
# SOL looked too aggressive" gets recorded as a reason.
MODEL_REVISION_CAUSES = ("missing-factor", "wrong-peak", "wrong-parallelism", "wrong-recipe")

# Why a recorded measurement was wrong.
MEASUREMENT_REVISION_CAUSES = (
    "mixed-state-capture",
    "wrong-window",
    "wrong-metric",
    "wrong-rank",
    "wrong-kernel-set",
)

# How much the recorded per-item gain can be trusted (§9.3): the scored
# arm alone, a repeated measurement, or a number its own author disowned.
MEASUREMENT_CONFIDENCES = ("single-arm", "repeated", "not-reproducible")

# The `delta[]` bucket every target must carry. Requiring the named
# causes alone to close the books would force the analyzer to invent a
# cause to balance the arithmetic — manufacturing exactly the fiction the
# target layer guards against.
UNATTRIBUTED_DELTA_CAUSE = "unattributed"

PART_LIFECYCLE_EVENTS = ("introduced", "eliminated")

_TOP_LEVEL_KEYS = (
    "version",
    "operating_point",
    "timing",
    "coverage",
    "parts",
    "part_lifecycle",
    "model_revisions",
    "measurement_revisions",
    "target_revisions",
)

_PARTITION_FIELDS = ("closed_ms", "attributed_ms", "open_ms", "unexplained_ms")
_COVERAGE_KERNEL_FIELDS = ("modeled_kernel_ms", "empirical_kernel_ms", "unmodeled_kernel_ms")
_AT_FIELDS = ("measured_ms", "sol_ms", "gap_ms")

# `gap_ms = measured_ms - sol_ms` is exact subtraction of two authored
# 4-decimal numbers, so the only slack it needs is rounding.
_MS_ABS_TOLERANCE = 1e-3
# Default reconciliation slack, overridable per task. The reference
# campaign's own two sources for "modeled ms" already disagree by 0.36%
# because the note and the roll-up were computed slightly differently; a
# validator demanding exact closure would wedge the round on it.
DEFAULT_TOLERANCE_PCT = 1.0
DEFAULT_MIN_SHARE_PCT = 0.5


class HeadroomLedgerError(ValueError):
    """Raised when ``headroom_ledger.yaml`` fails schema validation."""


@dataclass(frozen=True)
class LedgerContext:
    """The round artifacts :func:`cross_validate` checks the ledger against.

    Every field is optional: a round that could not produce one degrades
    to skipping the checks that need it rather than failing the ledger
    for an absence it did not cause.
    """

    sol: Mapping[str, Any] | None = None
    regions: Mapping[str, Any] | None = None
    kernels: Mapping[str, Any] | None = None


def _is_number(value: Any) -> bool:
    # bool is an int subclass — reject it explicitly so ``true`` cannot
    # slip through as a millisecond value.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= max(tolerance, _MS_ABS_TOLERANCE)


def _require_str(holder: Mapping[str, Any], field: str, where: str, errors: list[str]) -> None:
    if not _nonempty_str(holder.get(field)):
        errors.append(f"'{where}.{field}' must be a non-empty string, got {holder.get(field)!r}")


def _require_enum(
    holder: Mapping[str, Any], field: str, allowed: Sequence[str], where: str, errors: list[str]
) -> None:
    value = holder.get(field)
    if value not in allowed:
        errors.append(f"'{where}.{field}' must be one of {list(allowed)}, got {value!r}")


def _require_ms(holder: Mapping[str, Any], field: str, where: str, errors: list[str]) -> None:
    value = holder.get(field)
    if not _is_number(value) or value < 0:
        errors.append(f"'{where}.{field}' must be a number >= 0 (milliseconds), got {value!r}")


# --------------------------------------------------------------- shape: header


def _validate_operating_point(data: Mapping[str, Any], errors: list[str]) -> None:
    """Validate the block that says which measurements the model describes."""
    block = data.get("operating_point")
    if not isinstance(block, dict):
        errors.append(
            f"'operating_point' must be a mapping with 'concurrency', 'isl', "
            f"'osl', 'build_sha', 'node' and 'capture_state' — the ledger is "
            f"valid only for the points it was measured at; got {block!r}"
        )
        return
    points = block.get("concurrency")
    if (
        not isinstance(points, list)
        or not points
        or any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in points)
        or list(points) != sorted(set(points))
    ):
        errors.append(
            f"'operating_point.concurrency' must be a non-empty, strictly "
            f"ascending list of integers >= 1 — normally the lowest and "
            f"highest scored concurrency, so a part's sensitivity is "
            f"measurable rather than assumed; got {points!r}"
        )
    for field in ("isl", "osl"):
        value = block.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            errors.append(f"'operating_point.{field}' must be an integer >= 1, got {value!r}")
    for field in ("build_sha", "node"):
        _require_str(block, field, "operating_point", errors)
    _require_enum(block, "capture_state", CAPTURE_STATES, "operating_point", errors)


def _validate_timing(data: Mapping[str, Any], errors: list[str]) -> None:
    block = data.get("timing")
    if not isinstance(block, dict):
        errors.append(f"'timing' must be a mapping with 'step_ms' and 'kernel_ms', got {block!r}")
        return
    for field in ("step_ms", "kernel_ms"):
        _require_ms(block, field, "timing", errors)
    step, kernel = block.get("step_ms"), block.get("kernel_ms")
    if _is_number(step) and _is_number(kernel) and kernel > step + _MS_ABS_TOLERANCE:
        errors.append(
            f"'timing.kernel_ms' ({kernel}) exceeds 'timing.step_ms' ({step}) — "
            f"kernel time is a part of the iteration, not a superset of it"
        )


def _validate_coverage(data: Mapping[str, Any], errors: list[str], tolerance_pct: float) -> None:
    """Validate the coverage block and the time closure it asserts.

    Coverage is mandatory. Without it a campaign can read "the modeled
    parts have little gap" as "there is little headroom" while a large
    share of kernel time carries no ceiling at all.
    """
    block = data.get("coverage")
    if not isinstance(block, dict):
        errors.append(
            f"'coverage' must be a mapping with {list(_COVERAGE_KERNEL_FIELDS)}, "
            f"'modeled_pct' and 'non_kernel_ms' — a part the model does not "
            f"cover must be labelled as such, never left implicit; got {block!r}"
        )
        return
    for field in (*_COVERAGE_KERNEL_FIELDS, "non_kernel_ms"):
        _require_ms(block, field, "coverage", errors)
    modeled_pct = block.get("modeled_pct")
    if not _is_number(modeled_pct) or not 0 <= modeled_pct <= 100:
        errors.append(f"'coverage.modeled_pct' must be a number in [0, 100], got {modeled_pct!r}")
    residual = block.get("residual_ms")
    if residual is not None and not _is_number(residual):
        errors.append(
            f"'coverage.residual_ms' must be a number or omitted — what the "
            f"buckets miss is recorded, never absorbed into a neighbour; "
            f"got {residual!r}"
        )
    timing = data.get("timing")
    if not isinstance(timing, dict):
        return
    step, kernel = timing.get("step_ms"), timing.get("kernel_ms")
    values = [block.get(f) for f in _COVERAGE_KERNEL_FIELDS]
    if not all(_is_number(v) for v in values) or not _is_number(step) or not _is_number(kernel):
        return
    residual_ms = float(residual) if _is_number(residual) else 0.0
    kernel_sum = sum(float(v) for v in values)
    tolerance = tolerance_pct / 100.0 * float(step)
    if not _close(kernel_sum + residual_ms, float(kernel), tolerance):
        errors.append(
            f"'coverage' kernel buckets ({kernel_sum:.4f} ms + residual "
            f"{residual_ms:.4f}) do not reconcile to 'timing.kernel_ms' "
            f"({kernel}) within {tolerance_pct}% — analytic + empirical + "
            f"unmodeled must account for every kernel millisecond, with the "
            f"remainder recorded in 'residual_ms'"
        )
    total = kernel_sum + residual_ms + float(block.get("non_kernel_ms", 0.0))
    if not _close(total, float(step), tolerance):
        errors.append(
            f"'coverage' buckets plus 'non_kernel_ms' ({total:.4f} ms) do not "
            f"reconcile to 'timing.step_ms' ({step}) within {tolerance_pct}% — "
            f"host/scheduler time is outside the ceiling's model but inside "
            f"the iteration, and must never be silently treated as zero"
        )


# ---------------------------------------------------------------- shape: parts


def _normalize_at(part: dict[str, Any], where: str, errors: list[str]) -> None:
    """Validate — and key-normalize in place — a part's per-concurrency rows.

    Only an ``analytic`` part has a ceiling. An ``empirical`` or
    ``unmodeled`` part must leave ``sol_ms`` and ``gap_ms`` **null**:
    writing 0 would claim its whole measured time as headroom, and
    writing ``measured_ms`` would claim it has none. Both are assertions
    the campaign has no evidence for, and the honest answer — "this time
    is counted but not bounded" — is what keeps `no modeled gap` from
    being read as `no headroom`.
    """
    analytic = part.get("source") == "analytic"
    block = part.get("at")
    if not isinstance(block, dict) or not block:
        errors.append(
            f"'{where}.at' must be a non-empty mapping keyed by concurrency — "
            f"a part's gap is not concurrency-invariant, so a single number "
            f"cannot rank it at every scored point; got {block!r}"
        )
        return
    normalized: dict[int, Any] = {}
    for key, row in block.items():
        point = key
        if isinstance(point, str) and point.strip().isdigit():
            point = int(point.strip())
        if isinstance(point, bool) or not isinstance(point, int) or point < 1:
            errors.append(f"'{where}.at' keys must be integer concurrencies, got {key!r}")
            continue
        if not isinstance(row, dict):
            errors.append(f"'{where}.at[{point}]' must be a mapping, got {row!r}")
            continue
        if not _is_number(row.get("measured_ms")):
            errors.append(
                f"'{where}.at[{point}].measured_ms' must be a number, "
                f"got {row.get('measured_ms')!r}"
            )
        for field in ("sol_ms", "gap_ms"):
            value = row.get(field)
            if analytic and not _is_number(value):
                errors.append(f"'{where}.at[{point}].{field}' must be a number, got {value!r}")
            elif not analytic and value is not None:
                errors.append(
                    f"'{where}.at[{point}].{field}' must be null for a "
                    f"'{part.get('source')}' part — it carries no analytic "
                    f"ceiling, and inventing one would state headroom the "
                    f"campaign cannot support; got {value!r}"
                )
        normalized[point] = row
    if normalized:
        part["at"] = dict(sorted(normalized.items()))


def _validate_partition(part: Mapping[str, Any], where: str, errors: list[str]) -> None:
    """Validate the four buckets an analytic part's gap is split into.

    Only an analytic part owes one: a part with no ceiling has no gap to
    partition, and a partition over a gap that does not exist is a
    number with nothing behind it.
    """
    block = part.get("partition")
    if part.get("source") != "analytic":
        if block is not None:
            errors.append(
                f"'{where}.partition' must be omitted for a "
                f"'{part.get('source')}' part — with no ceiling there is no "
                f"gap to split, so its whole measured time is unbounded rather "
                f"than unexplained; got {block!r}"
            )
        return
    if not isinstance(block, dict):
        errors.append(
            f"'{where}.partition' must be a mapping with {list(_PARTITION_FIELDS)} — "
            f"every millisecond of the gap is closed, attributed, open, or "
            f"explicitly unexplained; got {block!r}"
        )
        return
    for field in ("closed_ms", "attributed_ms", "open_ms"):
        _require_ms(block, field, f"{where}.partition", errors)
    # `unexplained_ms` is the remainder, and the one bucket that may go
    # negative: a measurement below the ceiling is a model defect whose
    # residual has to land somewhere until the adjudication moves it.
    if not _is_number(block.get("unexplained_ms")):
        errors.append(
            f"'{where}.partition.unexplained_ms' must be a number (the "
            f"remainder — negative when the measurement beat the ceiling), "
            f"got {block.get('unexplained_ms')!r}"
        )


def _validate_disposition(entry: Any, where: str, errors: list[str]) -> None:
    """Validate one spent lever against a part.

    A disposition records that *one lever* is closed, with its mechanism
    and its citation. It never sets ``attributed_ms`` — that is a
    part-level claim needing its own basis.
    """
    if not isinstance(entry, dict):
        errors.append(f"'{where}' must be a mapping, got {type(entry).__name__}")
        return
    _require_str(entry, "item", where, errors)
    _require_str(entry, "lever", where, errors)
    _require_str(entry, "note", where, errors)
    _require_str(entry, "evidence", where, errors)
    _require_enum(entry, "gap_implication", GAP_IMPLICATIONS, where, errors)
    _require_enum(entry, "outcome", DISPOSITION_OUTCOMES, where, errors)
    round_no = entry.get("round")
    if isinstance(round_no, bool) or not isinstance(round_no, int) or round_no < 1:
        errors.append(f"'{where}.round' must be an integer >= 1, got {round_no!r}")
    directive = entry.get("directive")
    if directive is not None and not _nonempty_str(directive):
        errors.append(
            f"'{where}.directive' must be a non-empty string or omitted, got {directive!r}"
        )


def _validate_attribution(part: Mapping[str, Any], where: str, errors: list[str]) -> None:
    """Validate the only field that permanently retires headroom."""
    block = part.get("attribution")
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append(f"'{where}.attribution' must be a mapping or omitted, got {block!r}")
        return
    attributed = block.get("attributed_ms", 0.0)
    if not _is_number(attributed) or attributed < 0:
        errors.append(
            f"'{where}.attribution.attributed_ms' must be a number >= 0, got {attributed!r}"
        )
        return
    if attributed <= 0:
        return
    _require_enum(block, "basis", ATTRIBUTION_BASES, f"{where}.attribution", errors)
    if not _nonempty_str(block.get("note")):
        errors.append(
            f"'{where}.attribution.note' must be a non-empty string when "
            f"attributed_ms > 0 — declaring a gap unreachable forecloses it "
            f"for the rest of the campaign, so the reason is mandatory"
        )


def _validate_target_delta(target: Mapping[str, Any], where: str, errors: list[str]) -> None:
    delta = target.get("delta")
    if not isinstance(delta, list) or not delta:
        errors.append(
            f"'{where}.delta' must be a non-empty list of "
            f"{{cause, ms[, evidence]}} rows accounting for measured - target"
        )
        return
    seen_unattributed = 0
    for index, row in enumerate(delta):
        row_where = f"{where}.delta[{index}]"
        if not isinstance(row, dict):
            errors.append(f"'{row_where}' must be a mapping, got {type(row).__name__}")
            continue
        cause = row.get("cause")
        if not _nonempty_str(cause):
            errors.append(f"'{row_where}.cause' must be a non-empty string, got {cause!r}")
            continue
        if not _is_number(row.get("ms")):
            errors.append(f"'{row_where}.ms' must be a number, got {row.get('ms')!r}")
        if cause == UNATTRIBUTED_DELTA_CAUSE:
            seen_unattributed += 1
            continue
        if not _nonempty_str(row.get("evidence")):
            errors.append(
                f"'{row_where}.evidence' must be a non-empty string — every "
                f"named delta cause is observable (a trace, a kernel count, a "
                f"launch count), never a guess"
            )
    if seen_unattributed != 1:
        errors.append(
            f"'{where}.delta' must carry exactly one "
            f"'cause: {UNATTRIBUTED_DELTA_CAUSE}' row (got {seen_unattributed}) — "
            f"requiring the named causes alone to close the books would force "
            f"inventing a cause to balance the arithmetic"
        )


def _validate_target(part: Mapping[str, Any], where: str, errors: list[str]) -> None:
    """Validate the target layer: what should be there, and how fast."""
    target = part.get("target")
    if target is None:
        return
    if not isinstance(target, dict):
        errors.append(f"'{where}.target' must be a mapping or omitted, got {target!r}")
        return
    _require_ms(target, "target_ms", f"{where}.target", errors)
    _require_enum(target, "basis", TARGET_BASES, f"{where}.target", errors)
    for field in ("structure", "today", "falsifier"):
        _require_str(target, field, f"{where}.target", errors)
    if _nonempty_str(target.get("falsifier")) and target.get("falsifier", "").strip() == "none":
        errors.append(
            f"'{where}.target.falsifier' must name the dependency or "
            f"constraint that would make the target unbuildable — a target "
            f"nobody can falsify is an essay"
        )
    basis = target.get("basis")
    if basis == "none-known":
        if target.get("delta") is not None:
            errors.append(
                f"'{where}.target.delta' must be absent when basis is "
                f"'none-known' — the whole gap is structural, so there is no "
                f"engineering delta to itemize"
            )
        return
    if basis in _REF_REQUIRED_TARGET_BASES and not _nonempty_str(target.get("basis_ref")):
        errors.append(
            f"'{where}.target.basis_ref' must be a non-empty, resolvable "
            f"reference when basis is {basis!r} — an existing or published "
            f"implementation must cite something a reader can open"
        )
    elif basis == "derived" and not _nonempty_str(target.get("basis_ref")):
        errors.append(
            f"'{where}.target.basis_ref' must name the recipe arithmetic the "
            f"target was derived from, so the tier stays commensurable with "
            f"sol_ms and a saving cannot be double-counted"
        )
    efficiency = target.get("achieved_efficiency")
    if not isinstance(efficiency, dict):
        errors.append(
            f"'{where}.target.achieved_efficiency' must be a "
            f"{{value, source}} mapping — an efficiency with no named "
            f"measured source is the single easiest way to fabricate a "
            f"target; got {efficiency!r}"
        )
    else:
        value = efficiency.get("value")
        if not _is_number(value) or not 0 < value <= 1:
            errors.append(
                f"'{where}.target.achieved_efficiency.value' must be a number "
                f"in (0, 1] — a demonstrated fraction of peak, not a wish; "
                f"got {value!r}"
            )
        _require_str(efficiency, "source", f"{where}.target.achieved_efficiency", errors)
    _validate_target_delta(target, f"{where}.target", errors)


def _validate_part(part: Any, index: int, seen: set[str], errors: list[str]) -> None:
    where = f"parts[{index}]"
    if not isinstance(part, dict):
        errors.append(f"'{where}' must be a mapping, got {type(part).__name__}")
        return
    part_id = part.get("id")
    if not _nonempty_str(part_id):
        errors.append(f"'{where}.id' must be a non-empty string, got {part_id!r}")
    elif part_id in seen:
        errors.append(f"'{where}.id' duplicates {part_id!r} — part ids must be unique")
    else:
        seen.add(part_id)
    _require_enum(part, "source", PART_SOURCES, where, errors)
    _normalize_at(part, where, errors)
    sensitivity = part.get("sensitivity")
    if sensitivity is not None and sensitivity not in SENSITIVITIES:
        errors.append(
            f"'{where}.sensitivity' must be one of {list(SENSITIVITIES)} or null "
            f"(null = only one point was captured, so the part cannot be ranked "
            f"per scored point), got {sensitivity!r}"
        )
    kernels = part.get("kernels")
    if kernels is not None and (
        not isinstance(kernels, list) or not all(_nonempty_str(k) for k in kernels)
    ):
        errors.append(
            f"'{where}.kernels' must be a list of non-empty kernel-ledger row "
            f"labels (or omitted), got {kernels!r}"
        )
    _validate_partition(part, where, errors)
    dispositions = part.get("dispositions")
    if dispositions is not None:
        if not isinstance(dispositions, list):
            errors.append(f"'{where}.dispositions' must be a list, got {dispositions!r}")
        else:
            for d_index, entry in enumerate(dispositions):
                _validate_disposition(entry, f"{where}.dispositions[{d_index}]", errors)
    history = part.get("history")
    if history is not None:
        if not isinstance(history, list):
            errors.append(f"'{where}.history' must be a list, got {history!r}")
        else:
            for h_index, entry in enumerate(history):
                h_where = f"{where}.history[{h_index}]"
                if not isinstance(entry, dict):
                    errors.append(f"'{h_where}' must be a mapping, got {type(entry).__name__}")
                    continue
                round_no = entry.get("round")
                if isinstance(round_no, bool) or not isinstance(round_no, int) or round_no < 1:
                    errors.append(f"'{h_where}.round' must be an integer >= 1, got {round_no!r}")
                for field in ("measured_ms", "gap_ms"):
                    if not _is_number(entry.get(field)):
                        errors.append(f"'{h_where}.{field}' must be a number")
    _validate_attribution(part, where, errors)
    _validate_target(part, where, errors)


# ----------------------------------------------------------- shape: revisions


def _validate_revision(
    entry: Any,
    where: str,
    *,
    value_key: str,
    causes: Sequence[str],
    extra_str_fields: Sequence[str],
    errors: list[str],
) -> None:
    """Validate one revision entry under the three-field completeness rule.

    ``cause`` from the closed enum, a non-empty ``detail``, and a
    resolving ``evidence``. A revision missing any of the three is
    rejected — the round re-opens the adjudication rather than silently
    keeping the new number.
    """
    if not isinstance(entry, dict):
        errors.append(f"'{where}' must be a mapping, got {type(entry).__name__}")
        return
    round_no = entry.get("round")
    if isinstance(round_no, bool) or not isinstance(round_no, int) or round_no < 1:
        errors.append(f"'{where}.round' must be an integer >= 1, got {round_no!r}")
    _require_str(entry, "part", where, errors)
    _require_str(entry, "detail", where, errors)
    _require_str(entry, "evidence", where, errors)
    for field in extra_str_fields:
        _require_str(entry, field, where, errors)
    move = entry.get(value_key)
    if not isinstance(move, dict) or not all(_is_number(move.get(k)) for k in ("from", "to")):
        errors.append(
            f"'{where}.{value_key}' must be a {{from: <float>, to: <float>}} "
            f"mapping recording the move, got {move!r}"
        )
    cause = entry.get("cause")
    if causes is DISMISSAL_TAGS:
        # The target-revision vocabulary is the workflow's existing
        # dismissal vocabulary, whose entries are `tag` or `tag: <detail>`.
        if not _nonempty_str(cause) or dismissal_tag(cause) not in DISMISSAL_TAGS:
            errors.append(
                f"'{where}.cause' must lead with one of the existing "
                f"dismissal tags {list(DISMISSAL_TAGS)} (optionally followed "
                f"by ': <detail>'), got {cause!r}"
            )
    elif cause not in causes:
        errors.append(f"'{where}.cause' must be one of {list(causes)}, got {cause!r}")
    detail = entry.get("detail")
    if _nonempty_str(detail) and _nonempty_str(cause) and detail.strip() == str(cause).strip():
        errors.append(
            f"'{where}.detail' restates '{where}.cause' — the symptom is never "
            f"the fact; name the specific defect and where it can be re-checked"
        )


def _validate_revisions(data: Mapping[str, Any], errors: list[str]) -> None:
    specs = (
        ("model_revisions", "sol_ms", MODEL_REVISION_CAUSES, ()),
        ("measurement_revisions", "measured_ms", MEASUREMENT_REVISION_CAUSES, ()),
        ("target_revisions", "target_ms", DISMISSAL_TAGS, ("found_by",)),
    )
    for key, value_key, causes, extra in specs:
        entries = data.get(key)
        if entries is None:
            continue
        if not isinstance(entries, list):
            errors.append(f"'{key}' must be a list, got {type(entries).__name__}")
            continue
        for index, entry in enumerate(entries):
            _validate_revision(
                entry,
                f"{key}[{index}]",
                value_key=value_key,
                causes=causes,
                extra_str_fields=extra,
                errors=errors,
            )


def _validate_lifecycle(data: Mapping[str, Any], errors: list[str]) -> None:
    entries = data.get("part_lifecycle")
    if entries is None:
        return
    if not isinstance(entries, list):
        errors.append(f"'part_lifecycle' must be a list, got {type(entries).__name__}")
        return
    for index, entry in enumerate(entries):
        where = f"part_lifecycle[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"'{where}' must be a mapping, got {type(entry).__name__}")
            continue
        round_no = entry.get("round")
        if isinstance(round_no, bool) or not isinstance(round_no, int) or round_no < 1:
            errors.append(f"'{where}.round' must be an integer >= 1, got {round_no!r}")
        _require_str(entry, "part", where, errors)
        _require_str(entry, "by", where, errors)
        _require_enum(entry, "event", PART_LIFECYCLE_EVENTS, where, errors)
        for field in ("measured_ms", "sol_ms"):
            _require_ms(entry, field, where, errors)


# ------------------------------------------------------------------ public API


def load_ledger(path: str | Path, tolerance_pct: float = DEFAULT_TOLERANCE_PCT) -> dict[str, Any]:
    """Parse ``path`` as YAML and validate the headroom-ledger schema.

    Shape and internal arithmetic only — the roadmap join, the SOL
    correlation and the previous round's ledger need context the file
    does not carry; run :func:`cross_validate` for those. Raises
    :class:`HeadroomLedgerError` with **every** detected problem batched
    into a single message.
    """
    ledger_path = Path(path)
    if not ledger_path.is_file():
        raise HeadroomLedgerError(f"headroom ledger not found: {ledger_path}")

    try:
        data = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HeadroomLedgerError(f"{ledger_path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise HeadroomLedgerError(
            f"{ledger_path} must be a YAML mapping at the top level, got {type(data).__name__}"
        )

    errors: list[str] = []

    if data.get("version") != HEADROOM_LEDGER_VERSION:
        errors.append(f"'version' must be {HEADROOM_LEDGER_VERSION}, got {data.get('version')!r}")

    # Unknown keys are rejected rather than ignored: a silently dropped
    # block is how a campaign ends up running under settings nobody
    # applied, and this ledger's blocks all change what gets built.
    unknown = sorted(str(key) for key in data if key not in _TOP_LEVEL_KEYS)
    if unknown:
        errors.append(
            f"unknown top-level key(s) {', '.join(repr(k) for k in unknown)} — "
            f"valid keys are {', '.join(repr(k) for k in _TOP_LEVEL_KEYS)}"
        )

    _validate_operating_point(data, errors)
    _validate_timing(data, errors)
    _validate_coverage(data, errors, tolerance_pct)

    parts = data.get("parts")
    if not isinstance(parts, list) or not parts:
        errors.append(f"'parts' must be a non-empty list, got {type(parts).__name__}")
        parts = []
    seen: set[str] = set()
    for index, part in enumerate(parts):
        _validate_part(part, index, seen, errors)

    _validate_lifecycle(data, errors)
    _validate_revisions(data, errors)

    if not errors:
        errors.extend(_arithmetic_problems(data, tolerance_pct))

    if errors:
        bullet = "\n  - "
        raise HeadroomLedgerError(
            f"{ledger_path} failed headroom-ledger schema validation:{bullet}{bullet.join(errors)}"
        )
    return data


def primary_concurrency(ledger: Mapping[str, Any]) -> int | None:
    """The point ``partition`` / ``target`` / ``history`` are stated at.

    The highest bracketing concurrency: the campaign's primary ranking
    point. Returns ``None`` when the operating point is unreadable.
    """
    points = (ledger.get("operating_point") or {}).get("concurrency")
    if isinstance(points, list) and points and isinstance(points[-1], int):
        return int(points[-1])
    return None


def part_at(part: Mapping[str, Any], concurrency: int | None) -> Mapping[str, Any] | None:
    """The part's ``at`` row for ``concurrency``, or its highest point."""
    rows = part.get("at")
    if not isinstance(rows, dict) or not rows:
        return None
    if concurrency is not None and concurrency in rows:
        row = rows[concurrency]
        return row if isinstance(row, Mapping) else None
    row = rows[max(rows)]
    return row if isinstance(row, Mapping) else None


def _arithmetic_problems(data: Mapping[str, Any], tolerance_pct: float) -> list[str]:
    """Per-part arithmetic the file can check against itself."""
    problems: list[str] = []
    primary = primary_concurrency(data)
    revised_parts = {
        entry.get("part")
        for entry in data.get("model_revisions") or []
        if isinstance(entry, Mapping)
    }
    for index, part in enumerate(data.get("parts") or []):
        where = f"parts[{index}]"
        rows = part.get("at")
        if not isinstance(rows, dict) or part.get("source") != "analytic":
            continue
        for point, row in rows.items():
            if not isinstance(row, Mapping):
                continue
            measured, sol, gap = (row.get(f) for f in _AT_FIELDS)
            if not all(_is_number(v) for v in (measured, sol, gap)):
                continue
            if not _close(float(measured) - float(sol), float(gap), _MS_ABS_TOLERANCE):
                problems.append(
                    f"'{where}.at[{point}]' gap_ms ({gap}) != measured_ms "
                    f"({measured}) - sol_ms ({sol})"
                )
            if float(gap) < -_MS_ABS_TOLERANCE and part.get("id") not in revised_parts:
                problems.append(
                    f"'{where}.at[{point}]' has a negative gap_ms ({gap}) — "
                    f"measured beat the ceiling, which is a model defect: add a "
                    f"'model_revisions' entry for {part.get('id')!r} rather than "
                    f"leaving a ceiling the measurement already disproved"
                )
        row = part_at(part, primary)
        partition = part.get("partition")
        if isinstance(row, Mapping) and isinstance(partition, Mapping):
            gap = row.get("gap_ms")
            values = [partition.get(f) for f in _PARTITION_FIELDS]
            if _is_number(gap) and all(_is_number(v) for v in values):
                total = sum(float(v) for v in values)
                tolerance = tolerance_pct / 100.0 * abs(float(gap))
                if not _close(total, float(gap), tolerance):
                    problems.append(
                        f"'{where}.partition' sums to {total:.4f} ms but the "
                        f"gap at the primary point is {gap} — every millisecond "
                        f"is closed, attributed, open, or unexplained"
                    )
            attribution = part.get("attribution")
            if isinstance(attribution, Mapping) and _is_number(partition.get("attributed_ms")):
                claimed = attribution.get("attributed_ms", 0.0)
                if _is_number(claimed) and not _close(
                    float(claimed), float(partition["attributed_ms"]), _MS_ABS_TOLERANCE
                ):
                    problems.append(
                        f"'{where}.partition.attributed_ms' "
                        f"({partition['attributed_ms']}) disagrees with "
                        f"'{where}.attribution.attributed_ms' ({claimed})"
                    )
        problems.extend(_target_problems(part, where, row, tolerance_pct))
    return problems


def _target_problems(
    part: Mapping[str, Any],
    where: str,
    row: Mapping[str, Any] | None,
    tolerance_pct: float,
) -> list[str]:
    """The three-tier invariant and the delta's own books."""
    target = part.get("target")
    if not isinstance(target, Mapping) or not isinstance(row, Mapping):
        return []
    target_ms, measured, sol = target.get("target_ms"), row.get("measured_ms"), row.get("sol_ms")
    if not all(_is_number(v) for v in (target_ms, measured, sol)):
        return []
    problems: list[str] = []
    if float(target_ms) < float(sol) - _MS_ABS_TOLERANCE:
        problems.append(
            f"'{where}.target.target_ms' ({target_ms}) is below sol_ms ({sol}) — "
            f"a target under the floor is a model defect, not a better target: "
            f"open a 'model_revisions' adjudication instead"
        )
    if float(target_ms) > float(measured) + _MS_ABS_TOLERANCE:
        problems.append(
            f"'{where}.target.target_ms' ({target_ms}) is above measured_ms "
            f"({measured}) — a target the implementation already beats was "
            f"wrong too; record a downward 'target_revisions' entry"
        )
    if target.get("basis") == "none-known":
        if not _close(float(target_ms), float(measured), _MS_ABS_TOLERANCE):
            problems.append(
                f"'{where}.target.target_ms' ({target_ms}) must equal measured_ms "
                f"({measured}) when basis is 'none-known' — with no named "
                f"implementation the whole gap is structural"
            )
        return problems
    delta = target.get("delta")
    if isinstance(delta, list) and all(
        isinstance(r, Mapping) and _is_number(r.get("ms")) for r in delta
    ):
        total = sum(float(r["ms"]) for r in delta)
        expected = float(measured) - float(target_ms)
        tolerance = tolerance_pct / 100.0 * max(abs(expected), _MS_ABS_TOLERANCE)
        if not _close(total, expected, tolerance):
            problems.append(
                f"'{where}.target.delta' sums to {total:.4f} ms but "
                f"measured_ms - target_ms is {expected:.4f} — the named causes "
                f"plus the explicit 'unattributed' remainder must close the books"
            )
    return problems


# ------------------------------------------------------------- cross-validate


def load_context(analysis_dir: str | Path) -> LedgerContext:
    """Best-effort read of the round artifacts the ledger is checked against.

    A missing or malformed artifact yields ``None`` for that field rather
    than raising: the ledger is not at fault for a pipeline that could
    not run, and the checks needing it are skipped with the absence
    reported by the caller.
    """
    directory = Path(analysis_dir)

    def _json(name: str) -> dict[str, Any] | None:
        try:
            data = json.loads((directory / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    try:
        kernels = yaml.safe_load((directory / "kernel_ledger.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        kernels = None
    return LedgerContext(
        sol=_json("sol.json"),
        regions=_json("regions.json"),
        kernels=kernels if isinstance(kernels, dict) else None,
    )


def _sol_regions(sol: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not isinstance(sol, Mapping):
        return {}
    return {
        str(row["region"]): row
        for row in sol.get("per_op") or []
        if isinstance(row, Mapping) and row.get("region")
    }


def _measured_regions(regions: Mapping[str, Any] | None) -> dict[str, float]:
    """Raw per-region kernel time from ``regions.json``.

    Deliberately **not** ``sol.json``'s measured column: the calculator
    substitutes ``exposed_ms`` for communication rows, so the raw kernel
    time is the right closure target for the kernel join.
    """
    if not isinstance(regions, Mapping):
        return {}
    out: dict[str, float] = {}
    for row in regions.get("regions") or []:
        if isinstance(row, Mapping) and row.get("region") and _is_number(row.get("measured_ms")):
            out[str(row["region"])] = float(row["measured_ms"])
    return out


def _kernel_shares(kernels: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(kernels, Mapping):
        return {}
    return {
        str(row["kernel"]): float(row["share_pct"])
        for row in kernels.get("kernels") or []
        if isinstance(row, Mapping) and row.get("kernel") and _is_number(row.get("share_pct"))
    }


def _check_join_closure(
    ledger: Mapping[str, Any], context: LedgerContext, tolerance_pct: float
) -> list[str]:
    """The kernel→part join, validated as the arithmetic identity it is.

    ``sum(share_pct) x kernel_ms`` must reproduce the part's raw
    ``measured_ms`` in ``regions.json``. A failure reports the residual in
    **both** milliseconds and share_pct, because that is usually enough to
    name the kernel the join is missing.
    """
    shares = _kernel_shares(context.kernels)
    measured = _measured_regions(context.regions)
    kernel_ms = (ledger.get("timing") or {}).get("kernel_ms")
    if not shares or not measured or not _is_number(kernel_ms):
        return []
    problems: list[str] = []
    for index, part in enumerate(ledger.get("parts") or []):
        rows = part.get("kernels")
        part_id = str(part.get("id"))
        if not isinstance(rows, list) or not rows or part_id not in measured:
            continue
        missing = [k for k in rows if k not in shares]
        if missing:
            problems.append(
                f"'parts[{index}].kernels' names {missing} which are not rows "
                f"in this round's kernel_ledger.yaml — the join must be made "
                f"against the enumerated ledger, not from memory"
            )
            continue
        joined = sum(shares[k] for k in rows) / 100.0 * float(kernel_ms)
        expected = measured[part_id]
        tolerance = tolerance_pct / 100.0 * expected
        if not _close(joined, expected, tolerance):
            residual = expected - joined
            problems.append(
                f"'parts[{index}].kernels' joins to {joined:.4f} ms but "
                f"regions.json measures {part_id!r} at {expected:.4f} ms — "
                f"residual {residual:+.4f} ms = {residual / float(kernel_ms) * 100:+.3f}% "
                f"of kernel_ms; look for a ledger row with that share"
            )
    return problems


def _check_kernel_claims(
    ledger: Mapping[str, Any], context: LedgerContext, min_share_pct: float
) -> list[str]:
    """No row claimed twice, and no hot row silently unclaimed."""
    shares = _kernel_shares(context.kernels)
    if not shares:
        return []
    problems: list[str] = []
    owner: dict[str, str] = {}
    for part in ledger.get("parts") or []:
        part_id = str(part.get("id"))
        for kernel in part.get("kernels") or []:
            if kernel in owner:
                problems.append(
                    f"kernel row {kernel!r} is claimed by both {owner[kernel]!r} "
                    f"and {part_id!r} — a row belongs to exactly one part, or "
                    f"its time is counted twice"
                )
            else:
                owner[str(kernel)] = part_id
    empirical = {
        str(part.get("id"))
        for part in ledger.get("parts") or []
        if part.get("source") in ("empirical", "unmodeled")
    }
    unclaimed = sorted(
        k
        for k, share in shares.items()
        if k not in owner and share >= min_share_pct and k not in empirical
    )
    if unclaimed:
        problems.append(
            f"kernel rows {unclaimed} are at/above {min_share_pct}% and belong "
            f"to no part — an unclaimed row above the bar becomes an "
            f"'empirical' part rather than silently vanishing from the accounting"
        )
    return problems


def _dismissed_questions(kernels: Mapping[str, Any] | None, kernel: str) -> bool:
    for row in (kernels or {}).get("kernels") or []:
        if isinstance(row, Mapping) and row.get("kernel") == kernel:
            return all(
                isinstance(row.get(q), Mapping) and row[q].get("disposition") == "dismissed"
                for q in QUESTIONS
            )
    return False


def _check_attribution_basis(ledger: Mapping[str, Any], context: LedgerContext) -> list[str]:
    """The basis behind a foreclosure must actually hold.

    A failed item closes a *lever*, never a part. Retiring a gap needs
    either an exhaustiveness proof (every kernel dismissed on all four
    questions) or convergence from two or more *distinct* levers. One
    failure is an anecdote.
    """
    problems: list[str] = []
    for index, part in enumerate(ledger.get("parts") or []):
        attribution = part.get("attribution")
        if not isinstance(attribution, Mapping):
            continue
        attributed = attribution.get("attributed_ms", 0.0)
        if not _is_number(attributed) or attributed <= 0:
            continue
        basis = attribution.get("basis")
        where = f"parts[{index}].attribution"
        if basis == "kernel-ledger-exhaustive":
            rows = part.get("kernels") or []
            if not rows:
                problems.append(
                    f"'{where}.basis' is 'kernel-ledger-exhaustive' but "
                    f"'parts[{index}].kernels' is empty — there is nothing to "
                    f"have been exhaustive about"
                )
                continue
            unproven = [k for k in rows if not _dismissed_questions(context.kernels, k)]
            if unproven:
                problems.append(
                    f"'{where}.basis' is 'kernel-ledger-exhaustive' but {unproven} "
                    f"do not have all four kernel_ledger.yaml questions "
                    f"({' / '.join(QUESTIONS)}) dispositioned 'dismissed' — the "
                    f"exhaustiveness proof does not hold"
                )
        elif basis == "convergent-levers":
            levers = {
                str(d.get("lever"))
                for d in part.get("dispositions") or []
                if isinstance(d, Mapping)
                and d.get("gap_implication") in CONVERGENT_GAP_IMPLICATIONS
                and _nonempty_str(d.get("lever"))
            }
            if len(levers) < 2:
                problems.append(
                    f"'{where}.basis' is 'convergent-levers' but only "
                    f"{len(levers)} distinct lever(s) converge "
                    f"({sorted(levers)}) — a basis needs two or more, and only "
                    f"{list(CONVERGENT_GAP_IMPLICATIONS)} count (a "
                    f"'change-not-live' or 'blocked-by-constraint' verdict never "
                    f"tested the mechanism against the part)"
                )
    return problems


def _item_index(roadmap: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["id"]): item
        for item in roadmap.get("items") or []
        if isinstance(item, Mapping) and item.get("id")
    }


def _check_roadmap_join(ledger: Mapping[str, Any], roadmap: Mapping[str, Any]) -> list[str]:
    """Dispositions name real items; open/closed time is backed by one."""
    problems: list[str] = []
    items = _item_index(roadmap)
    parts_by_item: dict[str, set[str]] = {}
    for item_id, item in items.items():
        for part_id in item.get("parts") or []:
            parts_by_item.setdefault(str(part_id), set()).add(item_id)
    for index, part in enumerate(ledger.get("parts") or []):
        where = f"parts[{index}]"
        part_id = str(part.get("id"))
        for d_index, entry in enumerate(part.get("dispositions") or []):
            if not isinstance(entry, Mapping):
                continue
            item_id = str(entry.get("item"))
            if item_id not in items:
                problems.append(
                    f"'{where}.dispositions[{d_index}].item' ({item_id!r}) does "
                    f"not match any roadmap item id — a disposition records what "
                    f"a real attempt proved"
                )
        partition = part.get("partition")
        if not isinstance(partition, Mapping):
            continue
        claimants = parts_by_item.get(part_id, set())
        live = {i for i in claimants if items[i].get("status") in ("pending", "in_progress")}
        accepted = {i for i in claimants if items[i].get("status") == "accepted"}
        if _is_number(partition.get("open_ms")) and partition["open_ms"] > 0 and not live:
            problems.append(
                f"'{where}.partition.open_ms' is {partition['open_ms']} but no "
                f"pending or in-progress roadmap item names {part_id!r} in its "
                f"'parts' — open time is a claim someone is about to test"
            )
        if _is_number(partition.get("closed_ms")) and partition["closed_ms"] > 0 and not accepted:
            problems.append(
                f"'{where}.partition.closed_ms' is {partition['closed_ms']} but "
                f"no accepted roadmap item names {part_id!r} in its 'parts'"
            )
    return problems


def _check_against_previous(
    ledger: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> list[str]:
    """A number that moved needs the revision entry that moved it."""
    if not isinstance(previous, Mapping):
        return []
    problems: list[str] = []
    revised = {
        key: {
            str(entry.get("part")) for entry in ledger.get(key) or [] if isinstance(entry, Mapping)
        }
        for key in ("model_revisions", "measurement_revisions", "target_revisions")
    }
    previous_parts = {
        str(part.get("id")): part
        for part in previous.get("parts") or []
        if isinstance(part, Mapping) and part.get("id")
    }
    current_ids = {str(part.get("id")) for part in ledger.get("parts") or []}
    lifecycle = {
        str(entry.get("part"))
        for entry in ledger.get("part_lifecycle") or []
        if isinstance(entry, Mapping)
    }
    for part_id in sorted(set(previous_parts) - current_ids - lifecycle):
        problems.append(
            f"part {part_id!r} was in the previous ledger and is gone without a "
            f"'part_lifecycle' entry — a part that disappears took its measured "
            f"time and its ceiling with it, and that has to be recorded"
        )
    for index, part in enumerate(ledger.get("parts") or []):
        part_id = str(part.get("id"))
        before = previous_parts.get(part_id)
        if not isinstance(before, Mapping):
            continue
        for point, row in (part.get("at") or {}).items():
            old = (before.get("at") or {}).get(point)
            if not isinstance(row, Mapping) or not isinstance(old, Mapping):
                continue
            for field, key in (("sol_ms", "model_revisions"),):
                if (
                    _is_number(row.get(field))
                    and _is_number(old.get(field))
                    and not _close(float(row[field]), float(old[field]), _MS_ABS_TOLERANCE)
                    and part_id not in revised[key]
                ):
                    problems.append(
                        f"'parts[{index}].at[{point}].{field}' moved "
                        f"{old[field]} -> {row[field]} with no '{key}' entry for "
                        f"{part_id!r} — the ceiling is a bound, not a "
                        f"description: it moves only when the model was wrong"
                    )
        old_target = (before.get("target") or {}).get("target_ms")
        new_target = (part.get("target") or {}).get("target_ms")
        if (
            _is_number(old_target)
            and _is_number(new_target)
            and not _close(float(old_target), float(new_target), _MS_ABS_TOLERANCE)
            and part_id not in revised["target_revisions"]
        ):
            problems.append(
                f"'parts[{index}].target.target_ms' moved {old_target} -> "
                f"{new_target} with no 'target_revisions' entry for {part_id!r} — "
                f"a target is a prediction, so a correction records what the "
                f"attempt actually found"
            )
    return problems


def cross_validate(
    ledger: Mapping[str, Any],
    *,
    roadmap: Mapping[str, Any],
    context: LedgerContext | None = None,
    previous: Mapping[str, Any] | None = None,
    focus_points: Sequence[int] | None = None,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
    min_share_pct: float = DEFAULT_MIN_SHARE_PCT,
) -> list[str]:
    """Context checks a shape-valid ledger still owes; returns the problems.

    Never raises and never mutates: the caller decides whether a problem
    warns or aborts the round.
    """
    context = context or LedgerContext()
    problems: list[str] = []

    regions = _sol_regions(context.sol)
    if regions:
        for index, part in enumerate(ledger.get("parts") or []):
            if part.get("source") != "analytic":
                continue
            part_id = str(part.get("id"))
            if part_id not in regions:
                problems.append(
                    f"'parts[{index}].id' ({part_id!r}) is declared 'analytic' but "
                    f"is not a region in this round's sol.json — an analytic part "
                    f"is one the projection actually bounds"
                )

    problems.extend(_check_join_closure(ledger, context, tolerance_pct))
    problems.extend(_check_kernel_claims(ledger, context, min_share_pct))
    problems.extend(_check_attribution_basis(ledger, context))
    problems.extend(_check_roadmap_join(ledger, roadmap))
    problems.extend(_check_against_previous(ledger, previous))
    problems.extend(_check_operating_point(ledger, focus_points))
    return problems


def _check_operating_point(
    ledger: Mapping[str, Any], focus_points: Sequence[int] | None
) -> list[str]:
    """The bracket must match the scored regime, and be a bracket at all."""
    points = (ledger.get("operating_point") or {}).get("concurrency")
    if not isinstance(points, list) or not points:
        return []
    problems: list[str] = []
    if focus_points:
        expected = sorted({min(focus_points), max(focus_points)})
        if list(points) != expected:
            problems.append(
                f"'operating_point.concurrency' is {list(points)} but the scored "
                f"regime brackets to {expected} — the ledger must be measured at "
                f"the lowest and highest scored concurrency, or it ranks parts "
                f"at points it never saw"
            )
    if len(points) < 2:
        problems.append(
            f"'operating_point.concurrency' has a single point ({list(points)}): "
            f"a part's sensitivity cannot be established from one endpoint, so "
            f"every part is ranked campaign-wide and any concurrency-localized "
            f"gain will be mis-ranked at the other points"
        )
        return problems
    for index, part in enumerate(ledger.get("parts") or []):
        rows = part.get("at")
        if isinstance(rows, dict) and not set(points) <= set(rows):
            problems.append(
                f"'parts[{index}].at' is missing {sorted(set(points) - set(rows))} — "
                f"every part is measured at both bracketing points"
            )
        elif part.get("sensitivity") not in SENSITIVITIES:
            problems.append(
                f"'parts[{index}].sensitivity' must be one of "
                f"{list(SENSITIVITIES)} when both bracketing points were "
                f"captured — whether the gap is flat or steep is the primary "
                f"output of measuring two points"
            )
    return problems


# ------------------------------------------------------- orchestrator mutators


def dump_ledger(data: Mapping[str, Any]) -> str:
    """Serialize a ledger mapping back to YAML text."""
    return yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True, default_flow_style=False)


def save_ledger(path: str | Path, data: Mapping[str, Any]) -> None:
    Path(path).write_text(dump_ledger(data), encoding="utf-8")


def _read_raw(path: str | Path) -> dict[str, Any]:
    """Load the ledger without validating — mutators must not gate on shape.

    A mutator runs after a batch closes, when the round's verdicts are the
    only durable record of what was learned. Refusing to record them
    because an analyzer-authored field elsewhere in the file is malformed
    would lose the fact to protect the schema.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HeadroomLedgerError(f"{path} must be a YAML mapping at the top level")
    return data


def append_dispositions(
    path: str | Path,
    *,
    round_no: int,
    item_id: str,
    outcome: str,
    gap_implication: str,
    lever: str,
    note: str,
    evidence: str,
    parts: Iterable[str],
    directive: str | None = None,
) -> list[str]:
    """Record one spent lever against each part the verdict bears on.

    Orchestrator-owned, and idempotent on ``(round_no, item_id)`` so a
    resume never double-appends. Returns the part ids actually written;
    an id the ledger does not carry is skipped rather than invented.
    """
    if outcome not in DISPOSITION_OUTCOMES:
        raise HeadroomLedgerError(
            f"invalid outcome {outcome!r}; expected one of {list(DISPOSITION_OUTCOMES)}"
        )
    if gap_implication not in GAP_IMPLICATIONS:
        raise HeadroomLedgerError(
            f"invalid gap_implication {gap_implication!r}; expected one of {list(GAP_IMPLICATIONS)}"
        )
    data = _read_raw(path)
    wanted = {str(p) for p in parts}
    written: list[str] = []
    for part in data.get("parts") or []:
        if not isinstance(part, dict) or str(part.get("id")) not in wanted:
            continue
        existing = part.setdefault("dispositions", [])
        if not isinstance(existing, list):
            existing = []
            part["dispositions"] = existing
        if any(
            isinstance(e, Mapping) and e.get("item") == item_id and e.get("round") == round_no
            for e in existing
        ):
            written.append(str(part["id"]))
            continue
        entry: dict[str, Any] = {
            "item": item_id,
            "round": round_no,
            "outcome": outcome,
            "gap_implication": gap_implication,
            "lever": lever,
            "note": note,
            "evidence": evidence,
        }
        if directive:
            entry["directive"] = directive
        existing.append(entry)
        written.append(str(part["id"]))
    if written:
        save_ledger(path, data)
    return written


def record_history(path: str | Path, round_no: int) -> None:
    """Snapshot every part's primary-point measurement into ``history``.

    Idempotent on ``round_no``: the round's row is written once, and a
    resume through the same batch close leaves it unchanged.
    """
    data = _read_raw(path)
    primary = primary_concurrency(data)
    changed = False
    for part in data.get("parts") or []:
        if not isinstance(part, dict):
            continue
        row = part_at(part, primary)
        if not isinstance(row, Mapping):
            continue
        history = part.setdefault("history", [])
        if not isinstance(history, list):
            history = []
            part["history"] = history
        if any(isinstance(e, Mapping) and e.get("round") == round_no for e in history):
            continue
        history.append(
            {
                "round": round_no,
                "measured_ms": row.get("measured_ms"),
                "gap_ms": row.get("gap_ms"),
            }
        )
        changed = True
    if changed:
        save_ledger(path, data)


# ------------------------------------------------------------- report helpers


def partition_totals(ledger: Mapping[str, Any], concurrency: int | None = None) -> dict[str, float]:
    """Campaign-level gap accounting, in absolute milliseconds.

    Absolute ms, never % of SOL: the denominator moves whenever an
    accepted item deletes a part, so the ratio changes for reasons
    unrelated to whether the deployment got faster.
    """
    point = concurrency if concurrency is not None else primary_concurrency(ledger)
    totals = {field: 0.0 for field in _PARTITION_FIELDS}
    totals["gap_ms"] = 0.0
    totals["unbounded_ms"] = 0.0
    for part in ledger.get("parts") or []:
        if not isinstance(part, Mapping):
            continue
        partition = part.get("partition")
        if isinstance(partition, Mapping):
            for field in _PARTITION_FIELDS:
                if _is_number(partition.get(field)):
                    totals[field] += float(partition[field])
        row = part_at(part, point)
        if isinstance(row, Mapping) and _is_number(row.get("gap_ms")):
            totals["gap_ms"] += float(row["gap_ms"])
        elif isinstance(row, Mapping) and _is_number(row.get("measured_ms")):
            # Time that is counted but carries no ceiling. Reported on its
            # own axis so it can never be mistaken for either headroom or
            # its absence.
            totals["unbounded_ms"] = totals.get("unbounded_ms", 0.0) + float(row["measured_ms"])
    # Time an accepted item deleted outright: the part is gone, measured
    # and ceiling together, so it is a campaign credit rather than a
    # partition bucket on a part that no longer exists.
    eliminated = 0.0
    for entry in ledger.get("part_lifecycle") or []:
        if not isinstance(entry, Mapping) or entry.get("event") != "eliminated":
            continue
        if _is_number(entry.get("measured_ms")) and _is_number(entry.get("sol_ms")):
            eliminated += float(entry["measured_ms"]) - float(entry["sol_ms"])
    totals["eliminated_ms"] = eliminated
    return totals


def unexplained_ranking(
    ledger: Mapping[str, Any], concurrency: int | None = None
) -> list[tuple[str, float]]:
    """Parts ordered by unexplained milliseconds, descending.

    This is the campaign's work queue. A part with two spent levers and
    no holding attribution basis stays on it, carrying its accumulating
    list of what has already been tried.
    """
    ranked: list[tuple[str, float]] = []
    for part in ledger.get("parts") or []:
        if not isinstance(part, Mapping):
            continue
        partition = part.get("partition")
        if not isinstance(partition, Mapping) or not _is_number(partition.get("unexplained_ms")):
            continue
        ranked.append((str(part.get("id")), float(partition["unexplained_ms"])))
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return ranked


def engineering_ranking(
    ledger: Mapping[str, Any], concurrency: int | None = None
) -> list[tuple[str, float]]:
    """Parts ordered by the engineering gap (``measured - target``).

    What the roadmap is ranked on when the target layer is live: the
    distance to a named, buildable implementation rather than to physics
    no kernel achieves. Parts with no target contribute nothing here —
    their headroom is real but unplanned, which is what
    :func:`unexplained_ranking` surfaces.
    """
    point = concurrency if concurrency is not None else primary_concurrency(ledger)
    ranked: list[tuple[str, float]] = []
    for part in ledger.get("parts") or []:
        if not isinstance(part, Mapping):
            continue
        target = part.get("target")
        row = part_at(part, point)
        if not isinstance(target, Mapping) or not isinstance(row, Mapping):
            continue
        if not _is_number(target.get("target_ms")) or not _is_number(row.get("measured_ms")):
            continue
        ranked.append((str(part.get("id")), float(row["measured_ms"]) - float(target["target_ms"])))
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return ranked
