"""Unified per-round kernel coverage and theoretical performance ledger.

The analyzer owns ``kernel_ledger.yaml``. Every enumerated kernel answers
elimination, faster, fusion, and overlap, and references the best current
performance model for that kernel, logical region, or iteration. Shared
models survive fusion and changing kernel boundaries without duplicating
or summing their predicted time.

Version 2 adds these fields to the coverage and four-question contract::

    version: 2
    kernels:
      - kernel: state_update
        model: state_region
        # full_name, share_pct, ncu, and all four questions as before
    models:
      - id: state_region
        scope: region                 # kernel | region | iteration
        operating_point:             # prediction and measurement match these conditions
          concurrency: 512
          precision: bf16
          hardware: B200
          timing: in-window GPU interval union
        derivation: "mandatory bytes / independently measured bandwidth = 2 ms"
        assumptions: ["state is read once and written once"]
        evidence: ["analysis/traffic.md: byte derivation and bandwidth experiment"]
        predicted_ms: 2.0              # null only with unexplained + next_test
        measured_ms: 3.0               # null only with unexplained + next_test
        measurement_evidence: ["analysis/nsys_analysis/regions.json: state_region"]
        unexplained: "1 ms remains; replay suggests incomplete memory-level parallelism"
        next_test: "measure occupancy and outstanding requests at this shape"
    model_revisions:
      - round: 2
        model: state_region
        reason: "counter evidence shows a required second state read"
        evidence: ["rounds/round_2/analysis/ncu.txt: DRAM read bytes"]
        changes:
          predicted_ms: {from: 1.0, to: 2.0}

The initial ledger has an empty revision history. New models establish their
first derivation and evidence in ``models``; a revision requires a model from
the previous ledger. Revisions retain their ordered history. Changes to the model's scope,
operating point, derivation, assumptions, or prediction require a current-round,
evidence-backed revision matching the previous and current values. Removing
a model requires ``changes.removed: {from: <previous model>, to: null}``,
which also preserves what was removed. Measurements may improve beyond a
prediction: that falsifies the model and leaves an explicit residual to
investigate. A failed optimization alone does not establish a physical floor.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import yaml

LEDGER_VERSION = 2
LEDGER_FILENAME = "kernel_ledger.yaml"

DISPOSITIONS = ("item", "dismissed")

# The per-kernel questions every row must answer. Order is the order they
# are posed in the analyzer prompt and rendered in the report.
QUESTIONS = ("elimination", "faster", "fusion", "overlap")

# The curated vocabulary of *why a kernel cannot be eliminated, made
# faster, fused, or overlapped* — the leading tag of a `dismissed` ref,
# spelled out per question in the analyzer prompt. Collected here so
# consumers can reuse the same "why not" enum rather than growing a
# parallel vocabulary that drifts.
#
# A ref is `tag` or `tag: <detail>`; :func:`dismissal_tag` splits it.
# Not enforced on ``ref`` itself: a dismissal's value is its cited
# evidence, and rejecting a well-evidenced ref for leading with an
# unlisted tag would cost a whole analyzer re-run over a label.
DISMISSAL_TAGS: tuple[str, ...] = (
    # elimination — should this work happen at all?
    "mandatory-math",
    "padding-minimal",
    "already-hoisted",
    "fast-path-active",
    "fast-path-blocked",
    # faster — is there a better kernel for this shape?
    "at-sol-floor",
    "needs-rebuild",
    # fusion — can a neighbor absorb it?
    "multi-consumer-pinned",
    "already-fused",
    "phase-boundary",
    "neighbors-at-bandwidth-floor",
    # overlap — must it run alone?
    "graph-disabled",
    "no-independent-partner",
    "resource-saturated",
    "already-concurrent",
    # scope and materiality — shared across the four
    "below-materiality",
    "approach-restricted",
    "accuracy-scope",
)


def dismissal_tag(ref: str) -> str:
    """The leading tag of a dismissal ``ref`` (``tag`` or ``tag: <detail>``)."""
    return str(ref).split(":", 1)[0].strip()


# Questions whose verdict rests on an observed relationship to other work:
# question -> (field name, what the field must carry). Recorded so a
# fusion/overlap verdict cites the trace rather than a guess.
_EVIDENCE_FIELD = {
    "elimination": (
        "why_it_runs",
        "what consumes this kernel's output, or the guard/selector that chose "
        "this path — what an elimination verdict rests on",
    ),
    "fusion": ("neighbors", "the observed adjacency a fusion verdict rests on"),
    "overlap": (
        "concurrent_with",
        "the candidate partner work (or the observed serialization) an overlap verdict rests on",
    ),
}

# ncu bound classes per the perf-nsight-compute-analysis skill.
BOUND_CLASSES = ("compute", "memory", "latency", "balanced", "comm")

# Shorthand analyzers have written (or plausibly will) for the bound
# enum, mapped to the canonical value. Normalized on load — an alias here
# must be unambiguous; anything else still fails validation.
_BOUND_ALIASES = {
    "compute-bound": "compute",
    "sm": "compute",
    "math": "compute",
    "memory-bound": "memory",
    "mem": "memory",
    "memory-bw": "memory",
    "bandwidth": "memory",
    "latency-bound": "latency",
    "launch": "latency",
    "launch-latency": "latency",
    "mixed": "balanced",
    "communication": "comm",
    "comm-bound": "comm",
    "nccl": "comm",
    "collective": "comm",
}

_NCU_METRIC_FIELDS = ("duration_us", "sm_sol_pct", "mem_sol_pct", "occupancy_pct")

# |enumerated + other - 100| tolerance: the share percentages are rounded
# per row, so the two buckets may miss 100 by a little — but a large gap
# means rows were dropped without being rolled into `other`.
_COVERAGE_SUM_TOLERANCE = 2.0
# Slack on the coverage target itself (rounding of the enumerated sum).
_COVERAGE_TARGET_TOLERANCE = 0.5
# Rounding may differ between individual rows and their declared total.
_COVERAGE_ROWS_TOLERANCE = 0.5


class LedgerError(ValueError):
    """Raised when ``kernel_ledger.yaml`` fails schema validation."""


def _is_number(value: Any) -> bool:
    # bool is an int subclass — reject it explicitly.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_coverage(data: Mapping[str, Any], errors: list[str]) -> None:
    coverage = data.get("coverage")
    if not isinstance(coverage, dict):
        errors.append(
            f"'coverage' must be a mapping with 'enumerated_share_pct', "
            f"'other_share_pct', 'min_share_pct', and 'gpu_busy_pct', "
            f"got {coverage!r}"
        )
        return
    values: dict[str, float] = {}
    for field in ("enumerated_share_pct", "other_share_pct", "min_share_pct"):
        value = coverage.get(field)
        if not _is_number(value) or value < 0:
            errors.append(f"'coverage.{field}' must be a number >= 0, got {value!r}")
        else:
            values[field] = float(value)
    # The wall-clock conversion factor. `share_pct` is a share of *GPU
    # time*, while every gate downstream (`optimize.noise_floor_pct`,
    # `expected_gain_pct`) is a share of wall clock — on a host-bound
    # deployment the two differ by 1/busy, so a materiality dismissal
    # computed in GPU-time units overstates every candidate.
    busy = coverage.get("gpu_busy_pct")
    if not _is_number(busy) or not 0 < busy <= 100:
        errors.append(
            f"'coverage.gpu_busy_pct' must be a number in (0, 100] — the GPU "
            f"busy share of the profiled window (nsys), which converts a "
            f"kernel's share of GPU time into its share of wall clock; "
            f"got {busy!r}"
        )
    if {"enumerated_share_pct", "other_share_pct"} <= values.keys():
        total = values["enumerated_share_pct"] + values["other_share_pct"]
        if abs(total - 100.0) > _COVERAGE_SUM_TOLERANCE:
            errors.append(
                f"'coverage.enumerated_share_pct' + 'coverage.other_share_pct' "
                f"must account for ~100% of profiled GPU time, got {total:.1f} — "
                f"kernels dropped from the ledger must be rolled into "
                f"'other_share_pct', never silently discarded"
            )


def _validate_bound(holder: dict[str, Any], where: str, errors: list[str], hint: str = "") -> None:
    """Validate — and normalize in place — the ``bound`` class in ``holder``."""
    bound = holder.get("bound")
    if isinstance(bound, str) and bound not in BOUND_CLASSES:
        canonical = _BOUND_ALIASES.get(bound.strip().lower(), bound.strip().lower())
        if canonical in BOUND_CLASSES:
            holder["bound"] = canonical
            bound = canonical
    if bound not in BOUND_CLASSES:
        errors.append(f"'{where}' must be one of {list(BOUND_CLASSES)}, got {bound!r}{hint}")


def _enumerated_share(data: Mapping[str, Any]) -> float | None:
    """Sum valid kernel rows, without trusting the declared coverage total."""
    rows = data.get("kernels")
    if not isinstance(rows, list) or not rows:
        return None
    shares = [row.get("share_pct") for row in rows if isinstance(row, Mapping)]
    if len(shares) != len(rows) or any(not _is_number(share) or share < 0 for share in shares):
        return None
    return sum(shares)


def _validate_ncu(row: dict[str, Any], where: str, errors: list[str]) -> None:
    """Validate a row's ``ncu`` block and the ``bound`` class it owes.

    ``ncu`` is either the metrics mapping or the ``unavailable: <reason>``
    degrade string; ``bound`` lives inside ``ncu`` in the first shape and on
    the row in the second, since the dispositions rest on it.
    """
    entry = row.get("ncu")
    if isinstance(entry, str):
        # The honest degrade for a kernel no capture pass reached — the
        # dispositions and their bound class are still owed (from nsys
        # shares + the source).
        if not entry.strip():
            errors.append(f"'{where}.ncu' string form must be non-empty (the reason)")
        _validate_bound(
            row,
            f"{where}.bound",
            errors,
            hint=(
                " — with the whole-block 'unavailable: <reason>' degrade the bound "
                "class lives on the row, beside 'ncu' (a collective records 'comm')"
            ),
        )
        return
    if not isinstance(entry, dict):
        errors.append(
            f"'{where}.ncu' must be a metrics mapping or a non-empty "
            f"'unavailable: <reason>' string, got {entry!r}"
        )
        return
    # ncu often times a kernel while its SOL / occupancy sections come back
    # empty (replay stalls, LaunchFailed, the hang-detector budget). Those
    # metrics may be null, but only with a non-empty `note` saying why.
    note = entry.get("note")
    has_note = isinstance(note, str) and bool(note.strip())
    missing: list[str] = []
    for field in _NCU_METRIC_FIELDS:
        value = entry.get(field)
        if value is None:
            missing.append(field)
            continue
        if not _is_number(value) or value < 0:
            errors.append(f"'{where}.ncu.{field}' must be a number >= 0, got {value!r}")
    if missing and not has_note:
        errors.append(
            f"'{where}.ncu' leaves {missing} null without a 'note' — a null "
            f"metric must be accompanied by a non-empty 'note' explaining why "
            f"the capture did not yield it (or use the whole-block "
            f"'unavailable: <reason>' string form)"
        )
    _validate_bound(entry, f"{where}.ncu.bound", errors)


def _validate_disposition(
    row: Mapping[str, Any], question: str, where: str, errors: list[str]
) -> None:
    block = row.get(question)
    if not isinstance(block, dict):
        errors.append(
            f"'{where}.{question}' must be a mapping with 'disposition' and "
            f"'ref' — every kernel row answers all four questions "
            f"{list(QUESTIONS)}; got {block!r}"
        )
        return
    disposition = block.get("disposition")
    if disposition not in DISPOSITIONS:
        errors.append(
            f"'{where}.{question}.disposition' must be one of {list(DISPOSITIONS)}, "
            f"got {disposition!r}"
        )
    ref = block.get("ref")
    if not isinstance(ref, str) or not ref.strip():
        errors.append(
            f"'{where}.{question}.ref' must be a non-empty string (a roadmap "
            f"item id, or the evidence-backed dismissal), got {ref!r}"
        )
    evidence = _EVIDENCE_FIELD.get(question)
    if evidence is not None and disposition == "dismissed":
        # The observed relationship is the evidence a dismissal rests on;
        # a promoted `item` carries it in the roadmap entry `ref` names.
        field, carries = evidence
        observed = block.get(field)
        if not isinstance(observed, str) or not observed.strip():
            errors.append(
                f"'{where}.{question}.{field}' must be a non-empty string when "
                f"the disposition is 'dismissed' — {carries}; got {observed!r}"
            )


def _validate_row(row: Any, index: int, seen: set[str], errors: list[str]) -> None:
    where = f"kernels[{index}]"
    if not isinstance(row, dict):
        errors.append(f"'{where}' must be a mapping, got {type(row).__name__}")
        return
    kernel = row.get("kernel")
    if not isinstance(kernel, str) or not kernel.strip():
        errors.append(f"'{where}.kernel' must be a non-empty string, got {kernel!r}")
    elif kernel in seen:
        errors.append(f"'{where}.kernel' duplicates {kernel!r} — row keys must be unique")
    else:
        seen.add(kernel)
    full_name = row.get("full_name")
    if not isinstance(full_name, str) or not full_name.strip():
        errors.append(f"'{where}.full_name' must be a non-empty string, got {full_name!r}")
    model = row.get("model")
    if not isinstance(model, str) or not model.strip():
        errors.append(f"'{where}.model' must be a non-empty model id, got {model!r}")
    share = row.get("share_pct")
    if not _is_number(share) or share < 0:
        errors.append(f"'{where}.share_pct' must be a number >= 0, got {share!r}")
    _validate_ncu(row, where, errors)
    for question in QUESTIONS:
        _validate_disposition(row, question, where, errors)


MODEL_SCOPES = ("kernel", "region", "iteration")
_THEORY_FIELDS = ("scope", "operating_point", "derivation", "assumptions", "predicted_ms")
_MODEL_FIELDS = (
    *_THEORY_FIELDS,
    "evidence",
    "measured_ms",
    "measurement_evidence",
    "unexplained",
    "next_test",
)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any, where: str, errors: list[str], *, nonempty: bool) -> None:
    if (
        not isinstance(value, list)
        or (nonempty and not value)
        or any(not _nonempty_string(entry) for entry in value)
    ):
        qualifier = "non-empty " if nonempty else ""
        errors.append(f"'{where}' must be a {qualifier}list of non-empty strings")


def _validate_models(data: Mapping[str, Any], errors: list[str]) -> None:
    models = data.get("models")
    if not isinstance(models, list) or not models:
        errors.append("'models' must be a non-empty list of performance models")
        return
    by_id: dict[str, dict[str, Any]] = {}
    for index, model in enumerate(models):
        where = f"models[{index}]"
        if not isinstance(model, dict):
            errors.append(f"'{where}' must be a mapping")
            continue
        model_id = model.get("id")
        if not _nonempty_string(model_id):
            errors.append(f"'{where}.id' must be a non-empty string")
        elif model_id in by_id:
            errors.append(f"'{where}.id' duplicates {model_id!r}")
        else:
            by_id[model_id] = model
        if model.get("scope") not in MODEL_SCOPES:
            errors.append(f"'{where}.scope' must be one of {list(MODEL_SCOPES)}")
        point = model.get("operating_point")
        if not isinstance(point, dict) or not point:
            errors.append(f"'{where}.operating_point' must be a non-empty mapping")
        if not _nonempty_string(model.get("derivation")):
            errors.append(f"'{where}.derivation' must be a non-empty string")
        _string_list(model.get("assumptions"), f"{where}.assumptions", errors, nonempty=False)
        _string_list(model.get("evidence"), f"{where}.evidence", errors, nonempty=True)
        for field in ("predicted_ms", "measured_ms"):
            value = model.get(field)
            if field not in model or (value is not None and (not _is_number(value) or value < 0)):
                errors.append(f"'{where}.{field}' must be a finite number >= 0 or explicit null")
        predicted = model.get("predicted_ms")
        measured = model.get("measured_ms")
        _string_list(
            model.get("measurement_evidence"),
            f"{where}.measurement_evidence",
            errors,
            nonempty=measured is not None,
        )
        unresolved = predicted is None or measured is None or predicted != measured
        for field in ("unexplained", "next_test"):
            value = model.get(field)
            if not isinstance(value, str) or (unresolved and not value.strip()):
                errors.append(
                    f"'{where}.{field}' must be a string, non-empty for a null time or "
                    "nonzero measured-minus-predicted residual; missing evidence or a "
                    "falsified prediction remains open for investigation"
                )
    references: dict[str, list[str]] = {}
    kernels = data.get("kernels")
    for index, row in enumerate(kernels if isinstance(kernels, list) else []):
        if not isinstance(row, dict) or not _nonempty_string(row.get("model")):
            continue
        model_id = row["model"]
        if model_id not in by_id:
            errors.append(f"'kernels[{index}].model' references unknown model {model_id!r}")
        references.setdefault(model_id, []).append(row.get("kernel", f"kernels[{index}]"))
    for model_id, model in by_id.items():
        if model.get("scope") == "kernel" and len(references.get(model_id, [])) != 1:
            errors.append(
                f"model {model_id!r} with scope 'kernel' must reference exactly one kernel row; "
                "use 'region' or 'iteration' for a shared model"
            )


def _is_removal(revision: Mapping[str, Any]) -> bool:
    removed = revision.get("changes", {}).get("removed", {})
    return (
        isinstance(removed, dict)
        and isinstance(removed.get("from"), dict)
        and ("to" in removed and removed["to"] is None)
    )


def _validate_revisions(data: Mapping[str, Any], errors: list[str]) -> None:
    revisions = data.get("model_revisions")
    if not isinstance(revisions, list):
        errors.append("'model_revisions' must be a list (empty for an initial model)")
        return
    models = data.get("models")
    known_ids = {
        model["id"]
        for model in (models if isinstance(models, list) else [])
        if isinstance(model, dict) and _nonempty_string(model.get("id"))
    }
    # Historical revisions to a retired model remain valid after its removal.
    for revision in revisions:
        if (
            isinstance(revision, dict)
            and isinstance(revision.get("changes"), dict)
            and _is_removal(revision)
            and _nonempty_string(revision.get("model"))
        ):
            known_ids.add(revision["model"])
    for index, revision in enumerate(revisions):
        where = f"model_revisions[{index}]"
        if not isinstance(revision, dict):
            errors.append(f"'{where}' must be a mapping")
            continue
        round_no = revision.get("round")
        if not isinstance(round_no, int) or isinstance(round_no, bool) or round_no < 1:
            errors.append(f"'{where}.round' must be a positive integer")
        model_id = revision.get("model")
        if not _nonempty_string(model_id):
            errors.append(f"'{where}.model' must be a non-empty model id")
        elif model_id not in known_ids:
            errors.append(f"'{where}.model' references unknown model {model_id!r}")
        if not _nonempty_string(revision.get("reason")):
            errors.append(f"'{where}.reason' must explain the model revision")
        _string_list(revision.get("evidence"), f"{where}.evidence", errors, nonempty=True)
        changes = revision.get("changes")
        if not isinstance(changes, dict) or not changes:
            errors.append(f"'{where}.changes' must be a non-empty mapping of from/to values")
            continue
        for field, change in changes.items():
            if field not in (*_MODEL_FIELDS, "removed"):
                errors.append(f"'{where}.changes' names unknown model field {field!r}")
            if not isinstance(change, dict) or set(change) != {"from", "to"}:
                errors.append(f"'{where}.changes.{field}' must contain exactly 'from' and 'to'")
            elif change["from"] == change["to"]:
                errors.append(f"'{where}.changes.{field}' must record an actual change")
            elif field in ("predicted_ms", "measured_ms"):
                for endpoint, value in change.items():
                    if value is not None and (not _is_number(value) or value < 0):
                        errors.append(
                            f"'{where}.changes.{field}.{endpoint}' must be a finite "
                            "number >= 0 or null"
                        )
            if field == "removed" and not _is_removal(revision):
                errors.append(
                    f"'{where}.changes.removed' must record the previous model mapping "
                    "as 'from' and null as 'to'"
                )


def _validate_model_updates(
    ledger: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    round_no: int | None,
    errors: list[str],
) -> None:
    revisions = ledger.get("model_revisions", [])
    if previous is None:
        if revisions:
            errors.append("initial ledger must have an empty 'model_revisions' history")
        return
    old_history = previous.get("model_revisions", [])
    if revisions[: len(old_history)] != old_history:
        errors.append("'model_revisions' must preserve the complete previous revision history")
        return
    additions = revisions[len(old_history) :]
    if round_no is not None and any(entry["round"] != round_no for entry in additions):
        errors.append("new 'model_revisions' must record the current round")
    old_models = {model["id"]: model for model in previous.get("models", [])}
    new_models = {model["id"]: model for model in ledger.get("models", [])}
    for entry in additions:
        if entry["model"] not in old_models:
            errors.append(
                f"model_revisions for {entry['model']!r} requires a model from the previous "
                "ledger; a new model establishes its first derivation and evidence in 'models'"
            )
    for model_id, old in old_models.items():
        updates = [entry for entry in additions if entry["model"] == model_id]
        new = new_models.get(model_id)
        if new is None:
            if not any(
                entry["changes"].get("removed") == {"from": old, "to": None} for entry in updates
            ):
                errors.append(
                    f"removed model {model_id!r} requires an evidence-backed current-round "
                    "model_revisions entry with changes.removed from the previous model to null"
                )
            continue
        changed_theory = {field for field in _THEORY_FIELDS if old.get(field) != new.get(field)}
        recorded_fields = {field for entry in updates for field in entry["changes"]}
        for field in changed_theory - recorded_fields:
            errors.append(
                f"model {model_id!r} changed {field!r} without an evidence-backed "
                "current-round model_revisions entry"
            )
        for field in recorded_fields:
            if field == "removed":
                errors.append(f"model {model_id!r} is still present but has a removal revision")
                continue
            value = old.get(field)
            for entry in updates:
                change = entry["changes"].get(field)
                if change is None:
                    continue
                if change["from"] != value:
                    errors.append(
                        f"model_revisions for {model_id!r}.{field} has 'from' that does not "
                        "match the previous value"
                    )
                value = change["to"]
            if value != new.get(field):
                errors.append(
                    f"model_revisions for {model_id!r}.{field} has 'to' that does not "
                    "match the current value"
                )
        if old.get("measured_ms") != new.get("measured_ms") and new.get("measured_ms") is not None:
            if not new.get("measurement_evidence"):
                errors.append(
                    f"model {model_id!r} refreshed measured_ms without measurement evidence"
                )


def load_ledger(path: str | Path) -> dict[str, Any]:
    """Parse ``path`` as YAML and validate the ledger schema.

    Shape only — roadmap cross-references and the coverage target need
    context the file does not carry; run :func:`cross_validate` for
    those. Raises :class:`LedgerError` with **every** detected problem
    batched into a single message.
    """
    ledger_path = Path(path)
    if not ledger_path.is_file():
        raise LedgerError(f"kernel ledger not found: {ledger_path}")

    try:
        data = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LedgerError(f"{ledger_path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise LedgerError(
            f"{ledger_path} must be a YAML mapping at the top level, got {type(data).__name__}"
        )

    errors: list[str] = []

    if data.get("version") != LEDGER_VERSION:
        errors.append(f"'version' must be {LEDGER_VERSION}, got {data.get('version')!r}")

    source = data.get("source")
    if not isinstance(source, str) or not source.strip():
        errors.append(f"'source' must name the shares enumerated, got {source!r}")

    _validate_coverage(data, errors)

    kernels = data.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        errors.append(f"'kernels' must be a non-empty list, got {type(kernels).__name__}")
        kernels = []
    seen: set[str] = set()
    for index, row in enumerate(kernels):
        _validate_row(row, index, seen, errors)

    enumerated = _enumerated_share(data)
    coverage = data.get("coverage")
    declared = coverage.get("enumerated_share_pct") if isinstance(coverage, dict) else None
    if enumerated is not None and _is_number(declared):
        if abs(enumerated - declared) > _COVERAGE_ROWS_TOLERANCE:
            errors.append(
                f"'coverage.enumerated_share_pct' ({declared}) must match the sum of "
                f"'kernels[].share_pct' ({enumerated:.3f}) within "
                f"{_COVERAGE_ROWS_TOLERANCE} percentage points"
            )

    _validate_models(data, errors)
    _validate_revisions(data, errors)

    if errors:
        bullet = "\n  - "
        raise LedgerError(
            f"{ledger_path} failed kernel-ledger schema validation:{bullet}{bullet.join(errors)}"
        )
    return data


def cross_validate(
    ledger: Mapping[str, Any],
    roadmap: Mapping[str, Any],
    coverage_target_pct: float,
    *,
    previous: Mapping[str, Any] | None = None,
    round_no: int | None = None,
) -> list[str]:
    """Context checks a shape-valid ledger still owes; returns the problems.

    - Every ``disposition: item`` ref must name an id present in
      ``roadmap.yaml`` (any status — accepted / failed items *were*
      considered). A ref to a nonexistent id means the possibility was
      claimed planned but never actually landed in the plan.
    - The enumerated rows must reach the task's declared coverage target
      — the deterministic teeth behind "every kernel was considered".
    - When ``previous`` is supplied, model revisions preserve their full
      history and account for changes to the theory or removal of a model.
      ``round_no`` identifies the current round, including repeated analyzer
      turns within that round.
    """
    errors: list[str] = []
    item_ids = {
        item.get("id")
        for item in roadmap.get("items", [])
        if isinstance(item, Mapping) and item.get("id")
    }
    for index, row in enumerate(ledger.get("kernels", [])):
        for question in QUESTIONS:
            block = row.get(question, {})
            if block.get("disposition") == "item" and block.get("ref") not in item_ids:
                errors.append(
                    f"'kernels[{index}].{question}.ref' ({block.get('ref')!r}) does "
                    f"not match any roadmap item id — a disposition of 'item' "
                    f"must point at a real roadmap.yaml entry"
                )
    enumerated = _enumerated_share(ledger)
    if enumerated is not None and enumerated < coverage_target_pct - _COVERAGE_TARGET_TOLERANCE:
        errors.append(
            f"sum of 'kernels[].share_pct' ({enumerated}) is below the task's "
            f"'profile.kernel_coverage.coverage_target_pct' ({coverage_target_pct}) — "
            f"enumerate further down the ranking (grouping related kernels is "
            f"fine) until the target is covered"
        )
    _validate_model_updates(ledger, previous, round_no, errors)
    return errors
