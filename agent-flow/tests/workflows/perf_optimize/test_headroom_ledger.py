"""Tests for the perf-optimize ``headroom_ledger.yaml`` schema."""

from __future__ import annotations

import copy

import pytest
import yaml

from agent_flow.workflows.perf_optimize import headroom_ledger
from agent_flow.workflows.perf_optimize.headroom_ledger import (
    HeadroomLedgerError,
    LedgerContext,
    append_dispositions,
    cross_validate,
    engineering_ranking,
    load_ledger,
    partition_totals,
    record_history,
    unexplained_ranking,
)

# --------------------------------------------------------------------- helpers


def _part(part_id: str = "alpha:bf16", **overrides) -> dict:
    part = {
        "id": part_id,
        "source": "analytic",
        "at": {
            64: {"measured_ms": 1.0, "sol_ms": 0.6, "gap_ms": 0.4},
            512: {"measured_ms": 4.0, "sol_ms": 2.0, "gap_ms": 2.0},
        },
        "sensitivity": "steep",
        "bound": "memory",
        "kernels": ["alpha_kernel"],
        "partition": {
            "closed_ms": 0.0,
            "attributed_ms": 0.0,
            "open_ms": 0.0,
            "unexplained_ms": 2.0,
        },
    }
    part.update(overrides)
    return part


def _second_part(**overrides) -> dict:
    part = _part(
        "beta:fp8",
        at={
            64: {"measured_ms": 0.5, "sol_ms": 0.4, "gap_ms": 0.1},
            512: {"measured_ms": 2.0, "sol_ms": 1.4, "gap_ms": 0.6},
        },
        sensitivity="flat",
        kernels=["beta_kernel"],
        partition={
            "closed_ms": 0.0,
            "attributed_ms": 0.0,
            "open_ms": 0.0,
            "unexplained_ms": 0.6,
        },
    )
    part.update(overrides)
    return part


def _ledger(**overrides) -> dict:
    data = {
        "version": headroom_ledger.HEADROOM_LEDGER_VERSION,
        "operating_point": {
            "concurrency": [64, 512],
            "isl": 1024,
            "osl": 1024,
            "build_sha": "1418149a84",
            "node": "node-017",
            "capture_state": "gpu-bound",
        },
        "timing": {"step_ms": 10.0, "kernel_ms": 9.0},
        "coverage": {
            "modeled_kernel_ms": 6.0,
            "modeled_pct": 66.7,
            "empirical_kernel_ms": 2.0,
            "unmodeled_kernel_ms": 1.0,
            "non_kernel_ms": 1.0,
        },
        "parts": [_part(), _second_part()],
    }
    data.update(overrides)
    return data


def _target(**overrides) -> dict:
    target = {
        "target_ms": 3.0,
        "structure": "One persistent kernel per layer; S stays resident.",
        "basis": "derived",
        "basis_ref": "same recipe arithmetic as sol_ms (hybrid.py)",
        "today": "5 kernels round-tripping S through HBM",
        "achieved_efficiency": {
            "value": 0.78,
            "source": "moe_gemm_fc1 demonstrates 78% MBU in this trace",
        },
        "delta": [
            {"cause": "state-round-trip", "ms": 0.8, "evidence": "cuda_gpu_trace step 120"},
            {"cause": "unattributed", "ms": 0.2},
        ],
        "falsifier": "if S^T k cannot be held in registers the fusion is unbuildable",
    }
    target.update(overrides)
    return target


def _disposition(**overrides) -> dict:
    entry = {
        "item": "opt-004",
        "round": 3,
        "outcome": "failed",
        "gap_implication": "mechanism-inapplicable",
        "lever": "launch-geometry-tuning",
        "note": "the tuned mapping is gated on T == 4; this deployment runs T == 3",
        "evidence": "rounds/round_3/item_1_opt-004/attempt_1/evaluation.md",
    }
    entry.update(overrides)
    return entry


def _write(tmp_path, data) -> str:
    path = tmp_path / headroom_ledger.HEADROOM_LEDGER_FILENAME
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(path)


def _roadmap(*items) -> dict:
    """A roadmap from ``(id, status, parts)`` triples (or bare ids)."""
    rows = []
    for item in items:
        if isinstance(item, str):
            item = (item, "pending", [])
        item_id, status, parts = item
        rows.append({"id": item_id, "status": status, "parts": list(parts)})
    return {"items": rows}


def _kernel_row(kernel: str, share: float, dismissed: bool = True) -> dict:
    verdict = "dismissed" if dismissed else "item"
    return {
        "kernel": kernel,
        "share_pct": share,
        **{q: {"disposition": verdict, "ref": "mandatory-math: x"} for q in ("elimination",)},
        **{
            q: {"disposition": verdict, "ref": "at-sol-floor: mem SOL 89%"}
            for q in ("faster", "fusion", "overlap")
        },
    }


def _context(**overrides) -> LedgerContext:
    defaults = {
        "sol": {
            "per_op": [
                {"region": "alpha:bf16", "measured_ms": 4.0, "sol_ms": 2.0, "gap_ms": 2.0},
                {"region": "beta:fp8", "measured_ms": 2.0, "sol_ms": 1.4, "gap_ms": 0.6},
            ]
        },
        "regions": {
            "regions": [
                {"region": "alpha:bf16", "measured_ms": 4.0},
                {"region": "beta:fp8", "measured_ms": 2.0},
            ]
        },
        # 4.0 / 9.0 = 44.444%, 2.0 / 9.0 = 22.222% of kernel_ms.
        "kernels": {
            "kernels": [
                _kernel_row("alpha_kernel", 44.4444),
                _kernel_row("beta_kernel", 22.2222),
            ]
        },
    }
    defaults.update(overrides)
    return LedgerContext(**defaults)


def _invalid_errors(tmp_path, mutate) -> str:
    """Apply ``mutate`` to a deep copy of a valid ledger; return the error text."""
    data = _ledger()
    mutate(data)
    with pytest.raises(HeadroomLedgerError) as exc_info:
        load_ledger(_write(tmp_path, data))
    return str(exc_info.value)


# ------------------------------------------------------------------ load_ledger


def test_valid_ledger_loads(tmp_path):
    data = load_ledger(_write(tmp_path, _ledger()))
    assert data["version"] == headroom_ledger.HEADROOM_LEDGER_VERSION
    assert [p["id"] for p in data["parts"]] == ["alpha:bf16", "beta:fp8"]


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(HeadroomLedgerError, match="not found"):
        load_ledger(tmp_path / "nope.yaml")


def test_non_mapping_is_rejected(tmp_path):
    path = tmp_path / "headroom_ledger.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(HeadroomLedgerError, match="mapping at the top level"):
        load_ledger(path)


def test_unknown_top_level_keys_are_rejected(tmp_path):
    # A silently ignored block is how a campaign runs under settings
    # nobody applied — every block here changes what gets built.
    message = _invalid_errors(tmp_path, lambda d: d.update({"targets": []}))
    assert "unknown top-level key" in message
    assert "'targets'" in message


@pytest.mark.parametrize("bad", [0, 2, None, "1"])
def test_version_must_match(tmp_path, bad):
    assert "'version'" in _invalid_errors(tmp_path, lambda d: d.update({"version": bad}))


def test_errors_are_batched(tmp_path):
    def mutate(data):
        data["version"] = 99
        data["operating_point"]["capture_state"] = "unknown"
        data["parts"][0]["source"] = "guessed"

    message = _invalid_errors(tmp_path, mutate)
    assert "'version'" in message
    assert "capture_state" in message
    assert "source" in message


# ------------------------------------------------------------- operating point


def test_operating_point_requires_the_measurement_context(tmp_path):
    message = _invalid_errors(tmp_path, lambda d: d.pop("operating_point"))
    assert "operating_point" in message


@pytest.mark.parametrize("bad", [[], [512, 64], [0], ["64"], None])
def test_concurrency_must_be_ascending_positive_ints(tmp_path, bad):
    message = _invalid_errors(
        tmp_path, lambda d: d["operating_point"].__setitem__("concurrency", bad)
    )
    assert "operating_point.concurrency" in message


@pytest.mark.parametrize("state", headroom_ledger.CAPTURE_STATES)
def test_every_capture_state_is_accepted(tmp_path, state):
    data = _ledger()
    data["operating_point"]["capture_state"] = state
    assert load_ledger(_write(tmp_path, data))["operating_point"]["capture_state"] == state


# -------------------------------------------------------------------- coverage


def test_coverage_is_mandatory(tmp_path):
    # Without it, "the modeled parts have little gap" reads as "there is
    # little headroom" while a large share carries no ceiling at all.
    assert "'coverage'" in _invalid_errors(tmp_path, lambda d: d.pop("coverage"))


def test_coverage_must_reconcile_to_kernel_ms(tmp_path):
    message = _invalid_errors(
        tmp_path, lambda d: d["coverage"].__setitem__("empirical_kernel_ms", 0.5)
    )
    assert "kernel buckets" in message


def test_coverage_must_reconcile_to_step_ms(tmp_path):
    message = _invalid_errors(tmp_path, lambda d: d["coverage"].__setitem__("non_kernel_ms", 5.0))
    assert "non_kernel_ms" in message


def test_coverage_residual_absorbs_rounding(tmp_path):
    # The reference campaign's own two sources for "modeled ms" disagree
    # by 0.36%; an exact-closure validator would wedge the round.
    data = _ledger()
    data["coverage"]["modeled_kernel_ms"] = 5.99
    data["coverage"]["residual_ms"] = 0.01
    assert load_ledger(_write(tmp_path, data))


def test_kernel_ms_cannot_exceed_step_ms(tmp_path):
    message = _invalid_errors(tmp_path, lambda d: d["timing"].__setitem__("kernel_ms", 12.0))
    assert "exceeds" in message


# ----------------------------------------------------------------------- parts


def test_part_ids_must_be_unique(tmp_path):
    message = _invalid_errors(tmp_path, lambda d: d["parts"].append(_part()))
    assert "duplicates" in message


def _unbounded(part_id: str = "alpha:bf16", source: str = "empirical") -> dict:
    """A part whose time is counted but carries no analytic ceiling."""
    return {
        "id": part_id,
        "source": source,
        "at": {
            64: {"measured_ms": 1.0, "sol_ms": None, "gap_ms": None},
            512: {"measured_ms": 4.0, "sol_ms": None, "gap_ms": None},
        },
        "sensitivity": "flat",
        "bound": "memory",
    }


@pytest.mark.parametrize("source", headroom_ledger.PART_SOURCES)
def test_every_part_source_is_accepted(tmp_path, source):
    data = _ledger()
    if source != "analytic":
        data["parts"][0] = _unbounded(source=source)
    else:
        data["parts"][0]["source"] = source
    assert load_ledger(_write(tmp_path, data))["parts"][0]["source"] == source


def test_a_part_with_no_ceiling_may_not_invent_one(tmp_path):
    # Writing sol_ms: 0 claims the whole measured time is headroom;
    # writing measured_ms claims there is none. Both are assertions the
    # campaign has no evidence for.
    data = _ledger()
    part = _unbounded()
    part["at"][512]["sol_ms"] = 0.0
    data["parts"][0] = part
    with pytest.raises(HeadroomLedgerError, match="must be null"):
        load_ledger(_write(tmp_path, data))


def test_a_part_with_no_ceiling_owes_no_partition(tmp_path):
    data = _ledger()
    part = _unbounded()
    part["partition"] = {
        "closed_ms": 0.0,
        "attributed_ms": 0.0,
        "open_ms": 0.0,
        "unexplained_ms": 4.0,
    }
    data["parts"][0] = part
    with pytest.raises(HeadroomLedgerError, match="no gap to split"):
        load_ledger(_write(tmp_path, data))


def test_unbounded_time_is_totalled_on_its_own_axis(tmp_path):
    # So "no modeled gap" can never be read as "no headroom".
    data = _ledger()
    data["parts"][0] = _unbounded()
    ledger = load_ledger(_write(tmp_path, data))
    totals = partition_totals(ledger)
    assert totals["unbounded_ms"] == pytest.approx(4.0)
    assert totals["gap_ms"] == pytest.approx(0.6)
    assert totals["unexplained_ms"] == pytest.approx(0.6)


def test_gap_must_equal_measured_minus_sol(tmp_path):
    message = _invalid_errors(
        tmp_path, lambda d: d["parts"][0]["at"][512].__setitem__("gap_ms", 1.5)
    )
    assert "gap_ms" in message


def test_negative_gap_requires_a_model_revision(tmp_path):
    def mutate(data):
        data["parts"][0]["at"][512] = {"measured_ms": 1.0, "sol_ms": 2.0, "gap_ms": -1.0}
        data["parts"][0]["partition"]["unexplained_ms"] = -1.0

    # A negative gap is the calculator's own "the model is wrong" signal;
    # leaving it unadjudicated keeps a ceiling the measurement disproved.
    message = _invalid_errors(tmp_path, mutate)
    assert "model_revisions" in message


def test_negative_gap_is_accepted_with_the_adjudication(tmp_path):
    data = _ledger()
    data["parts"][0]["at"][512] = {"measured_ms": 1.0, "sol_ms": 2.0, "gap_ms": -1.0}
    data["parts"][0]["partition"] = {
        "closed_ms": 0.0,
        "attributed_ms": 0.0,
        "open_ms": 0.0,
        "unexplained_ms": -1.0,
    }
    data["model_revisions"] = [
        {
            "round": 2,
            "part": "alpha:bf16",
            "sol_ms": {"from": 2.0, "to": 0.5},
            "cause": "missing-factor",
            "detail": "the recipe multiplied by 1 layer where the model has 92",
            "evidence": "sol_work/sol_recipes/hybrid.py@abc1234",
        }
    ]
    assert load_ledger(_write(tmp_path, data))


def test_partition_must_sum_to_the_primary_gap(tmp_path):
    message = _invalid_errors(
        tmp_path, lambda d: d["parts"][0]["partition"].__setitem__("unexplained_ms", 0.1)
    )
    assert "partition" in message


def test_at_keys_written_as_strings_are_normalized(tmp_path):
    data = _ledger()
    data["parts"][0]["at"] = {
        "64": {"measured_ms": 1.0, "sol_ms": 0.6, "gap_ms": 0.4},
        "512": {"measured_ms": 4.0, "sol_ms": 2.0, "gap_ms": 2.0},
    }
    loaded = load_ledger(_write(tmp_path, data))
    assert sorted(loaded["parts"][0]["at"]) == [64, 512]


def test_sensitivity_may_be_null_for_a_single_point_capture(tmp_path):
    data = _ledger()
    for part in data["parts"]:
        part["at"] = {512: part["at"][512]}
        part["sensitivity"] = None
    data["operating_point"]["concurrency"] = [512]
    assert load_ledger(_write(tmp_path, data))


def test_unknown_sensitivity_is_rejected(tmp_path):
    message = _invalid_errors(tmp_path, lambda d: d["parts"][0].__setitem__("sensitivity", "mild"))
    assert "sensitivity" in message


# ---------------------------------------------------------------- dispositions


def test_disposition_requires_lever_note_and_evidence(tmp_path):
    for field in ("lever", "note", "evidence"):
        entry = _disposition()
        del entry[field]
        message = _invalid_errors(
            tmp_path, lambda d, e=entry: d["parts"][0].__setitem__("dispositions", [e])
        )
        assert field in message


@pytest.mark.parametrize("implication", headroom_ledger.GAP_IMPLICATIONS)
def test_every_gap_implication_is_accepted(tmp_path, implication):
    data = _ledger()
    data["parts"][0]["dispositions"] = [_disposition(gap_implication=implication)]
    assert load_ledger(_write(tmp_path, data))


def test_change_not_live_is_in_the_enum(tmp_path):
    # The value the four-value vocabulary lacked: a change that ran in the
    # diff but never executed in the measured binary bounds nothing.
    assert "change-not-live" in headroom_ledger.GAP_IMPLICATIONS
    assert "change-not-live" not in headroom_ledger.CONVERGENT_GAP_IMPLICATIONS


def test_free_text_gap_implication_is_rejected(tmp_path):
    message = _invalid_errors(
        tmp_path,
        lambda d: d["parts"][0].__setitem__(
            "dispositions", [_disposition(gap_implication="it did not help")]
        ),
    )
    assert "gap_implication" in message


# ----------------------------------------------------------------- attribution


def test_attribution_requires_basis_and_note_when_it_forecloses(tmp_path):
    message = _invalid_errors(
        tmp_path,
        lambda d: d["parts"][0].__setitem__(
            "attribution", {"attributed_ms": 1.0, "basis": None, "note": None}
        ),
    )
    assert "attribution.basis" in message
    assert "attribution.note" in message


def test_zero_attribution_needs_no_basis(tmp_path):
    data = _ledger()
    data["parts"][0]["attribution"] = {"attributed_ms": 0.0, "basis": None, "note": None}
    assert load_ledger(_write(tmp_path, data))


def test_attribution_must_match_the_partition(tmp_path):
    def mutate(data):
        data["parts"][0]["partition"] = {
            "closed_ms": 0.0,
            "attributed_ms": 2.0,
            "open_ms": 0.0,
            "unexplained_ms": 0.0,
        }
        data["parts"][0]["attribution"] = {
            "attributed_ms": 1.0,
            "basis": "convergent-levers",
            "note": "two levers converge",
        }

    assert "disagrees" in _invalid_errors(tmp_path, mutate)


# ---------------------------------------------------------------- target layer


def test_target_block_validates(tmp_path):
    data = _ledger()
    data["parts"][0]["target"] = _target()
    assert load_ledger(_write(tmp_path, data))["parts"][0]["target"]["target_ms"] == 3.0


def test_target_below_sol_is_a_model_defect(tmp_path):
    message = _invalid_errors(
        tmp_path,
        lambda d: d["parts"][0].__setitem__(
            "target",
            _target(
                target_ms=1.0,
                delta=[
                    {"cause": "state-round-trip", "ms": 2.5, "evidence": "trace"},
                    {"cause": "unattributed", "ms": 0.5},
                ],
            ),
        ),
    )
    assert "model_revisions" in message


def test_target_above_measured_forces_a_downward_revision(tmp_path):
    # The same rule that stops optimism stops pessimism: a target the
    # implementation beat was wrong too.
    message = _invalid_errors(
        tmp_path,
        lambda d: d["parts"][0].__setitem__(
            "target",
            _target(
                target_ms=5.0,
                delta=[
                    {"cause": "unattributed", "ms": -1.0},
                ],
            ),
        ),
    )
    assert "target_revisions" in message


def test_target_requires_a_falsifier(tmp_path):
    target = _target()
    del target["falsifier"]
    message = _invalid_errors(tmp_path, lambda d: d["parts"][0].__setitem__("target", target))
    assert "falsifier" in message


def test_target_requires_a_sourced_measured_efficiency(tmp_path):
    target = _target()
    del target["achieved_efficiency"]
    message = _invalid_errors(tmp_path, lambda d: d["parts"][0].__setitem__("target", target))
    assert "achieved_efficiency" in message


def test_target_efficiency_source_must_be_named(tmp_path):
    message = _invalid_errors(
        tmp_path,
        lambda d: d["parts"][0].__setitem__(
            "target", _target(achieved_efficiency={"value": 0.9, "source": ""})
        ),
    )
    assert "achieved_efficiency.source" in message


def test_target_delta_must_carry_exactly_one_unattributed_remainder(tmp_path):
    # Requiring the named causes alone to close the books would force
    # inventing a cause to balance the arithmetic.
    message = _invalid_errors(
        tmp_path,
        lambda d: d["parts"][0].__setitem__(
            "target",
            _target(delta=[{"cause": "state-round-trip", "ms": 1.0, "evidence": "trace"}]),
        ),
    )
    assert "unattributed" in message


def test_target_delta_must_sum_to_the_engineering_gap(tmp_path):
    message = _invalid_errors(
        tmp_path,
        lambda d: d["parts"][0].__setitem__(
            "target",
            _target(
                delta=[
                    {"cause": "state-round-trip", "ms": 0.1, "evidence": "trace"},
                    {"cause": "unattributed", "ms": 0.1},
                ]
            ),
        ),
    )
    assert "delta" in message


def test_named_delta_causes_need_evidence(tmp_path):
    message = _invalid_errors(
        tmp_path,
        lambda d: d["parts"][0].__setitem__(
            "target",
            _target(
                delta=[
                    {"cause": "launch-overhead", "ms": 0.8},
                    {"cause": "unattributed", "ms": 0.2},
                ]
            ),
        ),
    )
    assert "evidence" in message


def test_none_known_is_a_legal_answer(tmp_path):
    # Saying "I don't know how to build this" is a result, not a failure
    # to fill the form: the whole gap is then structural.
    data = _ledger()
    data["parts"][0]["target"] = {
        "target_ms": 4.0,
        "structure": "no known implementation reaches the floor here",
        "basis": "none-known",
        "today": "5 kernels round-tripping S through HBM",
        "falsifier": "a published persistent-state kernel for this shape would refute it",
    }
    assert load_ledger(_write(tmp_path, data))


def test_none_known_must_not_manufacture_headroom(tmp_path):
    data = _ledger()
    data["parts"][0]["target"] = {
        "target_ms": 3.0,
        "structure": "s",
        "basis": "none-known",
        "today": "t",
        "falsifier": "f",
    }
    with pytest.raises(HeadroomLedgerError, match="none-known"):
        load_ledger(_write(tmp_path, data))


# ------------------------------------------------------------------- revisions


def test_model_revision_requires_cause_detail_and_evidence(tmp_path):
    base = {
        "round": 2,
        "part": "alpha:bf16",
        "sol_ms": {"from": 2.0, "to": 0.5},
        "cause": "missing-factor",
        "detail": "the recipe multiplied by 1 layer where the model has 92",
        "evidence": "sol_recipes/hybrid.py@abc1234",
    }
    for field in ("cause", "detail", "evidence"):
        entry = dict(base)
        del entry[field]
        message = _invalid_errors(
            tmp_path, lambda d, e=entry: d.__setitem__("model_revisions", [e])
        )
        assert field in message


def test_a_revision_restating_the_symptom_is_rejected(tmp_path):
    # "The SOL looks too aggressive" is the trigger for an adjudication,
    # not its justification.
    message = _invalid_errors(
        tmp_path,
        lambda d: d.__setitem__(
            "model_revisions",
            [
                {
                    "round": 2,
                    "part": "alpha:bf16",
                    "sol_ms": {"from": 2.0, "to": 0.5},
                    "cause": "wrong-peak",
                    "detail": "wrong-peak",
                    "evidence": "peaks.json",
                }
            ],
        ),
    )
    assert "restates" in message


def test_target_revisions_reuse_the_dismissal_vocabulary(tmp_path):
    data = _ledger()
    data["target_revisions"] = [
        {
            "round": 3,
            "part": "alpha:bf16",
            "target_ms": {"from": 2.62, "to": 2.94},
            "cause": "multi-consumer-pinned",
            "detail": "the gated-norm output feeds the residual add too",
            "found_by": "opt-009 / attempt 2",
            "evidence": "rounds/round_3/item_1_opt-009/attempt_2/optimization_summary.md",
        }
    ]
    assert load_ledger(_write(tmp_path, data))


def test_target_revision_cause_outside_the_vocabulary_is_rejected(tmp_path):
    message = _invalid_errors(
        tmp_path,
        lambda d: d.__setitem__(
            "target_revisions",
            [
                {
                    "round": 3,
                    "part": "alpha:bf16",
                    "target_ms": {"from": 2.6, "to": 2.9},
                    "cause": "on reflection this seems hard",
                    "detail": "register pressure",
                    "found_by": "opt-009 / 2",
                    "evidence": "rounds/round_3/...",
                }
            ],
        ),
    )
    assert "dismissal tags" in message


def test_target_revision_cause_may_carry_a_detail_suffix(tmp_path):
    data = _ledger()
    data["target_revisions"] = [
        {
            "round": 3,
            "part": "alpha:bf16",
            "target_ms": {"from": 2.6, "to": 2.9},
            "cause": "fast-path-blocked: unsupported head_dim guard",
            "detail": "the fused path asserts head_dim in (64, 128)",
            "found_by": "opt-009 / 2",
            "evidence": "rounds/round_3/...",
        }
    ]
    assert load_ledger(_write(tmp_path, data))


def test_lifecycle_entries_validate(tmp_path):
    data = _ledger()
    data["part_lifecycle"] = [
        {
            "round": 2,
            "part": "logits_upcast:bf16_to_fp32",
            "event": "eliminated",
            "by": "opt-001",
            "measured_ms": 0.9792,
            "sol_ms": 0.2871,
        }
    ]
    assert load_ledger(_write(tmp_path, data))


# --------------------------------------------------------------- cross_validate


def test_clean_ledger_cross_validates_clean(tmp_path):
    ledger = load_ledger(_write(tmp_path, _ledger()))
    problems = cross_validate(
        ledger, roadmap=_roadmap("opt-001"), context=_context(), focus_points=[64, 128, 512]
    )
    assert problems == []


def test_analytic_part_must_exist_in_sol_json(tmp_path):
    ledger = load_ledger(_write(tmp_path, _ledger()))
    context = _context(sol={"per_op": [{"region": "beta:fp8", "measured_ms": 2.0}]})
    problems = cross_validate(ledger, roadmap=_roadmap(), context=context)
    assert len(problems) == 1
    assert "alpha:bf16" in problems[0]


def test_join_closure_reports_the_residual_in_both_units(tmp_path):
    ledger = load_ledger(_write(tmp_path, _ledger()))
    context = _context(
        kernels={
            "kernels": [
                # alpha under-claimed by 11.1111% of kernel_ms = 1.0 ms.
                _kernel_row("alpha_kernel", 33.3333),
                _kernel_row("beta_kernel", 22.2222),
                _kernel_row("alpha_kernel_variant", 11.1111),
            ]
        }
    )
    problems = cross_validate(ledger, roadmap=_roadmap(), context=context)
    joined = [p for p in problems if "residual" in p]
    assert len(joined) == 1
    assert "+1.0000 ms" in joined[0]
    # The share form is what names the missing row.
    assert "+11.111%" in joined[0]


def test_join_against_an_unknown_kernel_row_is_reported(tmp_path):
    data = _ledger()
    data["parts"][0]["kernels"] = ["not_in_the_ledger"]
    ledger = load_ledger(_write(tmp_path, data))
    problems = cross_validate(ledger, roadmap=_roadmap(), context=_context())
    assert any("not_in_the_ledger" in p for p in problems)


def test_a_kernel_row_may_not_be_claimed_twice(tmp_path):
    data = _ledger()
    data["parts"][1]["kernels"] = ["alpha_kernel", "beta_kernel"]
    ledger = load_ledger(_write(tmp_path, data))
    problems = cross_validate(ledger, roadmap=_roadmap(), context=_context())
    assert any("claimed by both" in p for p in problems)


def test_unclaimed_hot_rows_must_become_parts(tmp_path):
    ledger = load_ledger(_write(tmp_path, _ledger()))
    context = _context(
        kernels={
            "kernels": [
                _kernel_row("alpha_kernel", 44.4444),
                _kernel_row("beta_kernel", 22.2222),
                _kernel_row("orphan_kernel", 9.0),
            ]
        }
    )
    problems = cross_validate(ledger, roadmap=_roadmap(), context=context, min_share_pct=0.5)
    assert any("orphan_kernel" in p for p in problems)


def test_dispositions_must_name_real_roadmap_items(tmp_path):
    data = _ledger()
    data["parts"][0]["dispositions"] = [_disposition(item="opt-999")]
    ledger = load_ledger(_write(tmp_path, data))
    problems = cross_validate(ledger, roadmap=_roadmap("opt-004"), context=_context())
    assert len(problems) == 1
    assert "opt-999" in problems[0]
    assert cross_validate(ledger, roadmap=_roadmap("opt-999"), context=_context()) == []


def test_open_time_requires_a_live_item_naming_the_part(tmp_path):
    data = _ledger()
    data["parts"][0]["partition"] = {
        "closed_ms": 0.0,
        "attributed_ms": 0.0,
        "open_ms": 2.0,
        "unexplained_ms": 0.0,
    }
    ledger = load_ledger(_write(tmp_path, data))
    problems = cross_validate(ledger, roadmap=_roadmap("opt-001"), context=_context())
    assert len(problems) == 1
    assert "open_ms" in problems[0]
    live = _roadmap(("opt-001", "pending", ["alpha:bf16"]))
    assert cross_validate(ledger, roadmap=live, context=_context()) == []


def test_closed_time_requires_an_accepted_item_naming_the_part(tmp_path):
    data = _ledger()
    data["parts"][0]["partition"] = {
        "closed_ms": 2.0,
        "attributed_ms": 0.0,
        "open_ms": 0.0,
        "unexplained_ms": 0.0,
    }
    ledger = load_ledger(_write(tmp_path, data))
    pending = _roadmap(("opt-001", "pending", ["alpha:bf16"]))
    assert any(
        "closed_ms" in p for p in cross_validate(ledger, roadmap=pending, context=_context())
    )
    accepted = _roadmap(("opt-001", "accepted", ["alpha:bf16"]))
    assert cross_validate(ledger, roadmap=accepted, context=_context()) == []


# --------------------------------------------------------- attribution bases


def _attributed(basis: str, dispositions: list[dict] | None = None) -> dict:
    data = _ledger()
    part = data["parts"][0]
    part["partition"] = {
        "closed_ms": 0.0,
        "attributed_ms": 2.0,
        "open_ms": 0.0,
        "unexplained_ms": 0.0,
    }
    part["attribution"] = {"attributed_ms": 2.0, "basis": basis, "note": "retired"}
    if dispositions is not None:
        part["dispositions"] = dispositions
    return data


def test_one_failed_lever_never_attributes_a_part(tmp_path):
    # The distinction that keeps the ledger honest: a failed item closes a
    # lever, not a part. One failure is an anecdote.
    ledger = load_ledger(_write(tmp_path, _attributed("convergent-levers", [_disposition()])))
    problems = cross_validate(ledger, roadmap=_roadmap("opt-004"), context=_context())
    assert any("convergent-levers" in p for p in problems)


def test_two_distinct_levers_establish_convergence(tmp_path):
    dispositions = [
        _disposition(
            item="opt-002", lever="glue-chain-fusion", gap_implication="applied-but-no-gain"
        ),
        _disposition(
            item="opt-004",
            lever="launch-geometry-tuning",
            gap_implication="mechanism-already-present",
        ),
    ]
    ledger = load_ledger(_write(tmp_path, _attributed("convergent-levers", dispositions)))
    roadmap = _roadmap("opt-002", "opt-004")
    assert cross_validate(ledger, roadmap=roadmap, context=_context()) == []


def test_the_same_lever_twice_is_not_convergence(tmp_path):
    dispositions = [
        _disposition(
            item="opt-002", lever="glue-chain-fusion", gap_implication="applied-but-no-gain"
        ),
        _disposition(
            item="opt-004", lever="glue-chain-fusion", gap_implication="applied-but-no-gain"
        ),
    ]
    ledger = load_ledger(_write(tmp_path, _attributed("convergent-levers", dispositions)))
    problems = cross_validate(ledger, roadmap=_roadmap("opt-002", "opt-004"), context=_context())
    assert any("convergent-levers" in p for p in problems)


@pytest.mark.parametrize("implication", ["change-not-live", "blocked-by-constraint"])
def test_untested_mechanisms_never_count_toward_convergence(tmp_path, implication):
    # In both, the mechanism was never actually tested against the part,
    # so neither bounds its headroom.
    dispositions = [
        _disposition(item="opt-002", lever="glue-chain-fusion", gap_implication=implication),
        _disposition(item="opt-004", lever="launch-geometry-tuning", gap_implication=implication),
    ]
    ledger = load_ledger(_write(tmp_path, _attributed("convergent-levers", dispositions)))
    problems = cross_validate(ledger, roadmap=_roadmap("opt-002", "opt-004"), context=_context())
    assert any("convergent-levers" in p for p in problems)


def test_kernel_ledger_exhaustive_basis_is_checked_against_the_ledger(tmp_path):
    ledger = load_ledger(_write(tmp_path, _attributed("kernel-ledger-exhaustive")))
    # All four questions dismissed on the joined row -> the proof holds.
    assert cross_validate(ledger, roadmap=_roadmap(), context=_context()) == []
    context = _context(
        kernels={
            "kernels": [
                _kernel_row("alpha_kernel", 44.4444, dismissed=False),
                _kernel_row("beta_kernel", 22.2222),
            ]
        }
    )
    problems = cross_validate(ledger, roadmap=_roadmap(), context=context)
    assert any("exhaustiveness proof does not hold" in p for p in problems)


# --------------------------------------------------------- previous-round drift


def _next_round(tmp_path, data) -> dict:
    """Load ``data`` as the following round's ledger (a separate file)."""
    directory = tmp_path / "next"
    directory.mkdir(exist_ok=True)
    return load_ledger(_write(directory, data))


def test_a_moved_ceiling_requires_a_model_revision(tmp_path):
    # The ceiling is a bound, not a description: it never drifts toward
    # what was measured.
    previous = load_ledger(_write(tmp_path, _ledger()))
    data = _ledger()
    data["parts"][0]["at"][512] = {"measured_ms": 4.0, "sol_ms": 1.0, "gap_ms": 3.0}
    data["parts"][0]["partition"]["unexplained_ms"] = 3.0
    ledger = _next_round(tmp_path, data)
    problems = cross_validate(ledger, roadmap=_roadmap(), context=_context(), previous=previous)
    assert any("model_revisions" in p for p in problems)


def test_a_vanished_part_requires_a_lifecycle_entry(tmp_path):
    previous = load_ledger(_write(tmp_path, _ledger()))
    data = _ledger()
    data["parts"] = [data["parts"][0]]
    problems = cross_validate(
        _next_round(tmp_path, data),
        roadmap=_roadmap(),
        context=_context(),
        previous=previous,
    )
    assert any("beta:fp8" in p and "part_lifecycle" in p for p in problems)
    data["part_lifecycle"] = [
        {
            "round": 2,
            "part": "beta:fp8",
            "event": "eliminated",
            "by": "opt-001",
            "measured_ms": 2.0,
            "sol_ms": 1.4,
        }
    ]
    # An eliminated part takes its kernels with it — the row is gone from
    # the fresh kernel ledger too, so nothing is left unclaimed.
    eliminated = _context(kernels={"kernels": [_kernel_row("alpha_kernel", 44.4444)]})
    assert (
        cross_validate(
            _next_round(tmp_path, data),
            roadmap=_roadmap("opt-001"),
            context=eliminated,
            previous=previous,
        )
        == []
    )


# ------------------------------------------------------------ operating point


def test_bracket_must_match_the_scored_regime(tmp_path):
    ledger = load_ledger(_write(tmp_path, _ledger()))
    problems = cross_validate(
        ledger, roadmap=_roadmap(), context=_context(), focus_points=[128, 256]
    )
    assert any("brackets to [128, 256]" in p for p in problems)


def test_single_point_ledger_is_accepted_but_reported(tmp_path):
    # A part's gap is not concurrency-invariant: one endpoint ranks a
    # concurrency-localized win identically everywhere, and wrongly.
    data = _ledger()
    data["operating_point"]["concurrency"] = [512]
    for part in data["parts"]:
        part["at"] = {512: part["at"][512]}
        part["sensitivity"] = None
    ledger = load_ledger(_write(tmp_path, data))
    problems = cross_validate(ledger, roadmap=_roadmap(), context=_context())
    assert len(problems) == 1
    assert "single point" in problems[0]


def test_a_part_missing_a_bracketing_point_is_reported(tmp_path):
    data = _ledger()
    data["parts"][0]["at"] = {512: data["parts"][0]["at"][512]}
    ledger = load_ledger(_write(tmp_path, data))
    problems = cross_validate(ledger, roadmap=_roadmap(), context=_context())
    assert any("missing [64]" in p for p in problems)


# ---------------------------------------------------------------- mutators


def test_append_dispositions_writes_only_the_named_parts(tmp_path):
    path = _write(tmp_path, _ledger())
    written = append_dispositions(
        path,
        round_no=3,
        item_id="opt-004",
        outcome="failed",
        gap_implication="mechanism-inapplicable",
        lever="launch-geometry-tuning",
        note="gated on T == 4; this deployment runs T == 3",
        evidence="rounds/round_3/item_1_opt-004/attempt_1/evaluation.md",
        parts=["alpha:bf16", "not-a-part"],
    )
    assert written == ["alpha:bf16"]
    data = yaml.safe_load(open(path, encoding="utf-8"))
    (entry,) = data["parts"][0]["dispositions"]
    assert entry["item"] == "opt-004"
    assert entry["round"] == 3
    assert entry["gap_implication"] == "mechanism-inapplicable"
    assert "dispositions" not in data["parts"][1]


def test_append_dispositions_is_idempotent_across_a_resume(tmp_path):
    path = _write(tmp_path, _ledger())
    kwargs = dict(
        round_no=3,
        item_id="opt-004",
        outcome="failed",
        gap_implication="applied-but-no-gain",
        lever="launch-geometry-tuning",
        note="it ran and the kernel shrank; the part did not get faster",
        evidence="rounds/round_3/...",
        parts=["alpha:bf16"],
    )
    append_dispositions(path, **kwargs)
    append_dispositions(path, **kwargs)
    data = yaml.safe_load(open(path, encoding="utf-8"))
    assert len(data["parts"][0]["dispositions"]) == 1


def test_append_dispositions_rejects_an_off_enum_value(tmp_path):
    path = _write(tmp_path, _ledger())
    with pytest.raises(HeadroomLedgerError, match="gap_implication"):
        append_dispositions(
            path,
            round_no=1,
            item_id="opt-001",
            outcome="failed",
            gap_implication="it did not help",
            lever="x",
            note="y",
            evidence="z",
            parts=["alpha:bf16"],
        )


def test_appended_dispositions_survive_a_reload(tmp_path):
    path = _write(tmp_path, _ledger())
    append_dispositions(
        path,
        round_no=2,
        item_id="opt-001",
        outcome="accepted",
        gap_implication="applied-but-no-gain",
        lever="lm-head-sharding",
        note="the targeted kernel shrank as predicted",
        evidence="rounds/round_2/...",
        parts=["alpha:bf16"],
    )
    ledger = load_ledger(path)
    assert ledger["parts"][0]["dispositions"][0]["outcome"] == "accepted"


def test_record_history_snapshots_the_primary_point_once(tmp_path):
    path = _write(tmp_path, _ledger())
    record_history(path, 4)
    record_history(path, 4)
    data = yaml.safe_load(open(path, encoding="utf-8"))
    assert data["parts"][0]["history"] == [{"round": 4, "measured_ms": 4.0, "gap_ms": 2.0}]
    record_history(path, 5)
    assert [
        e["round"] for e in yaml.safe_load(open(path, encoding="utf-8"))["parts"][0]["history"]
    ] == [4, 5]


# ------------------------------------------------------------ report helpers


def test_partition_totals_are_absolute_milliseconds(tmp_path):
    # Never % of SOL: the denominator moves when an accepted item deletes
    # a part, so the ratio changes for reasons unrelated to speed.
    data = _ledger()
    data["part_lifecycle"] = [
        {
            "round": 2,
            "part": "logits_upcast",
            "event": "eliminated",
            "by": "opt-001",
            "measured_ms": 0.9792,
            "sol_ms": 0.2871,
        }
    ]
    ledger = load_ledger(_write(tmp_path, data))
    totals = partition_totals(ledger)
    assert totals["unexplained_ms"] == pytest.approx(2.6)
    assert totals["gap_ms"] == pytest.approx(2.6)
    assert totals["eliminated_ms"] == pytest.approx(0.6921)


def test_unexplained_ranking_is_the_work_queue(tmp_path):
    ledger = load_ledger(_write(tmp_path, _ledger()))
    assert unexplained_ranking(ledger) == [("alpha:bf16", 2.0), ("beta:fp8", 0.6)]


def test_engineering_ranking_uses_the_target_not_the_ceiling(tmp_path):
    data = _ledger()
    # alpha's gap-to-SOL is 2.0 but only 1.0 of it is buildable today;
    # beta has no target, so it does not rank on the engineering axis.
    data["parts"][0]["target"] = _target()
    ledger = load_ledger(_write(tmp_path, data))
    assert engineering_ranking(ledger) == [("alpha:bf16", 1.0)]
    assert unexplained_ranking(ledger)[0] == ("alpha:bf16", 2.0)


def test_primary_concurrency_is_the_highest_bracketing_point(tmp_path):
    ledger = load_ledger(_write(tmp_path, _ledger()))
    assert headroom_ledger.primary_concurrency(ledger) == 512


# ------------------------------------------ closure against the exp17 numbers

# The reference campaign the design was derived from: Qwen3.6-35B-A3B-NVFP4
# on 2xB200, round 4. Region ids, measured/SOL milliseconds and kernel
# shares are transcribed from its artifacts, so this fixture exercises the
# join against numbers a real analyzer produced rather than round ones.
_EXP17_KERNEL_MS = 17.335
_EXP17_STEP_MS = 18.372
_EXP17_PARTS = {
    # part id: (measured_ms, sol_ms, bound, [kernel rows])
    "gdn_state:linear_attn:bf16": (3.1321, 2.13862776, "memory", ["_cached_replay_kernel"]),
    "moe_gemm_gate_up:nvfp4": (1.8052, 0.9326096, "memory", ["moe_gemm_fc1_swiglu_nvfp4"]),
    "attention_kv_read:fp8:decode": (
        1.5100,
        1.100424672,
        "memory",
        ["fmha_paged_kv_Q32Kv128", "fmha_paged_kv_Q8Kv128"],
    ),
    "moe_gemm_down:nvfp4": (0.9248, 0.59706528, "memory", ["moe_gemm_fc2_nvfp4"]),
    "conv_state_update:bf16:b512": (
        0.6974,
        0.21973368,
        "memory",
        ["_causal_conv1d_update_kernel"],
    ),
    "gemm:lm_head:fp4": (0.2187, 0.087827131, "compute", ["lm_head_gemm_fp4"]),
    "kv_block_offset_copy:v2:int32": (
        0.2113,
        0.003401512,
        "memory",
        ["copyBatchBlockOffsetsToDeviceKernel"],
    ),
}
_EXP17_SHARES = {
    "_cached_replay_kernel": 18.068,
    "moe_gemm_fc1_swiglu_nvfp4": 10.414,
    "ar_fusion_allreduce_oneshot_lamport_pattern1": 9.16,
    "fmha_paged_kv_Q32Kv128": 8.037,
    "moe_gemm_fc2_nvfp4": 5.335,
    "_causal_conv1d_update_kernel": 4.023,
    "lm_head_gemm_fp4": 1.262,
    "copyBatchBlockOffsetsToDeviceKernel": 1.219,
    "ar_fusion_allreduce_oneshot_lamport_pattern0": 0.708,
    "fmha_paged_kv_Q8Kv128": 0.674,
}


def _exp17_ledger() -> dict:
    parts = []
    for part_id, (measured, sol, bound, kernels) in _EXP17_PARTS.items():
        gap = round(measured - sol, 9)
        parts.append(
            {
                "id": part_id,
                "source": "analytic",
                "at": {512: {"measured_ms": measured, "sol_ms": sol, "gap_ms": gap}},
                "sensitivity": None,
                "bound": bound,
                "kernels": kernels,
                "partition": {
                    "closed_ms": 0.0,
                    "attributed_ms": 0.0,
                    "open_ms": 0.0,
                    "unexplained_ms": gap,
                },
            }
        )
    # The communication row: sol.json substitutes the exposed time
    # (1.350) for the raw kernel time (1.7106), so the join must close
    # against regions.json, not against the correlation.
    parts.append(
        {
            "id": "comm:allreduce:tp2",
            "source": "analytic",
            "at": {
                512: {
                    "measured_ms": 1.3500,
                    "sol_ms": 0.622155093,
                    "gap_ms": 0.727844907,
                }
            },
            "sensitivity": None,
            "bound": "comm",
            "kernels": [
                "ar_fusion_allreduce_oneshot_lamport_pattern1",
                "ar_fusion_allreduce_oneshot_lamport_pattern0",
            ],
            "partition": {
                "closed_ms": 0.0,
                "attributed_ms": 0.0,
                "open_ms": 0.0,
                "unexplained_ms": 0.727844907,
            },
        }
    )
    modeled = sum(_EXP17_SHARES.values()) / 100 * _EXP17_KERNEL_MS
    return {
        "version": headroom_ledger.HEADROOM_LEDGER_VERSION,
        "operating_point": {
            "concurrency": [512],
            "isl": 1024,
            "osl": 6144,
            "build_sha": "1418149a84",
            "node": "b200-node-a",
            "capture_state": "mixed",
        },
        "timing": {"step_ms": _EXP17_STEP_MS, "kernel_ms": _EXP17_KERNEL_MS},
        "coverage": {
            "modeled_kernel_ms": round(modeled, 4),
            "modeled_pct": 58.9,
            "empirical_kernel_ms": 6.3418,
            "unmodeled_kernel_ms": 0.7835,
            "non_kernel_ms": round(_EXP17_STEP_MS - _EXP17_KERNEL_MS, 4),
            "residual_ms": -0.0007,
        },
        "parts": parts,
    }


def _exp17_context(**overrides) -> LedgerContext:
    regions = {part_id: measured for part_id, (measured, _, _, _) in _EXP17_PARTS.items()}
    # regions.json carries the RAW collective kernel time, not the exposed one.
    regions["comm:allreduce:tp2"] = 1.7106
    defaults = {
        "sol": {"per_op": [{"region": part_id} for part_id in regions]},
        "regions": {"regions": [{"region": r, "measured_ms": m} for r, m in regions.items()]},
        "kernels": {"kernels": [_kernel_row(k, s) for k, s in _EXP17_SHARES.items()]},
    }
    defaults.update(overrides)
    return LedgerContext(**defaults)


def test_exp17_ledger_closes_end_to_end(tmp_path):
    """The join reproduces every analytic part's measured time.

    ``share_pct x kernel_ms`` against ``regions.json`` for all eight
    analytic parts plus the collective — the identity that makes the
    kernel to part mapping machine-verifiable rather than a matter of
    judgement.
    """
    ledger = load_ledger(_write(tmp_path, _exp17_ledger()))
    problems = cross_validate(
        ledger, roadmap=_roadmap(), context=_exp17_context(), focus_points=[512]
    )
    # The single-point note is expected: exp17 captured only c=512.
    assert [p for p in problems if "single point" not in p] == []


def test_exp17_incomplete_join_localizes_the_missing_kernel(tmp_path):
    """An incomplete join does not merely fail — it names what is missing.

    Mapping ``attention_kv_read`` to the Q32 variant alone leaves a
    residual of 0.1168 ms, which is 0.674% of kernel time — the share of
    exactly one ledger row, ``fmha_paged_kv_Q8Kv128`` (the MTP
    draft-token attention variant).
    """
    data = _exp17_ledger()
    for part in data["parts"]:
        if part["id"] == "attention_kv_read:fp8:decode":
            part["kernels"] = ["fmha_paged_kv_Q32Kv128"]
    ledger = load_ledger(_write(tmp_path, data))
    problems = cross_validate(ledger, roadmap=_roadmap(), context=_exp17_context())
    residuals = [p for p in problems if "residual" in p]
    assert len(residuals) == 1
    assert "+0.1168 ms" in residuals[0]
    assert "+0.674%" in residuals[0]
    assert _EXP17_SHARES["fmha_paged_kv_Q8Kv128"] == 0.674


def test_exp17_comm_row_closes_against_regions_not_sol(tmp_path):
    """The collective's raw kernel time is the closure target.

    ``sol_calc.py`` substitutes ``exposed_ms`` (1.350) for communication
    rows, so joining against the correlation's measured column would
    report a phantom 0.36 ms residual on a join that is correct.
    """
    ledger = load_ledger(_write(tmp_path, _exp17_ledger()))
    sol_shaped = _exp17_context(
        regions={"regions": [{"region": "comm:allreduce:tp2", "measured_ms": 1.3500}]}
    )
    problems = cross_validate(ledger, roadmap=_roadmap(), context=sol_shaped)
    assert any("comm:allreduce:tp2" in p and "residual" in p for p in problems)
    assert not any(
        "comm:allreduce:tp2" in p and "residual" in p
        for p in cross_validate(ledger, roadmap=_roadmap(), context=_exp17_context())
    )


def test_filename_is_pinned():
    assert headroom_ledger.HEADROOM_LEDGER_FILENAME == "headroom_ledger.yaml"
    assert copy.deepcopy(headroom_ledger.GAP_IMPLICATIONS) == headroom_ledger.GAP_IMPLICATIONS
