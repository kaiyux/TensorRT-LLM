"""Tests for the perf-optimize ``kernel_ledger.yaml`` schema."""

from __future__ import annotations

import copy

import pytest
import yaml

from agent_flow.workflows.perf_optimize import kernel_ledger
from agent_flow.workflows.perf_optimize.kernel_ledger import (
    LedgerError,
    cross_validate,
    load_ledger,
)

# --------------------------------------------------------------------- helpers


def _row(kernel: str = "gdn_bf16_state", share: float = 60.0, **overrides) -> dict:
    row = {
        "kernel": kernel,
        "full_name": f"void tensorrt_llm::kernels::{kernel}<...>",
        "share_pct": share,
        "model": "iteration",
        "ncu": {
            "duration_us": 41.2,
            "sm_sol_pct": 12.1,
            "mem_sol_pct": 78.5,
            "occupancy_pct": 62.0,
            "bound": "memory",
        },
        "elimination": {
            "disposition": "dismissed",
            "why_it_runs": "state update read by the next layer's gate (NVTX + source)",
            "ref": "mandatory-math: per-step recurrence, no padded or invariant part",
        },
        "faster": {"disposition": "item", "ref": "opt-001"},
        "fusion": {
            "disposition": "dismissed",
            "neighbors": "rmsnorm -> THIS -> fp8_quant (cuda_gpu_trace, step 120)",
            "ref": "multi-consumer-pinned: intermediate feeds residual + norm (cuda_gpu_trace)",
        },
        "overlap": {
            "disposition": "dismissed",
            "concurrent_with": "moe_gemm (serialized on stream 7, cuda_gpu_trace step 120)",
            "ref": "no-independent-partner: moe_gemm consumes this output (cuda_gpu_trace)",
        },
    }
    row.update(overrides)
    return row


def _model(**overrides) -> dict:
    model = {
        "id": "iteration",
        "scope": "iteration",
        "operating_point": {"concurrency": 512, "hardware": "B200", "timing": "wall-clock"},
        "derivation": "critical-path mandatory traffic / independently measured bandwidth",
        "assumptions": ["state reads and writes each occur once"],
        "evidence": ["analysis/traffic.md: source byte count and bandwidth measurement"],
        "predicted_ms": 2.0,
        "measured_ms": 3.0,
        "measurement_evidence": ["analysis/nsys_analysis/regions.json: iteration interval"],
        "unexplained": "1 ms residual; memory-level parallelism is not yet explained",
        "next_test": "measure outstanding memory requests at the scored shape",
    }
    model.update(overrides)
    return model


def _ledger(**overrides) -> dict:
    data = {
        "version": kernel_ledger.LEDGER_VERSION,
        "source": "rounds/round_1/analysis/nsys_stats.txt",
        "models": [_model()],
        "model_revisions": [],
        "coverage": {
            "enumerated_share_pct": 96.0,
            "other_share_pct": 4.0,
            "min_share_pct": 0.5,
            "gpu_busy_pct": 82.4,
        },
        "kernels": [
            _row(),
            _row(
                "attention_fmha",
                36.0,
                faster={
                    "disposition": "dismissed",
                    "ref": "at-sol-floor: mem SOL 91% (ncu_details_pass1.txt)",
                },
            ),
        ],
    }
    data.update(overrides)
    return data


def _write(tmp_path, data) -> str:
    path = tmp_path / "kernel_ledger.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(path)


def _roadmap(*item_ids: str) -> dict:
    return {"items": [{"id": item_id, "status": "pending"} for item_id in item_ids]}


# ------------------------------------------------------------------ load_ledger


def test_valid_ledger_loads(tmp_path):
    data = load_ledger(_write(tmp_path, _ledger()))
    assert [row["kernel"] for row in data["kernels"]] == ["gdn_bf16_state", "attention_fmha"]
    assert data["coverage"]["enumerated_share_pct"] == pytest.approx(96.0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(LedgerError, match="not found"):
        load_ledger(tmp_path / "kernel_ledger.yaml")


def test_non_mapping_top_level_raises(tmp_path):
    path = tmp_path / "kernel_ledger.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="mapping at the top level"):
        load_ledger(path)


def test_bound_shorthand_is_normalized(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["bound"] = "memory-bound"
    ledger["kernels"][1]["ncu"]["bound"] = "SM"
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["ncu"]["bound"] == "memory"
    assert data["kernels"][1]["ncu"]["bound"] == "compute"


def test_unknown_bound_rejected(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["bound"] = "quantum"
    with pytest.raises(LedgerError, match="ncu.bound"):
        load_ledger(_write(tmp_path, ledger))


def test_comm_bound_class_is_accepted(tmp_path):
    # Collectives are never put under ncu (kernel replay deadlocks them), so an
    # allreduce row is dispositioned from nsys evidence and reports `comm`.
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = {
        "duration_us": 88.0,
        "sm_sol_pct": None,
        "mem_sol_pct": None,
        "occupancy_pct": None,
        "bound": "comm",
        "note": "collective — not captured under ncu (kernel replay deadlocks NCCL)",
    }
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["ncu"]["bound"] == "comm"


@pytest.mark.parametrize("shorthand", ["communication", "comm-bound", "NCCL", "Collective"])
def test_comm_bound_shorthand_is_normalized(tmp_path, shorthand):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["bound"] = shorthand
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["ncu"]["bound"] == "comm"


def test_ncu_degrade_string_is_allowed(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "unavailable: no pass captured this stem (3 passes exhausted)"
    ledger["kernels"][0]["bound"] = "memory"
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["ncu"].startswith("unavailable")
    assert data["kernels"][0]["bound"] == "memory"


def test_ncu_degrade_string_still_owes_a_row_level_bound(tmp_path):
    # A collective never goes under ncu, so the row-level `bound` beside the
    # degrade string is the only bound class it will ever have.
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "unavailable: collective — replay deadlocks the ranks"
    with pytest.raises(LedgerError, match=r"kernels\[0\].bound"):
        load_ledger(_write(tmp_path, ledger))


def test_collective_row_records_comm_beside_the_degrade_string(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "unavailable: collective — replay deadlocks the ranks"
    ledger["kernels"][0]["bound"] = "comm"
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["bound"] == "comm"


@pytest.mark.parametrize("shorthand", ["communication", "comm-bound", "NCCL", "Collective"])
def test_row_level_bound_shorthand_is_normalized(tmp_path, shorthand):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "unavailable: collective — replay deadlocks the ranks"
    ledger["kernels"][0]["bound"] = shorthand
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["bound"] == "comm"


def test_row_level_bound_must_be_a_known_class(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "unavailable: no pass captured this stem"
    ledger["kernels"][0]["bound"] = "quantum"
    with pytest.raises(LedgerError, match=r"kernels\[0\].bound"):
        load_ledger(_write(tmp_path, ledger))


def test_empty_ncu_degrade_string_rejected(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"] = "  "
    with pytest.raises(LedgerError, match="ncu.*non-empty"):
        load_ledger(_write(tmp_path, ledger))


def test_ncu_metrics_must_be_numbers(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["mem_sol_pct"] = "high"
    with pytest.raises(LedgerError, match="ncu.mem_sol_pct"):
        load_ledger(_write(tmp_path, ledger))


def test_null_ncu_metric_allowed_with_a_note(tmp_path):
    # A partial capture keeps what it measured; the note says what it lost.
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["sm_sol_pct"] = None
    ledger["kernels"][0]["ncu"]["occupancy_pct"] = None
    ledger["kernels"][0]["ncu"]["note"] = "SOL section empty — replay stalled, 3 passes"
    data = load_ledger(_write(tmp_path, ledger))
    assert data["kernels"][0]["ncu"]["sm_sol_pct"] is None
    assert data["kernels"][0]["ncu"]["duration_us"] == pytest.approx(41.2)


def test_null_ncu_metric_without_a_note_rejected(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["mem_sol_pct"] = None
    with pytest.raises(LedgerError, match="mem_sol_pct.*'note'"):
        load_ledger(_write(tmp_path, ledger))


def test_blank_note_does_not_excuse_a_null_metric(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["duration_us"] = None
    ledger["kernels"][0]["ncu"]["note"] = "   "
    with pytest.raises(LedgerError, match="duration_us.*'note'"):
        load_ledger(_write(tmp_path, ledger))


def test_absent_ncu_metric_is_treated_as_null(tmp_path):
    ledger = _ledger()
    del ledger["kernels"][0]["ncu"]["occupancy_pct"]
    with pytest.raises(LedgerError, match="occupancy_pct.*'note'"):
        load_ledger(_write(tmp_path, ledger))


def test_note_does_not_excuse_a_missing_bound(tmp_path):
    # `bound` is required on a partial capture too — null a metric so this
    # crosses that branch rather than the all-populated one.
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["sm_sol_pct"] = None
    ledger["kernels"][0]["ncu"]["bound"] = None
    ledger["kernels"][0]["ncu"]["note"] = "SOL sections came back empty"
    with pytest.raises(LedgerError, match="ncu.bound"):
        load_ledger(_write(tmp_path, ledger))


def test_note_does_not_excuse_a_non_numeric_metric(tmp_path):
    # The note licenses "not measured", not "measured, badly typed".
    ledger = _ledger()
    ledger["kernels"][0]["ncu"]["mem_sol_pct"] = "high"
    ledger["kernels"][0]["ncu"]["note"] = "ncu printed a bare string"
    with pytest.raises(LedgerError, match="ncu.mem_sol_pct"):
        load_ledger(_write(tmp_path, ledger))


@pytest.mark.parametrize("question", kernel_ledger.QUESTIONS)
def test_every_question_required_per_row(tmp_path, question):
    ledger = _ledger()
    del ledger["kernels"][0][question]
    with pytest.raises(LedgerError, match="answers all four questions"):
        load_ledger(_write(tmp_path, ledger))


def test_disposition_enum_is_exact(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["faster"]["disposition"] = "maybe"
    with pytest.raises(LedgerError, match="faster.disposition"):
        load_ledger(_write(tmp_path, ledger))


def test_empty_ref_rejected(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["faster"]["ref"] = ""
    with pytest.raises(LedgerError, match="faster.ref"):
        load_ledger(_write(tmp_path, ledger))


def test_fusion_dismissal_requires_observed_neighbors(tmp_path):
    ledger = _ledger()  # the default fusion block is a dismissal
    del ledger["kernels"][0]["fusion"]["neighbors"]
    with pytest.raises(LedgerError, match="fusion.neighbors"):
        load_ledger(_write(tmp_path, ledger))


def test_fusion_item_does_not_require_neighbors(tmp_path):
    # A promoted fusion keeps its adjacency in the roadmap item `ref` names;
    # only the dismissal owes the evidence here.
    ledger = _ledger()
    ledger["kernels"][0]["fusion"] = {"disposition": "item", "ref": "opt-001"}
    data = load_ledger(_write(tmp_path, ledger))
    assert "neighbors" not in data["kernels"][0]["fusion"]


def test_overlap_item_does_not_require_the_partner(tmp_path):
    # Same asymmetry as fusion: a promoted overlap names its pair in the
    # roadmap item, so only the dismissal owes `concurrent_with` here.
    ledger = _ledger()
    ledger["kernels"][0]["overlap"] = {"disposition": "item", "ref": "opt-001"}
    data = load_ledger(_write(tmp_path, ledger))
    assert "concurrent_with" not in data["kernels"][0]["overlap"]


def test_elimination_requires_why_the_kernel_runs(tmp_path):
    # Question 1's verdict rests on what consumes the output (or the
    # guard that selected this path) — its neighbors/concurrent_with
    # analogue, so "it is needed" cannot be asserted bare.
    ledger = _ledger()
    del ledger["kernels"][0]["elimination"]["why_it_runs"]
    with pytest.raises(LedgerError, match="elimination.why_it_runs"):
        load_ledger(_write(tmp_path, ledger))


def test_elimination_disposition_enum_is_exact(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["elimination"]["disposition"] = "probably"
    with pytest.raises(LedgerError, match="elimination.disposition"):
        load_ledger(_write(tmp_path, ledger))


def test_elimination_is_asked_before_the_other_three(tmp_path):
    # Order matters for the prompt and the report: a `yes` here moots the
    # rest and recovers the whole share, so it leads.
    assert kernel_ledger.QUESTIONS[0] == "elimination"
    assert kernel_ledger.QUESTIONS == ("elimination", "faster", "fusion", "overlap")


def test_overlap_requires_the_candidate_partner(tmp_path):
    # Question 3's verdict rests on an observed pair, not a guess — the
    # `concurrent_with` field is its `fusion.neighbors` analogue.
    ledger = _ledger()
    del ledger["kernels"][0]["overlap"]["concurrent_with"]
    with pytest.raises(LedgerError, match="overlap.concurrent_with"):
        load_ledger(_write(tmp_path, ledger))


def test_overlap_disposition_enum_is_exact(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["overlap"]["disposition"] = "someday"
    with pytest.raises(LedgerError, match="overlap.disposition"):
        load_ledger(_write(tmp_path, ledger))


def test_gpu_busy_pct_is_required(tmp_path):
    # Without it a `below-materiality` dismissal cannot be converted from
    # a share of GPU time into the share of wall clock the noise floor
    # judges — the arithmetic would overstate every candidate by 1/busy.
    ledger = _ledger()
    del ledger["coverage"]["gpu_busy_pct"]
    with pytest.raises(LedgerError, match="gpu_busy_pct"):
        load_ledger(_write(tmp_path, ledger))


@pytest.mark.parametrize("bad", [0, -1, 101, "high", True, None])
def test_gpu_busy_pct_must_be_a_percentage(tmp_path, bad):
    ledger = _ledger()
    ledger["coverage"]["gpu_busy_pct"] = bad
    with pytest.raises(LedgerError, match=r"gpu_busy_pct.*\(0, 100\]"):
        load_ledger(_write(tmp_path, ledger))


def test_coverage_error_names_every_required_field(tmp_path):
    ledger = _ledger()
    ledger["coverage"] = "nope"
    with pytest.raises(LedgerError, match="gpu_busy_pct"):
        load_ledger(_write(tmp_path, ledger))


def test_duplicate_kernel_keys_rejected(tmp_path):
    ledger = _ledger()
    ledger["kernels"].append(copy.deepcopy(ledger["kernels"][0]))
    with pytest.raises(LedgerError, match="duplicates"):
        load_ledger(_write(tmp_path, ledger))


def test_coverage_buckets_must_account_for_100(tmp_path):
    # Kernels dropped from the ledger must be rolled into other_share_pct.
    ledger = _ledger()
    ledger["coverage"]["enumerated_share_pct"] = 80.0
    ledger["coverage"]["other_share_pct"] = 4.0
    with pytest.raises(LedgerError, match="~100%"):
        load_ledger(_write(tmp_path, ledger))


def test_coverage_sum_tolerates_rounding(tmp_path):
    ledger = _ledger()
    ledger["coverage"]["enumerated_share_pct"] = 96.4
    ledger["coverage"]["other_share_pct"] = 4.8  # 101.2 — within tolerance
    load_ledger(_write(tmp_path, ledger))


@pytest.mark.parametrize("declared,shares", [(96.0, [1.0]), (94.0, [60.0, 36.0])])
def test_coverage_total_must_match_enumerated_rows(tmp_path, declared, shares):
    ledger = _ledger()
    ledger["coverage"].update(enumerated_share_pct=declared, other_share_pct=100 - declared)
    ledger["kernels"] = [_row(f"kernel-{index}", share) for index, share in enumerate(shares)]
    with pytest.raises(LedgerError, match="must match the sum"):
        load_ledger(_write(tmp_path, ledger))


def test_coverage_gate_uses_rows_even_with_rounding_in_declared_total(tmp_path):
    ledger = _ledger()
    ledger["coverage"].update(enumerated_share_pct=94.6, other_share_pct=5.4)
    ledger["kernels"][1]["share_pct"] = 34.2
    loaded = load_ledger(_write(tmp_path, ledger))
    problems = cross_validate(loaded, _roadmap("opt-001"), coverage_target_pct=95.0)
    assert len(problems) == 1
    assert "coverage_target_pct" in problems[0]


def test_empty_kernels_list_rejected(tmp_path):
    with pytest.raises(LedgerError, match="'kernels' must be a non-empty list"):
        load_ledger(_write(tmp_path, _ledger(kernels=[])))


def test_errors_are_batched(tmp_path):
    ledger = _ledger(version=99, source="")
    ledger["kernels"][0]["share_pct"] = -1
    with pytest.raises(LedgerError) as excinfo:
        load_ledger(_write(tmp_path, ledger))
    message = str(excinfo.value)
    assert "'version'" in message
    assert "'source'" in message
    assert "share_pct" in message


# --------------------------------------------------------------- cross_validate


def test_item_refs_must_resolve_to_roadmap_ids(tmp_path):
    ledger = load_ledger(_write(tmp_path, _ledger()))
    problems = cross_validate(ledger, _roadmap("opt-999"), coverage_target_pct=95.0)
    assert len(problems) == 1
    assert "faster.ref" in problems[0]
    assert "opt-001" in problems[0]


def test_elimination_item_refs_must_resolve_to_roadmap_ids(tmp_path):
    ledger = _ledger()
    ledger["kernels"][0]["elimination"] = {
        "disposition": "item",
        "why_it_runs": "unfused fallback: is_fused=False guard (modeling_x.py:412)",
        "ref": "opt-007",
    }
    loaded = load_ledger(_write(tmp_path, ledger))
    problems = cross_validate(loaded, _roadmap("opt-001"), coverage_target_pct=95.0)
    assert len(problems) == 1
    assert "elimination.ref" in problems[0]
    assert cross_validate(loaded, _roadmap("opt-001", "opt-007"), 95.0) == []


def test_overlap_item_refs_must_resolve_to_roadmap_ids(tmp_path):
    # An overlap answer is only an answer if the item really landed in
    # the plan — same bar as faster/fusion.
    ledger = _ledger()
    ledger["kernels"][0]["overlap"] = {
        "disposition": "item",
        "concurrent_with": "moe_gemm (data-independent; disjoint outputs)",
        "ref": "opt-004",
    }
    loaded = load_ledger(_write(tmp_path, ledger))
    problems = cross_validate(loaded, _roadmap("opt-001"), coverage_target_pct=95.0)
    assert len(problems) == 1
    assert "overlap.ref" in problems[0]
    assert cross_validate(loaded, _roadmap("opt-001", "opt-004"), 95.0) == []


def test_one_item_may_answer_two_questions_on_one_row(tmp_path):
    # A pair can be fused or overlapped — the same item id legitimately
    # appears in both cells as alternative realizations.
    ledger = _ledger()
    ledger["kernels"][0]["fusion"]["disposition"] = "item"
    ledger["kernels"][0]["fusion"]["ref"] = "opt-004"
    ledger["kernels"][0]["overlap"]["disposition"] = "item"
    ledger["kernels"][0]["overlap"]["ref"] = "opt-004"
    loaded = load_ledger(_write(tmp_path, ledger))
    assert cross_validate(loaded, _roadmap("opt-001", "opt-004"), 95.0) == []


def test_refs_to_terminal_items_are_considered(tmp_path):
    # An accepted/failed item still proves the possibility was considered.
    ledger = load_ledger(_write(tmp_path, _ledger()))
    roadmap = {"items": [{"id": "opt-001", "status": "failed"}]}
    assert cross_validate(ledger, roadmap, coverage_target_pct=95.0) == []


def test_coverage_below_target_is_a_problem(tmp_path):
    ledger = _ledger()
    ledger["coverage"]["enumerated_share_pct"] = 90.0
    ledger["coverage"]["other_share_pct"] = 10.0
    ledger["kernels"][1]["share_pct"] = 30.0
    loaded = load_ledger(_write(tmp_path, ledger))
    problems = cross_validate(loaded, _roadmap("opt-001"), coverage_target_pct=95.0)
    assert len(problems) == 1
    assert "coverage_target_pct" in problems[0]


def test_coverage_target_tolerates_rounding(tmp_path):
    ledger = _ledger()
    ledger["coverage"]["enumerated_share_pct"] = 94.7
    ledger["coverage"]["other_share_pct"] = 5.3
    ledger["kernels"][1]["share_pct"] = 34.7
    loaded = load_ledger(_write(tmp_path, ledger))
    assert cross_validate(loaded, _roadmap("opt-001"), coverage_target_pct=95.0) == []


def test_clean_ledger_cross_validates_clean(tmp_path):
    ledger = load_ledger(_write(tmp_path, _ledger()))
    assert cross_validate(ledger, _roadmap("opt-001"), coverage_target_pct=95.0) == []


def test_filename_constant_matches_contract():
    assert kernel_ledger.LEDGER_FILENAME == "kernel_ledger.yaml"


# --------------------------------------------------------------- model evidence


def test_dismissal_tags_are_published_for_reuse(tmp_path):
    # Prompts and reports share the same dismissal vocabulary.
    for tag in ("multi-consumer-pinned", "phase-boundary", "needs-rebuild", "at-sol-floor"):
        assert tag in kernel_ledger.DISMISSAL_TAGS
    assert len(set(kernel_ledger.DISMISSAL_TAGS)) == len(kernel_ledger.DISMISSAL_TAGS)


def test_dismissal_tag_splits_the_leading_tag(tmp_path):
    assert kernel_ledger.dismissal_tag("mandatory-math") == "mandatory-math"
    assert kernel_ledger.dismissal_tag("fast-path-blocked: head_dim guard") == "fast-path-blocked"
    assert kernel_ledger.dismissal_tag("  at-sol-floor : mem SOL 89%") == "at-sol-floor"


def test_legacy_kernel_only_ledger_requires_analyzer_refresh(tmp_path):
    with pytest.raises(LedgerError, match="'version' must be 2"):
        load_ledger(_write(tmp_path, _ledger(version=1)))


@pytest.mark.parametrize("field", ["models", "model_revisions"])
def test_unified_model_fields_are_required(tmp_path, field):
    data = _ledger()
    del data[field]
    with pytest.raises(LedgerError, match=field):
        load_ledger(_write(tmp_path, data))


def test_every_kernel_must_reference_a_known_model(tmp_path):
    data = _ledger()
    data["kernels"][0]["model"] = "not-a-model"
    with pytest.raises(LedgerError, match="references unknown model 'not-a-model'"):
        load_ledger(_write(tmp_path, data))
    del data["kernels"][0]["model"]
    with pytest.raises(LedgerError, match="model.*non-empty model id"):
        load_ledger(_write(tmp_path, data))


@pytest.mark.parametrize("scope", ["region", "iteration"])
def test_multiple_kernels_can_share_one_model(tmp_path, scope):
    data = _ledger(models=[_model(scope=scope)])
    loaded = load_ledger(_write(tmp_path, data))
    assert len(loaded["models"]) == 1
    assert {row["model"] for row in loaded["kernels"]} == {"iteration"}
    assert cross_validate(loaded, _roadmap("opt-001"), 95.0) == []


def test_kernel_scope_describes_one_kernel(tmp_path):
    data = _ledger(models=[_model(scope="kernel")])
    with pytest.raises(LedgerError, match="exactly one kernel row"):
        load_ledger(_write(tmp_path, data))
    data["models"].append(_model(id="attention", scope="kernel"))
    data["kernels"][1]["model"] = "attention"
    load_ledger(_write(tmp_path, data))


@pytest.mark.parametrize(
    "field,bad",
    [
        ("id", ""),
        ("scope", "part"),
        ("operating_point", {}),
        ("operating_point", "same shape"),
        ("derivation", " "),
        ("assumptions", "none"),
        ("assumptions", [""]),
        ("evidence", []),
        ("evidence", [" "]),
        ("measurement_evidence", []),
        ("measurement_evidence", "nsys"),
    ],
)
def test_model_requires_derivation_conditions_and_evidence(tmp_path, field, bad):
    data = _ledger(models=[_model(**{field: bad})])
    with pytest.raises(LedgerError, match=field):
        load_ledger(_write(tmp_path, data))


def test_duplicate_model_ids_rejected(tmp_path):
    with pytest.raises(LedgerError, match="id.*duplicates"):
        load_ledger(_write(tmp_path, _ledger(models=[_model(), _model()])))


@pytest.mark.parametrize("field", ["predicted_ms", "measured_ms"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -1, True, "2.0"])
def test_model_times_are_finite_nonnegative_numbers(tmp_path, field, bad):
    with pytest.raises(LedgerError, match=f"{field}.*finite number"):
        load_ledger(_write(tmp_path, _ledger(models=[_model(**{field: bad})])))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "holder,field",
    [("coverage", "enumerated_share_pct"), ("kernel", "share_pct"), ("ncu", "duration_us")],
)
def test_nonfinite_kernel_metrics_and_coverage_rejected(tmp_path, holder, field, bad):
    data = _ledger()
    target = {
        "coverage": data["coverage"],
        "kernel": data["kernels"][0],
        "ncu": data["kernels"][0]["ncu"],
    }[holder]
    target[field] = bad
    with pytest.raises(LedgerError, match=field):
        load_ledger(_write(tmp_path, data))


@pytest.mark.parametrize("field", ["predicted_ms", "measured_ms"])
def test_null_time_keeps_its_uncertainty_and_next_experiment(tmp_path, field):
    model = _model(**{field: None})
    model["unexplained"] = f"{field} unavailable: no bandwidth capture at this shape"
    model["next_test"] = "capture a matched bandwidth experiment"
    if field == "measured_ms":
        model["measurement_evidence"] = []
    load_ledger(_write(tmp_path, _ledger(models=[model])))
    for required in ("unexplained", "next_test"):
        invalid = copy.deepcopy(model)
        invalid[required] = ""
        with pytest.raises(LedgerError, match=f"{required}.*non-empty"):
            load_ledger(_write(tmp_path, _ledger(models=[invalid])))


@pytest.mark.parametrize("predicted,measured", [(2.0, 3.0), (3.0, 2.0)])
def test_positive_and_negative_residuals_remain_explicit(tmp_path, predicted, measured):
    model = _model(predicted_ms=predicted, measured_ms=measured)
    loaded = load_ledger(_write(tmp_path, _ledger(models=[model])))
    assert (
        loaded["models"][0]["measured_ms"] - loaded["models"][0]["predicted_ms"]
        == measured - predicted
    )
    model["unexplained"] = ""
    with pytest.raises(LedgerError, match="nonzero measured-minus-predicted residual"):
        load_ledger(_write(tmp_path, _ledger(models=[model])))


def test_matching_model_and_measurement_can_have_no_open_residual(tmp_path):
    model = _model(predicted_ms=3.0, measured_ms=3.0, unexplained="", next_test="")
    load_ledger(_write(tmp_path, _ledger(models=[model])))


def _revision(field="predicted_ms", before=2.0, after=2.5, **overrides) -> dict:
    revision = {
        "round": 2,
        "model": "iteration",
        "reason": "DRAM byte counters establish an additional mandatory read",
        "evidence": ["rounds/round_2/analysis/ncu.txt: DRAM bytes"],
        "changes": {field: {"from": before, "to": after}},
    }
    revision.update(overrides)
    return revision


def _validate_next(tmp_path, previous, current, round_no=2):
    loaded = load_ledger(_write(tmp_path, current))
    return cross_validate(loaded, _roadmap("opt-001"), 95.0, previous=previous, round_no=round_no)


@pytest.mark.parametrize(
    "field,after",
    [
        ("predicted_ms", 2.5),
        ("derivation", "traffic including mandatory reread / measured bandwidth"),
        ("assumptions", ["source and counters show state must be read twice"]),
        ("operating_point", {"concurrency": 256}),
        ("scope", "region"),
    ],
)
def test_theory_changes_require_matching_evidence_backed_revisions(tmp_path, field, after):
    previous = _ledger()
    current = copy.deepcopy(previous)
    before = current["models"][0][field]
    current["models"][0][field] = after
    assert any(
        "model_revisions" in problem for problem in _validate_next(tmp_path, previous, current)
    )
    current["model_revisions"] = [_revision(field, before, after)]
    assert _validate_next(tmp_path, previous, current) == []


@pytest.mark.parametrize("endpoint,bad", [("from", 0.5), ("to", 8.0)])
def test_revision_endpoints_must_match_actual_model_changes(tmp_path, endpoint, bad):
    previous = _ledger()
    current = _ledger(models=[_model(predicted_ms=2.5)], model_revisions=[_revision()])
    current["model_revisions"][0]["changes"]["predicted_ms"][endpoint] = bad
    problems = _validate_next(tmp_path, previous, current)
    assert any(f"'{endpoint}' that does not match" in problem for problem in problems)


def test_multiple_revisions_can_explain_a_chain_of_model_changes(tmp_path):
    previous = _ledger()
    current = _ledger(
        models=[_model(predicted_ms=3.0)],
        model_revisions=[_revision(), _revision(before=2.5, after=3.0)],
    )
    assert _validate_next(tmp_path, previous, current) == []


@pytest.mark.parametrize(
    "field,bad",
    [
        ("round", 0),
        ("round", True),
        ("reason", ""),
        ("evidence", []),
        ("evidence", [""]),
        ("changes", {}),
    ],
)
def test_revisions_require_specific_evidence_and_actual_changes(tmp_path, field, bad):
    revision = _revision(**{field: bad})
    with pytest.raises(LedgerError, match=field):
        load_ledger(_write(tmp_path, _ledger(model_revisions=[revision])))


def test_revision_model_reference_must_resolve(tmp_path):
    with pytest.raises(LedgerError, match="references unknown model"):
        load_ledger(_write(tmp_path, _ledger(model_revisions=[_revision(model="unknown")])))


def test_same_round_requires_new_revision_instead_of_reusing_old_evidence(tmp_path):
    previous = _ledger(models=[_model(predicted_ms=2.5)], model_revisions=[_revision()])
    current = copy.deepcopy(previous)
    current["models"][0]["predicted_ms"] = 3.0
    assert any(
        "without an evidence-backed" in p for p in _validate_next(tmp_path, previous, current)
    )
    current["model_revisions"].append(_revision(before=2.5, after=3.0))
    assert _validate_next(tmp_path, previous, current) == []


@pytest.mark.parametrize("change", ["drop", "rewrite", "reorder"])
def test_revision_history_must_be_preserved(tmp_path, change):
    history = [_revision(), _revision(before=2.5, after=3.0)]
    previous = _ledger(models=[_model(predicted_ms=3.0)], model_revisions=history)
    current = copy.deepcopy(previous)
    if change == "drop":
        current["model_revisions"] = []
    elif change == "rewrite":
        current["model_revisions"][0]["reason"] = "rewritten history"
    else:
        current["model_revisions"].reverse()
    assert any(
        "preserve the complete previous" in p
        for p in _validate_next(tmp_path, previous, current, 3)
    )


def test_stale_and_future_revisions_do_not_explain_current_changes(tmp_path):
    previous = _ledger()
    current = _ledger(models=[_model(predicted_ms=2.5)], model_revisions=[_revision(round=1)])
    assert any("current round" in p for p in _validate_next(tmp_path, previous, current))
    current["model_revisions"][0]["round"] = 3
    assert any("current round" in p for p in _validate_next(tmp_path, previous, current))


@pytest.mark.parametrize("removal", [False, True])
def test_initial_ledger_cannot_invent_revision_history(tmp_path, removal):
    revision = _revision("removed", _model(), None) if removal else _revision(before=0.5, after=4.0)
    loaded = load_ledger(_write(tmp_path, _ledger(model_revisions=[revision])))
    problems = cross_validate(loaded, _roadmap("opt-001"), 95.0, round_no=2)
    assert problems == ["initial ledger must have an empty 'model_revisions' history"]


def test_new_model_establishes_its_evidence_without_revision(tmp_path):
    previous = _ledger()
    current = _ledger(models=[_model(), _model(id="new-region", scope="region")])
    assert _validate_next(tmp_path, previous, current) == []
    current["model_revisions"] = [_revision(model="new-region", before=0.5, after=4.0)]
    problems = _validate_next(tmp_path, previous, current)
    assert len(problems) == 1
    assert "'new-region' requires a model from the previous ledger" in problems[0]


def test_revision_cannot_remove_a_model_absent_from_the_previous_ledger(tmp_path):
    previous = _ledger()
    current = _ledger(
        model_revisions=[
            _revision("removed", _model(id="invented-model"), None, model="invented-model")
        ]
    )
    problems = _validate_next(tmp_path, previous, current)
    assert len(problems) == 1
    assert "'invented-model' requires a model from the previous ledger" in problems[0]


def test_model_removal_preserves_its_fact_based_explanation(tmp_path):
    previous = _ledger()
    current = _ledger(models=[_model(id="fused-region", scope="region")])
    for row in current["kernels"]:
        row["model"] = "fused-region"
    assert any(
        "removed model 'iteration'" in p for p in _validate_next(tmp_path, previous, current)
    )
    current["model_revisions"] = [
        _revision(
            "removed", previous["models"][0], None, reason="accepted fusion changes timing boundary"
        )
    ]
    assert _validate_next(tmp_path, previous, current) == []
    # Later rounds retain the removed model's history even though it has no live row.
    assert _validate_next(tmp_path, current, current, 3) == []


def test_measurement_refresh_keeps_the_model_until_new_facts_revise_it(tmp_path):
    previous = _ledger()
    current = copy.deepcopy(previous)
    current["models"][0]["measured_ms"] = 2.7
    current["models"][0]["measurement_evidence"] = ["rounds/round_2/analysis/regions.json"]
    assert _validate_next(tmp_path, previous, current) == []
    current["models"][0]["measurement_evidence"] = []
    with pytest.raises(LedgerError, match="measurement_evidence"):
        _validate_next(tmp_path, previous, current)


def test_unchanged_model_carries_forward_without_invented_revisions(tmp_path):
    previous = _ledger()
    assert _validate_next(tmp_path, previous, copy.deepcopy(previous)) == []
