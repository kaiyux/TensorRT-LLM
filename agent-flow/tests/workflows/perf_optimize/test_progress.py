"""Tests for the perf-optimize progress.yaml tool flow."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_flow.workflows.perf_optimize import progress as progress_module

_ROLES = (
    "benchmarker",
    "projector",
    "profiler",
    "analyzer",
    "optimizer",
    "evaluator",
    "integrator",
    "qa",
    "reporter",
)


def _call(handler, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(handler(args))


def _tool(tools, role: str, name: str):
    return next(t for t in tools[role] if t.name == name)


# --------------------------------------------------------------------- module


def test_read_progress_handles_missing_empty_and_seeded(tmp_path):
    path = tmp_path / "progress.yaml"

    empty = {"optimization": []}
    assert progress_module.read_progress(path) == empty  # missing

    path.write_text("", encoding="utf-8")
    assert progress_module.read_progress(path) == empty  # empty

    path.write_text("optimization: []\n", encoding="utf-8")
    assert progress_module.read_progress(path) == empty  # explicit empty mapping

    path.write_text(
        "optimization:\n  - step: 1\n    agent: benchmarker\n    summary: hi\n",
        encoding="utf-8",
    )
    assert progress_module.read_progress(path) == {
        "optimization": [
            {"step": 1, "agent": "benchmarker", "summary": "hi"},
        ],
    }


def test_read_progress_rejects_legacy_list(tmp_path):
    path = tmp_path / "progress.yaml"
    path.write_text("- step: 1\n  agent: benchmarker\n", encoding="utf-8")
    with pytest.raises(ValueError, match="optimization"):
        progress_module.read_progress(path)


def test_read_progress_rejects_non_list_stage(tmp_path):
    path = tmp_path / "progress.yaml"
    path.write_text("optimization: oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match="optimization"):
        progress_module.read_progress(path)


def test_init_progress_file_writes_canonical_key(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data == {"optimization": []}


# ----------------------------------------------------------------- tool shape


def test_every_role_has_append_and_read_tools(tmp_path):
    ctx = progress_module.ProgressContext(path=tmp_path / "progress.yaml")
    tools = progress_module.build_progress_tools(ctx)
    assert sorted(tools) == sorted(_ROLES)
    for role in _ROLES:
        names = [t.name for t in tools[role]]
        assert f"append_{role}_progress" in names, role
        assert "read_latest_progress" in names, role

    # The read tool's `agent` enum exposes every role.
    read = _tool(tools, "optimizer", "read_latest_progress")
    assert sorted(read.input_schema["properties"]["agent"]["enum"]) == sorted(
        (*_ROLES, "optimizer_evaluator")
    )


def test_item_append_writes_global_first_and_item_local(tmp_path):
    global_path = tmp_path / "progress.yaml"
    item_path = tmp_path / "item" / "progress.yaml"
    item_path.parent.mkdir()
    progress_module.init_progress_file(global_path)
    progress_module.init_progress_file(item_path)
    ctx = progress_module.ProgressContext(
        path=item_path,
        global_path=global_path,
        global_lock=progress_module.Lock(),
        current_step=1,
        current_round=1,
        current_attempt=1,
        current_item_id="opt-001",
    )
    tools = progress_module.build_progress_tools(ctx)
    _call(
        _tool(tools, "optimizer", "append_optimizer_progress").handler,
        {"summary": "candidate"},
    )
    global_entry = progress_module.read_progress(global_path)["optimization"][0]
    local_entry = progress_module.read_progress(item_path)["optimization"][0]
    assert global_entry["item_id"] == local_entry["item_id"] == "opt-001"
    assert global_entry["step"] == local_entry["step"] == 1


def test_integrator_tool_records_authoritative_verdict(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)
    tools = progress_module.build_progress_tools(
        progress_module.ProgressContext(path=path, current_step=1, current_round=1)
    )
    append = _tool(tools, "integrator", "append_integrator_progress")
    _call(
        append.handler,
        {
            "summary": "combined candidates passed",
            "decision": "APPROVE",
            "included_item_ids": ["opt-001", "opt-002"],
            "dropped_item_ids": [],
            "remediation_attempts": 0,
            "measured_gain_pct": 11.0,
            "measured_value": 111.0,
            "required_gain_pct": 7.4,
            "best_candidate_id": "opt-001",
        },
    )
    entry = progress_module.latest_entry(path, "integrator")
    assert entry["decision"] == "APPROVE"
    assert entry["included_item_ids"] == ["opt-001", "opt-002"]


def test_summary_tool_handlers_stamp_loop_position(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)

    ctx = progress_module.ProgressContext(path=path, current_step=1, current_round=0)
    tools = progress_module.build_progress_tools(ctx)

    _call(_tool(tools, "benchmarker", "append_benchmarker_progress").handler, {"summary": "b"})
    ctx.current_step = 2
    _call(_tool(tools, "projector", "append_projector_progress").handler, {"summary": "p"})
    ctx.current_step = 3
    ctx.current_round = 1
    _call(_tool(tools, "analyzer", "append_analyzer_progress").handler, {"summary": "a"})
    ctx.current_step = 4
    ctx.current_attempt = 1
    ctx.current_item_id = "opt-001"
    _call(_tool(tools, "optimizer", "append_optimizer_progress").handler, {"summary": "o"})

    entries = progress_module.read_progress(path)["optimization"]
    assert [e["agent"] for e in entries] == ["benchmarker", "projector", "analyzer", "optimizer"]
    assert [e["step"] for e in entries] == [1, 2, 3, 4]
    # The projector stamps round 0 — it runs before the round loop.
    assert [e["round"] for e in entries] == [0, 0, 1, 1]

    # attempt / item_id are stamped only while the inner loop is active.
    assert "attempt" not in entries[0] and "item_id" not in entries[0]
    assert "attempt" not in entries[1] and "item_id" not in entries[1]
    assert "attempt" not in entries[2] and "item_id" not in entries[2]
    assert entries[3]["attempt"] == 1
    assert entries[3]["item_id"] == "opt-001"
    for e in entries:
        assert "T" in e["timestamp"]


def test_analyzer_reads_independent_profiler_progress(tmp_path: Path) -> None:
    """Capture completion has its own role and remains readable on an analysis retry."""
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)
    ctx = progress_module.ProgressContext(path=path, current_step=3, current_round=1)
    tools = progress_module.build_progress_tools(ctx)
    _call(
        _tool(tools, "profiler", "append_profiler_progress").handler,
        {"summary": "Captured nsys; ncu unavailable. Saved round_1/profile/profile_manifest.json."},
    )
    ctx.current_step = 4
    output = _call(
        _tool(tools, "analyzer", "read_latest_progress").handler,
        {"agent": "profiler"},
    )
    (entry,) = yaml.safe_load(output["content"][0]["text"])
    assert entry["agent"] == "profiler"
    assert entry["step"] == 3
    assert entry["round"] == 1
    assert "profile_manifest.json" in entry["summary"]
    assert "attempt" not in entry
    assert progress_module.latest_entry(path, "analyzer") is None


def test_evaluator_tool_requires_and_records_structured_fields(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)
    ctx = progress_module.ProgressContext(
        path=path, current_step=4, current_round=1, current_attempt=2, current_item_id="opt-001"
    )
    tools = progress_module.build_progress_tools(ctx)
    append = _tool(tools, "evaluator", "append_evaluator_progress")

    schema = append.input_schema
    assert sorted(schema["required"]) == sorted(
        ["summary", "decision", "reason_category", "measured_gain_pct", "measured_value"]
    )
    assert schema["properties"]["decision"]["enum"] == ["APPROVE", "REJECT", "PUSH_BACK"]
    assert schema["properties"]["reason_category"]["enum"] == [
        "none",
        "code_quality",
        "functionality",
        "perf_shortfall",
    ]

    _call(
        append.handler,
        {
            "summary": "gate passed",
            "decision": "APPROVE",
            "reason_category": "none",
            "measured_gain_pct": 8.4,
            "measured_value": 1298.7,
        },
    )
    (entry,) = progress_module.read_progress(path)["optimization"]
    assert entry["agent"] == "evaluator"
    assert entry["decision"] == "APPROVE"
    assert entry["reason_category"] == "none"
    assert entry["measured_gain_pct"] == pytest.approx(8.4)
    assert entry["measured_value"] == pytest.approx(1298.7)
    assert entry["round"] == 1
    assert entry["attempt"] == 2
    assert entry["item_id"] == "opt-001"
    # Scalar mode: no curve key sneaks into the entry.
    assert "curve" not in entry


def test_qa_tool_requires_and_records_structured_fields(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)
    ctx = progress_module.ProgressContext(path=path, current_step=5, current_round=1)
    tools = progress_module.build_progress_tools(ctx)
    append = _tool(tools, "qa", "append_qa_progress")

    schema = append.input_schema
    assert sorted(schema["required"]) == sorted(["summary", "cumulative_improvement_pct"])
    # The final verification carries no loop decision — the orchestrator
    # owns the loop; qa only verifies.
    assert "decision" not in schema["properties"]

    _call(
        append.handler,
        {"summary": "verified the final state", "cumulative_improvement_pct": 8.4},
    )
    (entry,) = progress_module.read_progress(path)["optimization"]
    assert entry["agent"] == "qa"
    assert "decision" not in entry
    assert entry["cumulative_improvement_pct"] == pytest.approx(8.4)
    # QA runs outside the inner attempt loop.
    assert "attempt" not in entry and "item_id" not in entry
    # Scalar mode: no curve key sneaks into the entry.
    assert "curve" not in entry


def test_evaluator_and_qa_tools_record_optional_curve(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)
    ctx = progress_module.ProgressContext(path=path, current_step=4, current_round=1)
    tools = progress_module.build_progress_tools(ctx)
    curve = [
        {"concurrency": 8, "value": 812, "tok_s_user": 21.4, "tok_s_gpu": 101.5},
        {"concurrency": 32, "value": 1657.0, "tok_s_user": 12.9, "tok_s_gpu": 207.1},
    ]

    evaluator_append = _tool(tools, "evaluator", "append_evaluator_progress")
    # curve stays optional — scalar runs are untouched.
    assert "curve" not in evaluator_append.input_schema["required"]
    assert "curve" in evaluator_append.input_schema["properties"]
    _call(
        evaluator_append.handler,
        {
            "summary": "curve gate passed",
            "decision": "APPROVE",
            "reason_category": "none",
            "measured_gain_pct": 3.24,
            "measured_value": 1234.5,
            "curve": curve,
        },
    )

    qa_append = _tool(tools, "qa", "append_qa_progress")
    assert "curve" not in qa_append.input_schema["required"]
    assert "curve" in qa_append.input_schema["properties"]
    _call(
        qa_append.handler,
        {
            "summary": "verified curve",
            "cumulative_improvement_pct": 5.5,
            "curve": curve,
        },
    )

    evaluator_entry, qa_entry = progress_module.read_progress(path)["optimization"]
    for entry in (evaluator_entry, qa_entry):
        assert entry["curve"] == [
            {"concurrency": 8, "value": 812.0, "tok_s_user": 21.4, "tok_s_gpu": 101.5},
            {"concurrency": 32, "value": 1657.0, "tok_s_user": 12.9, "tok_s_gpu": 207.1},
        ]
        # Coerced to plain ints/floats.
        assert isinstance(entry["curve"][0]["concurrency"], int)
        assert isinstance(entry["curve"][0]["value"], float)


def test_find_entries_and_latest_entry(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.write_progress(
        path,
        {
            "optimization": [
                {"step": 1, "agent": "benchmarker", "summary": "b"},
                {"step": 2, "agent": "analyzer", "summary": "a"},
                {"step": 3, "agent": "evaluator", "summary": "e1", "decision": "REJECT"},
                {"step": 4, "agent": "evaluator", "summary": "e2", "decision": "APPROVE"},
            ],
        },
    )

    assert len(progress_module.find_entries(path)) == 4
    assert [e["agent"] for e in progress_module.find_entries(path, last_steps=1)] == ["evaluator"]
    assert [e["step"] for e in progress_module.find_entries(path, agent="evaluator")] == [3, 4]

    # latest_entry returns the most recent evaluator verdict.
    assert progress_module.latest_entry(path, "evaluator")["decision"] == "APPROVE"
    assert progress_module.latest_entry(path, "qa") is None
    assert progress_module.latest_entry(path, "not_a_role") is None

    with pytest.raises(ValueError):
        progress_module.find_entries(path, last_steps=0)
    assert progress_module.find_entries(path, agent="not_a_role") == []


def test_read_latest_progress_tool_filters_by_agent(tmp_path):
    path = tmp_path / "progress.yaml"
    progress_module.write_progress(
        path,
        {
            "optimization": [
                {"step": 1, "agent": "optimizer", "timestamp": "t1", "summary": "o1"},
                {
                    "step": 2,
                    "agent": "evaluator",
                    "timestamp": "t2",
                    "summary": "e1",
                    "decision": "REJECT",
                },
            ],
        },
    )
    ctx = progress_module.ProgressContext(path=path, current_step=3)
    tools = progress_module.build_progress_tools(ctx)

    # The optimizer fetches the evaluator's REJECT feedback on a retry.
    read = _tool(tools, "optimizer", "read_latest_progress")
    out = _call(read.handler, {"agent": "evaluator"})
    rendered = yaml.safe_load(out["content"][0]["text"])
    assert rendered == [
        {
            "step": 2,
            "agent": "evaluator",
            "timestamp": "t2",
            "summary": "e1",
            "decision": "REJECT",
        },
    ]

    # Empty file gives a human-readable stub.
    empty = tmp_path / "empty.yaml"
    progress_module.init_progress_file(empty)
    ctx2 = progress_module.ProgressContext(path=empty)
    read2 = _tool(progress_module.build_progress_tools(ctx2), "qa", "read_latest_progress")
    out = _call(read2.handler, {})
    assert "No optimization entries yet" in out["content"][0]["text"]


# ---------------------------------------------------- optional evaluator facts


def _evaluator_tool(path):
    progress_module.init_progress_file(path)
    ctx = progress_module.ProgressContext(
        path=path,
        current_step=4,
        current_round=1,
        current_attempt=1,
        current_item_id="opt-004",
    )
    return _tool(
        progress_module.build_progress_tools(ctx), "evaluator", "append_evaluator_progress"
    )


_REJECT = {
    "summary": "the tuned mapping never fired",
    "decision": "REJECT",
    "reason_category": "perf_shortfall",
    "measured_gain_pct": 0.0,
    "measured_value": 1200.0,
}


def test_evaluator_facts_are_optional_in_the_schema(tmp_path):
    append = _evaluator_tool(tmp_path / "progress.yaml")
    schema = append.input_schema
    for field in (
        "gap_implication",
        "gap_implication_note",
        "lever",
        "target_blocker",
        "measured_gain_pooled_pct",
        "measurement_confidence",
    ):
        assert field in schema["properties"], field
        assert field not in schema["required"], field
    assert schema["properties"]["gap_implication"]["enum"] == [
        "mechanism-already-present",
        "mechanism-inapplicable",
        "applied-but-no-gain",
        "change-not-live",
        "blocked-by-constraint",
    ]


def test_evaluator_records_the_structured_gap_implication(tmp_path):
    path = tmp_path / "progress.yaml"
    append = _evaluator_tool(path)
    _call(
        append.handler,
        {
            **_REJECT,
            "gap_implication": "mechanism-inapplicable",
            "gap_implication_note": "gated on T == 4; this deployment runs T == 3",
            "lever": "launch-geometry-tuning",
            "measured_gain_pooled_pct": 0.57,
            "measurement_confidence": "not-reproducible",
        },
    )
    (entry,) = progress_module.read_progress(path)["optimization"]
    assert entry["gap_implication"] == "mechanism-inapplicable"
    assert entry["lever"] == "launch-geometry-tuning"
    # The scored number stays exactly as measured; the pooled estimate is
    # recorded beside it rather than overwriting it.
    assert entry["measured_gain_pct"] == pytest.approx(0.0)
    assert entry["measured_gain_pooled_pct"] == pytest.approx(0.57)
    assert entry["measurement_confidence"] == "not-reproducible"


def test_evaluator_forwards_a_confirmed_target_blocker(tmp_path):
    path = tmp_path / "progress.yaml"
    append = _evaluator_tool(path)
    _call(
        append.handler,
        {
            **_REJECT,
            "gap_implication": "mechanism-inapplicable",
            "target_blocker": {
                "cause": "multi-consumer-pinned",
                "detail": "the gated-norm output also feeds the residual add",
                "evidence": "rounds/round_3/item_1_opt-009/attempt_2/optimization_summary.md",
                "confirmed": True,
            },
        },
    )
    (entry,) = progress_module.read_progress(path)["optimization"]
    assert entry["target_blocker"]["cause"] == "multi-consumer-pinned"
    assert entry["target_blocker"]["confirmed"] is True


def test_scalar_runs_never_grow_the_new_keys(tmp_path):
    path = tmp_path / "progress.yaml"
    append = _evaluator_tool(path)
    _call(append.handler, {**_REJECT, "decision": "APPROVE", "reason_category": "none"})
    (entry,) = progress_module.read_progress(path)["optimization"]
    for field in ("gap_implication", "lever", "target_blocker"):
        assert field not in entry


@pytest.mark.parametrize("decision", ["APPROVE", "PUSH_BACK", "REJECT"])
def test_evaluator_accepts_verdicts_without_optional_facts(tmp_path, decision):
    path = tmp_path / "progress.yaml"
    append = _evaluator_tool(path)
    _call(
        append.handler,
        {
            **_REJECT,
            "decision": decision,
            "reason_category": "none" if decision == "APPROVE" else "perf_shortfall",
        },
    )
    (entry,) = progress_module.read_progress(path)["optimization"]
    assert entry["decision"] == decision
    assert "gap_implication" not in entry
