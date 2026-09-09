"""Interruption regressions using real campaign and candidate git repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.perf_optimize import gitops, progress, roadmap_schema, workflow
from agent_flow.workflows.perf_optimize.state import (
    STAGE_EVALUATOR,
    STAGE_INTEGRATOR,
    STAGE_OPTIMIZER_EVALUATOR,
    STAGE_PROFILER,
    STAGE_QA,
    load_state,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class _UnusedAgent:
    def __call__(self, *args, **kwargs):
        pytest.fail("A recovery test must not invoke an agent")

    def __exit__(self, *args):
        pass


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "src.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    task = tmp_path / "input_task.yaml"
    task.write_text(
        yaml.safe_dump(
            {
                "checkpoint_path": str(checkpoint),
                "trtllm_repo_path": str(repo),
                "sol": {"enabled": False},
                "optimize": {"item_execution": "serial", "max_rounds": 2},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(workflow, "_make_agent", lambda *args, **kwargs: _UnusedAgent())
    flow = workflow.PerfOptimizeWorkflow(tmp_path / "workspace")
    state = flow._init_state(str(task), None)
    reference = {"value": 100.0, "source": "baseline/benchmark_results.md"}
    roadmap_schema.save_roadmap(
        flow.roadmap_path,
        {
            "version": 1,
            "target_metric": "output_throughput",
            "baseline": dict(reference),
            "current_best": dict(reference),
            "items": [
                {
                    "id": "opt-001",
                    "title": "Reduce launch overhead",
                    "category": "launch-host",
                    "approach": "code",
                    "evidence": ["nsys launch overhead"],
                    "expected_gain_pct": 10.0,
                    "expected_gain_rationale": "Recover launch overhead",
                    "how_to_apply": "Edit src.py",
                }
            ],
        },
    )
    yield flow, state, repo
    flow.close()


@pytest.mark.parametrize("resuming", [False, True])
@pytest.mark.parametrize("dirty_kind", ["tracked", "staged", "untracked"])
def test_dirty_campaign_checkout_is_preserved(campaign, dirty_kind, resuming):
    flow, state, repo = campaign
    if resuming:
        flow._ensure_optimization_branch(state, None)
    initial_branch = gitops.current_branch(repo)
    initial_head = gitops.rev_parse_head(repo)
    path = repo / ("new.py" if dirty_kind == "untracked" else "src.py")
    path.write_text("user work\n", encoding="utf-8")
    if dirty_kind == "staged":
        _git(repo, "add", path.name)
    original_status = _git(repo, "status", "--porcelain")

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        flow._ensure_optimization_branch(state, None)

    assert path.read_text(encoding="utf-8") == "user work\n"
    assert _git(repo, "status", "--porcelain") == original_status
    assert gitops.current_branch(repo) == initial_branch
    assert gitops.rev_parse_head(repo) == initial_head
    if not resuming:
        assert not state.campaign_git_branch


def test_resume_creates_planned_branch_at_original_base(campaign, monkeypatch):
    flow, state, repo = campaign
    original_branch = gitops.current_branch(repo)
    original_base = gitops.rev_parse_head(repo)
    create_branch = gitops.create_branch

    def interrupt_creation(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(gitops, "create_branch", interrupt_creation)
    with pytest.raises(KeyboardInterrupt):
        flow._ensure_optimization_branch(state, None)
    resumed = load_state(flow.state_path)
    assert resumed.campaign_git_base_commit == original_base
    assert not gitops.branch_exists(repo, resumed.campaign_git_branch)

    # The user's original branch can advance while the campaign is stopped.
    (repo / "src.py").write_text("user committed work\n", encoding="utf-8")
    user_commit = gitops.commit_all(repo, "unrelated user work")
    monkeypatch.setattr(gitops, "create_branch", create_branch)
    flow._ensure_optimization_branch(resumed, None)

    assert gitops.current_branch(repo) == resumed.campaign_git_branch
    assert gitops.rev_parse_head(repo) == original_base
    assert _git(repo, "rev-parse", original_branch) == user_commit
    flow._ensure_optimization_branch(resumed, None)  # Existing branch is reused.
    assert gitops.rev_parse_head(repo) == original_base


def _prepare_cached_approval(flow, state):
    flow._ensure_optimization_branch(state, None)
    state.stage = STAGE_OPTIMIZER_EVALUATOR
    flow._prepare_item_batch(state, roadmap_schema.load_roadmap(flow.roadmap_path)["items"])
    entry = state.item_batch[0]
    flow._update_batch_item(state, "opt-001", phase=STAGE_EVALUATOR, status="running")
    local_state = flow._local_item_state(state, entry)
    attempt = flow._attempt_dir(local_state)
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "evaluation.md").write_text("# APPROVE: measured 110\n", encoding="utf-8")
    progress.write_progress(
        flow._item_progress_path(local_state),
        {
            "optimization": [
                {
                    "step": 1,
                    "agent": "evaluator",
                    "summary": "Measured a 10% gain",
                    "attempt": 1,
                    "item_id": "opt-001",
                    "decision": "APPROVE",
                    "reason_category": "none",
                    "measured_gain_pct": 10.0,
                    "measured_value": 110.0,
                }
            ]
        },
    )
    return entry


def test_resume_after_candidate_commit_promotes_the_committed_source(campaign, monkeypatch):
    flow, state, repo = campaign
    entry = _prepare_cached_approval(flow, state)
    worktree = Path(entry["item_worktree_path"])
    (worktree / "src.py").write_text("x = 2\n", encoding="utf-8")
    commit_all = gitops.commit_all
    committed = []

    def interrupt_after_commit(*args, **kwargs):
        committed.append(commit_all(*args, **kwargs))
        raise KeyboardInterrupt

    monkeypatch.setattr(gitops, "commit_all", interrupt_after_commit)
    with pytest.raises(KeyboardInterrupt):
        flow._run_opt_item(state, dict(entry), None)
    assert gitops.worktree_clean(worktree)
    assert len(committed) == 1
    assert state.item_batch[0]["candidate_commit"] == ""

    resumed = load_state(flow.state_path)
    flow._ensure_item_runtime(resumed, resumed.item_batch[0])
    flow._run_opt_item(resumed, dict(resumed.item_batch[0]), None)

    assert resumed.item_batch[0]["candidate_commit"] == committed[0]
    assert len(committed) == 1  # Reused approval does not commit a second time.
    flow._finalize_serial_item(resumed, "opt-001", None)
    assert (repo / "src.py").read_text(encoding="utf-8") == "x = 2\n"
    assert gitops.rev_parse_head(repo) == committed[0]
    roadmap = roadmap_schema.load_roadmap(flow.roadmap_path)
    assert roadmap["items"][0]["status"] == "accepted"
    assert roadmap["current_best"]["value"] == 110.0


def test_config_only_approval_keeps_empty_candidate_commit(campaign):
    flow, state, repo = campaign
    entry = _prepare_cached_approval(flow, state)
    local_state = flow._local_item_state(state, entry)
    live, _ = flow._state_tuning_paths(local_state)
    live.write_text("enable_chunked_prefill: true\n", encoding="utf-8")

    flow._run_opt_item(state, dict(entry), None)

    assert state.item_batch[0]["status"] == "candidate_ready"
    assert state.item_batch[0]["candidate_commit"] == ""
    assert gitops.rev_parse_head(repo) == entry["item_base_commit"]


def test_resume_corrects_invalid_cached_approval_without_discarding_evidence(campaign, monkeypatch):
    flow, state, _ = campaign
    entry = _prepare_cached_approval(flow, state)
    local_state = flow._local_item_state(state, entry)
    attempt = flow._attempt_dir(local_state)
    evaluation = attempt / "evaluation.md"
    evaluation.unlink()
    progress_path = flow._item_progress_path(local_state)
    progress.write_progress(progress_path, {"optimization": []})
    worktree = Path(entry["item_worktree_path"])
    (worktree / "src.py").write_text("x = 2\n", encoding="utf-8")
    result = attempt / "result.json"
    feedback_received = []

    def evaluator(state, *, agent=None, progress_ctx=None, validation_feedback=""):
        feedback_received.append(validation_feedback)
        assert (worktree / "src.py").read_text(encoding="utf-8") == "x = 2\n"
        if len(feedback_received) == 1:
            result.write_text('{"output_throughput": 110, "completed": 1}', encoding="utf-8")
            gain = 25.0  # The measurement is valid, but this arithmetic is wrong.
        else:
            assert "measured_gain_pct" in validation_feedback
            assert result.read_text(encoding="utf-8") == (
                '{"output_throughput": 110, "completed": 1}'
            )
            gain = 10.0
        evaluation.write_text("# Evaluated candidate\n", encoding="utf-8")
        data = progress.read_progress(progress_path)
        data["optimization"].append(
            {
                "agent": "evaluator",
                "attempt": 1,
                "item_id": "opt-001",
                "decision": "APPROVE",
                "measured_gain_pct": gain,
                "measured_value": 110.0,
            }
        )
        progress.write_progress(progress_path, data)

    monkeypatch.setattr(flow, "_run_evaluator", evaluator)
    with pytest.raises(RuntimeError, match="measured_gain_pct"):
        flow._run_opt_item(state, dict(entry), None)
    failed = load_state(flow.state_path)
    assert failed.item_batch[0]["phase"] == STAGE_EVALUATOR
    assert failed.item_batch[0]["status"] == "error"
    assert not gitops.worktree_clean(worktree)

    flow._run_opt_item(failed, dict(failed.item_batch[0]), None)

    assert len(feedback_received) == 2
    assert feedback_received[0] == ""
    assert failed.item_batch[0]["status"] == "candidate_ready"
    assert failed.item_batch[0]["last_error"] == ""
    assert failed.item_batch[0]["attempts"] == 1
    assert gitops.worktree_clean(worktree)


@pytest.mark.parametrize("max_rounds, expected_stage", [(1, STAGE_QA), (2, STAGE_PROFILER)])
def test_batch_completion_survives_interrupted_worktree_cleanup(
    campaign, monkeypatch, max_rounds, expected_stage
):
    flow, state, repo = campaign
    entry = _prepare_cached_approval(flow, state)
    state.max_rounds = max_rounds
    state.stage = STAGE_INTEGRATOR
    integration = flow.worktrees_dir / "round_1" / "integration"
    state.integration_worktree_path = str(integration)
    state.integration_branch = f"{state.campaign_git_branch}-integration"
    gitops.create_worktree(repo, integration, state.integration_branch, entry["item_base_commit"])
    (integration / "src.py").write_text("x = 2\n", encoding="utf-8")
    accepted = gitops.commit_all(integration, "integrated candidate")
    gitops.fast_forward(repo, state.integration_branch)
    flow._checkpoint(state)
    remove_worktree = gitops.remove_worktree

    def interrupt_after_removal(source_repo, path):
        checkpoint = load_state(flow.state_path)
        assert checkpoint.stage == expected_stage
        assert checkpoint.round_index == 1
        assert checkpoint.item_batch == []
        assert checkpoint.integration_worktree_path == ""
        remove_worktree(source_repo, path)
        if Path(path) == integration:
            raise KeyboardInterrupt

    monkeypatch.setattr(gitops, "remove_worktree", interrupt_after_removal)
    with pytest.raises(KeyboardInterrupt):
        flow._finish_item_batch(state, None)

    resumed = load_state(flow.state_path)
    assert resumed.stage == expected_stage
    assert resumed.round_index == 1
    assert not integration.exists()
    assert gitops.rev_parse_head(repo) == accepted
    assert (repo / "src.py").read_text(encoding="utf-8") == "x = 2\n"
