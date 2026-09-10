"""Runtime isolation and accepted-state evidence at role boundaries."""

import shlex
from pathlib import Path
from threading import Barrier, Lock

import pytest

from agent_flow.workflows.perf_optimize import workflow as workflow_module
from agent_flow.workflows.perf_optimize.state import WorkflowState
from tests.workflows.perf_optimize.test_workflow import (
    FakeGitOps,
    _analyze_workspace,
    _RecordingAgent,
    _stub_agents,
    _write_task,
)


@pytest.mark.parametrize(
    "runtime",
    [
        {},
        {"slurm-environment": {"slurm_partition": "batch"}},
        {"disagg": {"config": "harness.yaml"}},
    ],
    ids=["local", "slurm", "disagg"],
)
@pytest.mark.parametrize("resume", [False, True])
def test_parallel_batch_overlaps_workers_for_all_runtimes(
    tmp_path: Path, runtime: dict, resume: bool
) -> None:
    """Optimizer work overlaps even when GPU measurements share a local runtime."""
    with workflow_module.PerfOptimizeWorkflow(tmp_path / "workspace") as workflow:
        workflow._task_data = lambda: runtime
        workflow._ensure_item_runtime = lambda state, entry: None
        batch = [{"current_item_id": item_id} for item_id in ("a", "b", "c")]
        if resume:
            batch.extend(
                [
                    {"current_item_id": "ready", "status": "candidate_ready"},
                    {"current_item_id": "failed", "status": "failed"},
                ]
            )
        state = WorkflowState(
            task_path=str(workflow.task_path),
            item_execution="parallel",
            item_batch=batch,
            batch_started=resume,
        )
        lock = Lock()
        started = Barrier(3, timeout=10)
        item_ids = []

        def worker(state: WorkflowState, entry: dict, log) -> None:
            with lock:
                item_ids.append(entry["current_item_id"])
            # No item can finish until every pending sibling starts. The
            # timeout only bounds a scheduler regression that would deadlock.
            started.wait()

        workflow._run_opt_item = worker
        workflow._run_opt_items_parallel(state, workflow_module.get_logger().console)
        assert sorted(item_ids) == ["a", "b", "c"]
        assert state.batch_started
        assert state.batch_completed


@pytest.mark.parametrize(
    "runtime",
    [
        {},
        {"slurm-environment": {"slurm_partition": "batch"}},
        {"disagg": {"config": "harness.yaml"}},
    ],
    ids=["local", "slurm", "disagg"],
)
@pytest.mark.parametrize("execution", ["serial", "parallel"])
def test_item_prompts_coordinate_only_shared_parallel_runtime(
    tmp_path: Path, runtime: dict, execution: str
) -> None:
    """Sibling optimizers and evaluators share one lock for complete runtime sessions."""
    task = _write_task(
        tmp_path,
        {"sol": {"enabled": False}, "optimize": {"item_execution": execution}},
    )
    with workflow_module.PerfOptimizeWorkflow(tmp_path / "workspace's runtime") as workflow:
        state = workflow._init_state(str(task), workflow_module.get_logger().console)
        task_data = {**workflow._task_data(), **runtime}
        workflow._task_data = lambda: task_data
        expected_lock = str(workflow.workspace.resolve() / ".local_runtime.lock")
        for item_index in range(2):
            state.item_index = item_index
            state.current_item_id = f"opt-{item_index}"
            state.item_worktree_path = str(tmp_path / f"candidate-{item_index}")
            for run_role in (workflow._run_optimizer, workflow._run_evaluator):
                recorder = _RecordingAgent()
                run_role(state, agent=recorder)
                prompt = recorder.messages[-1]
                if execution == "serial" or runtime:
                    assert ".local_runtime.lock" not in prompt
                    assert "flock -x" not in prompt
                    continue

                command = next(part for part in prompt.split("`") if part.startswith("flock -x "))
                assert shlex.split(command) == [
                    "flock",
                    "-x",
                    "--close",
                    "--",
                    expected_lock,
                    "bash",
                    "<runtime-session-script>",
                ]
                assert "CPU-only checks" in prompt
                assert "outside the lock" in prompt
                assert "complete runtime session" in prompt
                assert "Re-establish and verify your candidate after every acquisition" in prompt
                assert "EXIT/INT/TERM cleanup traps" in prompt
                assert "tear down and wait for all owned server/profiler process groups" in prompt
                assert "before the script exits and releases the lock" in prompt
                assert "never unlink" in prompt
                assert "never kill or benchmark that sibling's server" in prompt


@pytest.mark.parametrize("execution", ["serial", "parallel"])
@pytest.mark.parametrize("capture_error", ["missing", "invalid_manifest"])
def test_accept_without_capture_invalidates_previous_runtime_evidence(
    tmp_path, monkeypatch, execution, capture_error
):
    monkeypatch.setattr(workflow_module, "gitops", FakeGitOps())
    task = _write_task(
        tmp_path,
        {"sol": {"enabled": False}, "optimize": {"max_rounds": 1, "item_execution": execution}},
    )
    with workflow_module.PerfOptimizeWorkflow(tmp_path / "workspace") as workflow:
        _stub_agents(workflow)
        if capture_error == "invalid_manifest":
            evaluator = workflow._run_evaluator
            integrator = workflow.integrator

            def broken_capture(directory):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "profile_manifest.json").write_text("{}", encoding="utf-8")

            def evaluate(state, **kwargs):
                evaluator(state, **kwargs)
                broken_capture(workflow._attempt_dir(state) / "profile")

            def integrate(prompt):
                integrator(prompt)
                state = workflow_module.load_state(workflow.state_path)
                broken_capture(workflow._round_dir(state) / "integration" / "profile")

            workflow._run_evaluator = evaluate
            workflow.integrator = integrate
        workflow.run(str(task))
        state = workflow_module.load_state(workflow.state_path)
        assert state.last_profile_dir  # baseline-era capture remains on disk
        assert state.last_nsys_dir == ""  # it cannot describe the accepted change
        assert "no previous capture" in workflow._evaluator_capture_context(state)
        workflow.reporter = _RecordingAgent()
        workflow_module.PerfOptimizeWorkflow._run_reporter(workflow, state)
        assert "must say which accepted items it misses" in workflow.reporter.messages[-1]


@pytest.mark.parametrize("role", ["benchmarker", "qa"])
def test_measurement_turn_identifies_and_verifies_campaign_runtime(tmp_path, role):
    task = _write_task(tmp_path, {"sol": {"enabled": False}})
    with workflow_module.PerfOptimizeWorkflow(tmp_path / "workspace") as workflow:
        state = workflow._init_state(str(task), workflow_module.get_logger().console)
        recorder = _RecordingAgent()
        setattr(workflow, role, recorder)
        getattr(workflow, f"_run_{role}")(state)
        prompt = recorder.messages[-1]
        assert f"Active runtime checkout: `{tmp_path / 'repo'}`" in prompt
        assert "PYTHONPATH" in prompt
        assert "os.path.realpath(tensorrt_llm.__file__)" in prompt
        assert "isolated staged copy" in prompt


def test_reporter_distinguishes_fresh_baseline_in_reused_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow_module, "gitops", FakeGitOps())
    source = _analyze_workspace(tmp_path / "prior", baseline=False)
    task = _write_task(tmp_path, {"sol": {"enabled": False}})
    with workflow_module.PerfOptimizeWorkflow(
        tmp_path / "workspace", reuse_analysis=source
    ) as workflow:
        _stub_agents(workflow, analyzer_items=[[]])
        workflow.run(str(task))
        state = workflow_module.load_state(workflow.state_path)
        workflow.reporter = _RecordingAgent()
        workflow_module.PerfOptimizeWorkflow._run_reporter(workflow, state)
        prompt = workflow.reporter.messages[-1]
        assert "baseline was measured by this campaign" in prompt
        assert "baseline was measured by that run" not in prompt


def test_baseline_turn_scores_focus_subset_but_measures_full_curve(tmp_path):
    task = _write_task(
        tmp_path,
        {
            "sol": {"enabled": False},
            "benchmark": {"concurrency": [8, 32]},
            "optimize": {"focus_concurrencies": [32]},
        },
    )
    with workflow_module.PerfOptimizeWorkflow(tmp_path / "workspace") as workflow:
        state = workflow._init_state(str(task), workflow_module.get_logger().console)
        workflow.benchmarker = _RecordingAgent()
        workflow._run_benchmarker(state)
        prompt = workflow.benchmarker.messages[-1]
        assert "once per concurrency point [8, 32]" in prompt
        assert "mean over scored concurrency points [32]" in prompt
