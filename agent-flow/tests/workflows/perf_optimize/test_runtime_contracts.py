"""Runtime isolation and accepted-state evidence at role boundaries."""

from threading import Event, Lock

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


@pytest.mark.parametrize("slurm", [False, True])
def test_parallel_batch_only_overlaps_workers_with_isolated_jobs(tmp_path, slurm):
    """A frozen local batch must not let sibling smoke tests share port 8000."""
    with workflow_module.PerfOptimizeWorkflow(tmp_path / "workspace") as workflow:
        task = {"slurm-environment": {"slurm_partition": "batch"}} if slurm else {}
        workflow._task_data = lambda: task
        workflow._ensure_item_runtime = lambda state, entry: None
        state = WorkflowState(
            task_path=str(workflow.task_path),
            item_batch=[{"current_item_id": "a"}, {"current_item_id": "b"}],
        )
        lock = Lock()
        overlapped = Event()
        active = 0
        maximum = 0

        def worker(state, entry, log):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                if active == 2:
                    overlapped.set()
            # A concurrent sibling releases this wait; a local batch proceeds
            # one at a time even though both items were submitted together.
            overlapped.wait(timeout=0.2)
            with lock:
                active -= 1

        workflow._run_opt_item = worker
        workflow._run_opt_items_parallel(state, workflow_module.get_logger().console)
        assert maximum == (2 if slurm else 1)
        assert state.batch_completed


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
