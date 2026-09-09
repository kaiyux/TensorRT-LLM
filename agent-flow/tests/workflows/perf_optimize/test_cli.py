"""CLI help, task validation, and dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import yaml

from agent_flow.workflows.perf_optimize import cli
from agent_flow.workflows.perf_optimize.state import STATE_FILENAME


def test_help_describes_one_evidence_driven_kernel_ledger(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli._parse_args(["--help"])

    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "kernel_ledger.yaml each round" in help_text
    assert "eliminable?/faster?/fusible?/overlappable? per kernel" in help_text
    assert "best current theoretical performance model" in help_text
    assert "model revisions based on evidence, and unexplained gaps" in help_text
    assert "headroom_ledger" not in help_text


def test_reanalyze_requires_a_reuse_source(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli._parse_args(["--task", "task.yaml", "--reanalyze"])

    assert exc.value.code == 2
    assert "--reanalyze requires --reuse-analysis DIR" in capsys.readouterr().err


def test_reanalyze_rejects_resume_without_changing_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / STATE_FILENAME
    checkpoint.write_text('{"stage": "optimizer_evaluator"}\n', encoding="utf-8")
    before = checkpoint.read_bytes()

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--task",
                "task.yaml",
                "--workspace",
                str(tmp_path),
                "--reuse-analysis",
                "previous-run",
                "--reanalyze",
            ]
        )

    assert exc.value.code == 2
    assert "--reanalyze is for fresh runs only" in capsys.readouterr().err
    assert checkpoint.read_bytes() == before


def test_clean_allows_reanalysis_in_a_previous_workspace(tmp_path: Path) -> None:
    (tmp_path / STATE_FILENAME).write_text("{}\n", encoding="utf-8")

    args = cli._parse_args(
        [
            "--task",
            "task.yaml",
            "--workspace",
            str(tmp_path),
            "--reuse-analysis",
            "previous-run",
            "--reanalyze",
            "--clean",
        ]
    )

    assert args.reanalyze is True
    assert args.clean is True


@pytest.mark.parametrize("reanalyze", [False, True])
def test_cli_forwards_the_selected_reuse_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reanalyze: bool
) -> None:
    checkpoint_path = tmp_path / "model"
    repo = tmp_path / "repo"
    source = tmp_path / "previous-run"
    for directory in (checkpoint_path, repo, source):
        directory.mkdir()
    task = tmp_path / "task.yaml"
    task.write_text(
        yaml.safe_dump(
            {
                "checkpoint_path": str(checkpoint_path),
                "trtllm_repo_path": str(repo),
                "sol": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    constructor = MagicMock()
    dump_prompts = Mock()
    monkeypatch.setattr(cli, "PerfOptimizeWorkflow", constructor)
    monkeypatch.setattr(cli, "dump_prompt_bundle", dump_prompts)
    argv = [
        "--task",
        str(task),
        "--workspace",
        str(workspace),
        "--reuse-analysis",
        str(source),
    ]
    if reanalyze:
        argv.append("--reanalyze")

    cli.main(argv)

    constructor.assert_called_once()
    options = constructor.call_args.kwargs
    assert options["reuse_analysis"] == str(source)
    assert options["reanalyze"] is reanalyze
    assert options["workspace"] == workspace
    dump_prompts.assert_called_once_with(options["prompts"], workspace / cli.PROMPTS_DIRNAME)
    workflow = constructor.return_value.__enter__.return_value
    workflow.run.assert_called_once_with(str(task))
