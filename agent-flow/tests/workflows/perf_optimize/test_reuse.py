"""Tests for importing a previous run's analysis into a new workspace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.perf_optimize import reuse
from agent_flow.workflows.perf_optimize.profile import (
    PROFILE_MANIFEST_NAME,
    validate_profile_manifest,
)

# --------------------------------------------------------------------- helpers


def _write(path: Path, text: str = "content\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _perf_analyze_workspace(root: Path) -> Path:
    """A completed perf-analyze workspace (flat layout)."""
    _write(root / "benchmark_results.md", "# baseline\n")
    _write(root / "profile_findings.md", "# findings\n")
    _write(root / "sol_projection.md", "# SOL\n")
    _write(root / "sol_work" / "peaks.json", "{}\n")
    _write(root / "server_nsys.nsys-rep", "trace\n")
    _write(root / "nsys_stats.txt", "kern_sum\n")
    _write(root / "concurrency_8" / "result.json", "{}\n")
    _write(root / "bench_result.json", "{}\n")
    # Not analysis artifacts — the import must leave these behind.
    _write(root / "performance_report.md", "# report\n")
    _write(root / "task.yaml", "checkpoint_path: /x\n")
    _write(root / ".perf_analyze_state.json", "{}\n")
    return root


def _perf_optimize_workspace(root: Path, rounds: int = 2) -> Path:
    """A completed perf-optimize campaign workspace (nested layout)."""
    _write(root / "baseline" / "benchmark_results.md", "# baseline\n")
    _write(root / "baseline" / "concurrency_8" / "result.json", "{}\n")
    _write(root / "sol_projection.md", "# SOL\n")
    _write(root / "sol_work" / "peaks.json", "{}\n")
    _write(root / "roadmap.yaml", "version: 1\n")
    for index in range(1, rounds + 1):
        analysis = root / "rounds" / f"round_{index}" / "analysis"
        _write(analysis / "profile_findings.md", f"# findings round {index}\n")
        _write(analysis / "server_nsys.nsys-rep", "trace\n")
        _write(analysis / "kernel_ledger.yaml", f"version: 2  # round {index}\n")
    _write(root / "optimization_report.md", "# report\n")
    return root


def _import_into(source: Path, workspace: Path) -> reuse.ImportedAnalysis:
    return reuse.import_analysis(
        reuse.discover(source),
        workspace=workspace,
        baseline_dir=workspace / "baseline",
        analysis_dir=workspace / "rounds" / "round_1" / "analysis",
        sol_projection_path=workspace / "sol_projection.md",
        sol_work_dir=workspace / "sol_work",
        reuse_dir=workspace / reuse.REUSE_DIRNAME,
    )


def _capture(root: Path, capture_id: str = "round_1-capture") -> Path:
    _write(root / "rank_0" / "server_nsys.nsys-rep", f"trace {capture_id}\n")
    _write(root / "nsys_analysis" / "summary.json", '{"preliminary": true}\n')
    manifest = {
        "schema_version": 1,
        "capture_id": capture_id,
        "runtime": {
            "serve_command": "serve model",
            "benchmark_command": "bench model",
            "config": "{backend: pytorch}",
            "build": "abc123",
            "import_path": "/repo/tensorrt_llm/__init__.py",
        },
        "methods": {
            "nsys": {
                "status": "captured",
                "command": "nsys profile ...",
                "artifacts": ["rank_0/server_nsys.nsys-rep", "nsys_analysis/summary.json"],
            },
            "ncu": {"status": "unavailable", "reason": "No counter permissions"},
        },
    }
    _write(root / PROFILE_MANIFEST_NAME, json.dumps(manifest))
    return root


def _split_import(source: Path, workspace: Path, *, reanalyze: bool = False):
    return reuse.import_analysis(
        reuse.discover(source),
        workspace=workspace,
        baseline_dir=workspace / "baseline",
        analysis_dir=workspace / "rounds" / "round_1" / "analysis",
        profile_dir=workspace / "rounds" / "round_1" / "profile",
        sol_projection_path=workspace / "sol_projection.md",
        sol_work_dir=workspace / "sol_work",
        reuse_dir=workspace / reuse.REUSE_DIRNAME,
        reanalyze=reanalyze,
    )


# -------------------------------------------------------------------- discover


def test_discover_reads_the_perf_analyze_layout(tmp_path):
    source = _perf_analyze_workspace(tmp_path / "analyze")
    found = reuse.discover(source)
    assert found.baseline_report == source / "benchmark_results.md"
    assert found.findings == source / "profile_findings.md"
    assert found.sol_projection == source / "sol_projection.md"
    assert found.sol_work == source / "sol_work"
    # perf-analyze never writes a roadmap.
    assert found.prior_roadmap is None
    assert found.kernel_ledger is None
    assert found.is_empty is False


def test_discover_reads_the_perf_optimize_layout(tmp_path):
    source = _perf_optimize_workspace(tmp_path / "optimize")
    found = reuse.discover(source)
    assert found.baseline_report == source / "baseline" / "benchmark_results.md"
    assert found.prior_roadmap == source / "roadmap.yaml"
    assert found.kernel_ledger == found.findings.parent / "kernel_ledger.yaml"


def test_discover_picks_the_newest_round_numerically(tmp_path):
    """``round_10`` outranks ``round_9`` — the last state profiled wins."""
    source = _perf_optimize_workspace(tmp_path / "optimize", rounds=10)
    found = reuse.discover(source)
    assert found.findings == (source / "rounds" / "round_10" / "analysis" / "profile_findings.md")


def test_discover_skips_a_trailing_replan_round(tmp_path):
    """The newest round is not always the one that profiled.

    A perf-optimize round opens **replan-only** when the previous one
    accepted nothing: it writes a short replan note into its `analysis/`
    and captures nothing. A plateau campaign typically ends on one, so
    taking the numerically newest findings would import a pointer note
    with no traces beside it — the round before holds the real evidence.
    """
    source = _perf_optimize_workspace(tmp_path / "optimize", rounds=2)
    replan = source / "rounds" / "round_3" / "analysis"
    _write(replan / "profile_findings.md", "# replan note (round 3)\n")

    found = reuse.discover(source)

    assert found.findings == (source / "rounds" / "round_2" / "analysis" / "profile_findings.md")
    assert found.kernel_ledger == found.findings.parent / "kernel_ledger.yaml"


@pytest.mark.parametrize("provenance", ["source", "manifest"])
def test_discover_reuses_newest_ledger_for_selected_capture(tmp_path, provenance):
    source = tmp_path / "source"
    profile = _capture(source / "rounds" / "round_1" / "profile", "standing-capture")
    analysis = source / "rounds" / "round_1" / "analysis"
    findings = _write(analysis / reuse.FINDINGS_NAME, "full findings")
    _write(analysis / reuse.KERNEL_LEDGER_NAME, "version: 2\n")
    _write(analysis / "nsys_analysis" / "summary.json", "{}")
    for number in (2, 10):
        later = source / "rounds" / f"round_{number}" / "analysis"
        _write(
            later / reuse.KERNEL_LEDGER_NAME,
            "version: 2\nsource: rounds/round_1/analysis/nsys_analysis\n"
            f"model_revisions: [{{round: {number}}}]\n",
        )
        if provenance == "manifest":
            _write(
                later / "analysis_manifest.yaml",
                "capture_id: standing-capture\nprofile_dir: rounds/round_1/profile\n",
            )

    found = reuse.discover(source)

    assert found.findings == findings
    assert found.profile_dir == profile
    assert (
        found.kernel_ledger
        == source / "rounds" / "round_10" / "analysis" / reuse.KERNEL_LEDGER_NAME
    )


@pytest.mark.parametrize("provenance", ["missing", "other-capture", "invalid-manifest", "outside"])
def test_discover_keeps_sibling_ledger_when_newer_provenance_does_not_match(tmp_path, provenance):
    source = tmp_path / "source"
    _capture(source / "rounds" / "round_1" / "profile", "selected-capture")
    analysis = source / "rounds" / "round_1" / "analysis"
    _write(analysis / reuse.FINDINGS_NAME)
    sibling = _write(analysis / reuse.KERNEL_LEDGER_NAME, "version: 2\n")
    _capture(source / "rounds" / "round_2" / "profile", "different-capture")
    _capture(tmp_path / "outside", "outside-capture")
    references = {
        "missing": "unknown capture",
        "other-capture": "rounds/round_2/profile",
        "invalid-manifest": "rounds/round_1/profile",
        "outside": "../outside",
    }
    later = source / "rounds" / "round_3" / "analysis"
    _write(later / reuse.KERNEL_LEDGER_NAME, f"version: 2\nsource: {references[provenance]}\n")
    if provenance == "invalid-manifest":
        _write(
            later / "analysis_manifest.yaml",
            "capture_id: wrong-capture\nprofile_dir: rounds/round_1/profile\n",
        )

    found = reuse.discover(source)

    assert found.findings == analysis / reuse.FINDINGS_NAME
    assert found.kernel_ledger == sibling


def test_discover_recognizes_a_profile_that_omitted_nsys(tmp_path):
    """`profile.methods` supports ncu-only profiling rounds."""
    source = tmp_path / "optimize"
    _write(source / "baseline" / "benchmark_results.md", "# baseline\n")
    profiled = source / "rounds" / "round_2" / "analysis"
    _write(profiled / "profile_findings.md", "# profiled without nsys\n")
    _write(profiled / "server_ncu.ncu-rep", "capture\n")
    replan = source / "rounds" / "round_3" / "analysis"
    _write(replan / "profile_findings.md", "# trailing replan note\n")

    found = reuse.discover(source)

    assert found.findings == profiled / "profile_findings.md"


def test_discover_falls_back_to_prose_when_no_round_profiled(tmp_path):
    """Importing findings without traces still beats importing nothing."""
    source = tmp_path / "optimize"
    _write(source / "baseline" / "benchmark_results.md", "# baseline\n")
    _write(source / "rounds" / "round_1" / "analysis" / "profile_findings.md", "# note\n")

    found = reuse.discover(source)

    assert found.findings == (source / "rounds" / "round_1" / "analysis" / "profile_findings.md")


def test_discover_ignores_blank_managed_placeholders(tmp_path):
    """A fresh run pre-creates blank files; those are not artifacts."""
    source = tmp_path / "aborted"
    _write(source / "baseline" / "benchmark_results.md", "")
    _write(source / "sol_projection.md", "   \n")
    _write(source / "roadmap.yaml", "")
    found = reuse.discover(source)
    assert found.baseline_report is None
    assert found.sol_projection is None
    assert found.prior_roadmap is None
    assert found.is_empty is True


def test_discover_rejects_a_non_directory(tmp_path):
    with pytest.raises(reuse.ReuseError, match="not a directory"):
        reuse.discover(tmp_path / "nope")


# ---------------------------------------------------------------------- import


def test_import_from_perf_analyze_lands_in_canonical_paths(tmp_path):
    source = _perf_analyze_workspace(tmp_path / "analyze")
    ws = tmp_path / "ws"
    imported = _import_into(source, ws)

    analysis = ws / "rounds" / "round_1" / "analysis"
    assert (ws / "baseline" / "benchmark_results.md").read_text(encoding="utf-8") == "# baseline\n"
    assert (analysis / "profile_findings.md").read_text(encoding="utf-8") == "# findings\n"
    assert (ws / "sol_projection.md").read_text(encoding="utf-8") == "# SOL\n"
    assert (ws / "sol_work" / "peaks.json").is_file()
    # Result JSONs follow the baseline report (the evaluator diffs its
    # full metric set against them), traces follow the findings.
    assert (ws / "baseline" / "bench_result.json").is_file()
    assert (ws / "baseline" / "concurrency_8" / "result.json").is_file()
    assert (analysis / "server_nsys.nsys-rep").is_file()
    assert (analysis / "nsys_stats.txt").is_file()

    assert imported.baseline_report is True
    assert imported.findings is True
    assert imported.sol_projection is True
    assert imported.sol_work is True
    assert imported.kernel_ledger is False
    assert imported.prior_roadmap is False


def test_import_leaves_the_sources_non_analysis_files_behind(tmp_path):
    """The sibling sweep is an allowlist, not a directory copy.

    A perf-analyze workspace keeps its report, spec and checkpoint next
    to the artifacts; importing those would collide with this run's own
    managed files.
    """
    source = _perf_analyze_workspace(tmp_path / "analyze")
    ws = tmp_path / "ws"
    _import_into(source, ws)

    analysis = ws / "rounds" / "round_1" / "analysis"
    for stray in ("performance_report.md", "task.yaml", ".perf_analyze_state.json"):
        assert not (analysis / stray).exists()
        assert not (ws / "baseline" / stray).exists()
    assert not (ws / "task.yaml").exists()


def test_import_brings_the_nsys_timeline_analysis_along(tmp_path):
    """The `nsys_analysis/` products travel with the trace they describe.

    A reused analysis is planned from, so the per-iteration budget and
    the compute-absent split matter as much as the `.nsys-rep` — and
    unlike the multi-GB `.sqlite` export they are small JSON.
    """
    source = _perf_analyze_workspace(tmp_path / "analyze")
    _write(source / "nsys_analysis" / "summary.json", '{"mode": "single-variant"}\n')
    _write(source / "nsys_analysis" / "rank-0" / "gap.json", "{}\n")
    _write(source / "server_nsys.sqlite", "regenerable export\n")
    ws = tmp_path / "ws"
    _import_into(source, ws)

    analysis = ws / "rounds" / "round_1" / "analysis"
    assert (analysis / "nsys_analysis" / "summary.json").is_file()
    assert (analysis / "nsys_analysis" / "rank-0" / "gap.json").is_file()
    # The sqlite stays behind: regenerable from the .nsys-rep next to it.
    assert not (analysis / "server_nsys.sqlite").exists()


def test_import_from_perf_optimize_brings_ledger_and_prior_roadmap(tmp_path):
    source = _perf_optimize_workspace(tmp_path / "optimize")
    ws = tmp_path / "ws"
    imported = _import_into(source, ws)

    analysis = ws / "rounds" / "round_1" / "analysis"
    # Full findings remain analysis; the source ledger is reference material.
    assert "round 2" in (analysis / "profile_findings.md").read_text(encoding="utf-8")
    prior = ws / reuse.REUSE_DIRNAME / reuse.PRIOR_KERNEL_LEDGER_NAME
    assert "round 2" in prior.read_text(encoding="utf-8")
    assert not (analysis / reuse.KERNEL_LEDGER_NAME).exists()
    assert imported.kernel_ledger is True
    # The source roadmap is prior art only — never this campaign's ledger.
    assert (ws / reuse.REUSE_DIRNAME / reuse.PRIOR_ROADMAP_NAME).is_file()
    assert not (ws / "roadmap.yaml").exists()
    assert imported.prior_roadmap is True


@pytest.mark.parametrize("reanalyze", [False, True])
def test_import_preserves_replan_ledger_as_prior_art_with_source_provenance(tmp_path, reanalyze):
    source = tmp_path / "source"
    _capture(source / "rounds" / "round_1" / "profile")
    analysis = source / "rounds" / "round_1" / "analysis"
    _write(analysis / reuse.FINDINGS_NAME, "full profiling findings")
    _write(analysis / reuse.KERNEL_LEDGER_NAME, "version: 2\n")
    source_ledger = _write(
        source / "rounds" / "round_7" / "analysis" / reuse.KERNEL_LEDGER_NAME,
        "version: 2\nsource: rounds/round_1/profile\n"
        "model_revisions: [{round: 7, evidence: [rounds/round_7/analysis/facts.md]}]\n",
    )
    workspace = tmp_path / "workspace"

    imported = _split_import(source, workspace, reanalyze=reanalyze)

    prior = workspace / reuse.REUSE_DIRNAME / reuse.PRIOR_KERNEL_LEDGER_NAME
    assert imported.kernel_ledger
    assert prior.read_text() == source_ledger.read_text()
    assert not (workspace / "rounds" / "round_1" / "analysis" / reuse.KERNEL_LEDGER_NAME).exists()
    assert not (
        workspace / reuse.REUSE_DIRNAME / reuse.PRIOR_ANALYSIS_DIRNAME / reuse.KERNEL_LEDGER_NAME
    ).exists()
    manifest = imported.manifest_path.read_text()
    assert str(source_ledger) in manifest
    assert "reused_analysis/kernel_ledger.yaml" in manifest
    assert "prior" in manifest and "model revision history" in manifest


def test_import_writes_a_manifest_naming_source_and_destinations(tmp_path):
    source = _perf_optimize_workspace(tmp_path / "optimize")
    ws = tmp_path / "ws"
    imported = _import_into(source, ws)

    assert imported.manifest_path == ws / reuse.REUSE_DIRNAME / reuse.MANIFEST_NAME
    manifest = imported.manifest_path.read_text(encoding="utf-8")
    assert str(source) in manifest
    assert "benchmark_results.md" in manifest
    assert "rounds/round_1/analysis/profile_findings.md" in manifest
    # The provenance warning the report is expected to relay.
    assert "measured" in manifest


def test_import_is_best_effort_per_artifact(tmp_path):
    """A source with only findings still saves the profile."""
    source = tmp_path / "partial"
    _write(source / "profile_findings.md", "# findings\n")
    ws = tmp_path / "ws"
    imported = _import_into(source, ws)

    assert imported.findings is True
    assert imported.baseline_report is False
    assert imported.sol_projection is False
    assert not (ws / "baseline" / "benchmark_results.md").exists()


def test_import_raises_when_nothing_is_reusable(tmp_path):
    source = tmp_path / "empty"
    _write(source / "notes.txt", "hello\n")
    with pytest.raises(reuse.ReuseError, match="no reusable analysis"):
        _import_into(source, tmp_path / "ws")


def test_import_summary_lists_what_came_in(tmp_path):
    source = _perf_optimize_workspace(tmp_path / "optimize")
    summary = _import_into(source, tmp_path / "ws").summary()
    assert "baseline benchmark" in summary
    assert "profile findings" in summary
    assert "kernel ledger" in summary


def test_discover_capture_that_completed_before_analysis_failed(tmp_path):
    source = tmp_path / "source"
    capture = _capture(source / "rounds" / "round_1" / "profile")

    found = reuse.discover(source)

    assert found.profile_dir == capture
    assert found.findings is None
    assert not found.is_empty
    assert reuse.discover(source, reanalyze=True).profile_dir == capture


def test_discover_profile_directory_directly(tmp_path):
    source = _capture(tmp_path / "profile")

    assert reuse.discover(source).profile_dir == source
    assert reuse.discover(source, reanalyze=True).profile_dir == source


def test_discover_newest_capture_numerically_and_skip_incomplete_manifest(tmp_path):
    source = tmp_path / "source"
    _capture(source / "rounds" / "round_9" / "profile", "nine")
    ten = _capture(source / "rounds" / "round_10" / "profile", "ten")
    incomplete = _capture(source / "rounds" / "round_11" / "profile", "eleven")
    (incomplete / "rank_0" / "server_nsys.nsys-rep").unlink()

    assert reuse.discover(source, reanalyze=True).profile_dir == ten


def test_default_reuse_keeps_findings_paired_with_their_capture(tmp_path):
    source = tmp_path / "source"
    first = _capture(source / "rounds" / "round_1" / "profile", "one")
    first_findings = _write(source / "rounds" / "round_1" / "analysis" / reuse.FINDINGS_NAME)
    second = _capture(source / "rounds" / "round_2" / "profile", "two")

    ordinary = reuse.discover(source)
    reanalysis = reuse.discover(source, reanalyze=True)

    assert ordinary.findings == first_findings
    assert ordinary.profile_dir == first
    assert reanalysis.profile_dir == second
    assert reanalysis.findings is None


def test_new_replan_note_does_not_replace_last_full_findings(tmp_path):
    source = tmp_path / "source"
    profile = _capture(source / "rounds" / "round_1" / "profile")
    findings = _write(source / "rounds" / "round_1" / "analysis" / reuse.FINDINGS_NAME)
    _write(source / "rounds" / "round_2" / "analysis" / reuse.FINDINGS_NAME, "Replan note")

    found = reuse.discover(source)

    assert found.profile_dir == profile
    assert found.findings == findings


def test_reanalysis_identity_links_new_findings_to_previous_capture(tmp_path):
    source = tmp_path / "source"
    profile = _capture(source / "rounds" / "round_1" / "profile", "first-capture")
    _write(source / "rounds" / "round_1" / "analysis" / reuse.FINDINGS_NAME, "old findings")
    latest = source / "rounds" / "round_3" / "analysis"
    _write(latest / reuse.FINDINGS_NAME, "fresh interpretation")
    _write(
        latest / "analysis_manifest.yaml",
        "schema_version: 1\ncapture_id: first-capture\nprofile_dir: rounds/round_1/profile\n",
    )

    for reanalyze in (False, True):
        found = reuse.discover(source, reanalyze=reanalyze)
        assert found.findings == latest / reuse.FINDINGS_NAME
        assert found.profile_dir == profile


@pytest.mark.parametrize(
    "reference,capture_id", [("../profile", "outside"), ("rounds/round_1/profile", "wrong")]
)
def test_discovery_rejects_invalid_analysis_capture_identity(tmp_path, reference, capture_id):
    source = tmp_path / "source"
    _capture(tmp_path / "profile", "outside")
    profile = _capture(source / "rounds" / "round_1" / "profile", "one")
    findings = _write(source / "rounds" / "round_1" / "analysis" / reuse.FINDINGS_NAME, "valid")
    later = source / "rounds" / "round_2" / "analysis"
    _write(later / reuse.FINDINGS_NAME, "untrustworthy pointer")
    _write(
        later / "analysis_manifest.yaml", f"capture_id: {capture_id}\nprofile_dir: {reference}\n"
    )

    found = reuse.discover(source)

    assert found.profile_dir == profile
    assert found.findings == findings


def test_flat_legacy_findings_do_not_borrow_an_unrelated_sibling_capture(tmp_path):
    source = _perf_analyze_workspace(tmp_path / "source")
    _capture(tmp_path / "profile", "unrelated")

    assert reuse.discover(source).profile_dir == source


def test_capture_only_import_preserves_nested_inventory_and_provenance(tmp_path):
    source = tmp_path / "source"
    capture = _capture(source / "rounds" / "round_2" / "profile", "capture-only")
    _write(capture / "serve_config.yaml", "backend: pytorch\n")
    manifest = json.loads((capture / PROFILE_MANIFEST_NAME).read_text())
    manifest["artifacts"] = ["serve_config.yaml"]
    manifest["runtime"]["config"] = "serve_config.yaml: backend=pytorch"
    _write(capture / PROFILE_MANIFEST_NAME, json.dumps(manifest))
    workspace = tmp_path / "destination"
    before = (capture / PROFILE_MANIFEST_NAME).read_bytes()

    imported = _split_import(source, workspace, reanalyze=True)

    profile = workspace / "rounds" / "round_1" / "profile"
    assert imported.profile
    assert not imported.findings
    assert validate_profile_manifest(profile, require_raw=True)["capture_id"] == "capture-only"
    assert (profile / "rank_0" / "server_nsys.nsys-rep").is_file()
    assert (profile / "nsys_analysis" / "summary.json").is_file()
    assert (profile / "serve_config.yaml").read_text() == "backend: pytorch\n"
    assert (capture / PROFILE_MANIFEST_NAME).read_bytes() == before
    assert not (workspace / "rounds" / "round_1" / "analysis" / reuse.FINDINGS_NAME).exists()


def test_reanalysis_preserves_prior_findings_separately_from_fresh_outputs(tmp_path):
    source = tmp_path / "source"
    _capture(source / "rounds" / "round_1" / "profile")
    analysis = source / "rounds" / "round_1" / "analysis"
    findings = _write(analysis / reuse.FINDINGS_NAME, "source conclusions")
    _write(analysis / reuse.KERNEL_LEDGER_NAME, "source ledger")
    _write(analysis / "nsys_analysis" / "summary.json", '{"old_derivation":true}')
    workspace = tmp_path / "destination"

    imported = _split_import(source, workspace, reanalyze=True)

    prior = workspace / reuse.REUSE_DIRNAME / reuse.PRIOR_ANALYSIS_DIRNAME
    assert (prior / reuse.FINDINGS_NAME).read_text() == "source conclusions"
    prior_ledger = workspace / reuse.REUSE_DIRNAME / reuse.PRIOR_KERNEL_LEDGER_NAME
    assert prior_ledger.read_text() == "source ledger"
    assert not (prior / reuse.KERNEL_LEDGER_NAME).exists()
    assert (prior / "nsys_analysis" / "summary.json").is_file()
    assert not imported.findings
    assert imported.kernel_ledger
    assert not (workspace / "rounds" / "round_1" / "analysis" / reuse.FINDINGS_NAME).exists()
    assert findings.read_text() == "source conclusions"


def test_reanalysis_import_reselects_newest_capture_without_mixing_findings(tmp_path):
    source = tmp_path / "source"
    _capture(source / "rounds" / "round_1" / "profile", "one")
    _write(source / "rounds" / "round_1" / "analysis" / reuse.FINDINGS_NAME, "older runtime")
    _capture(source / "rounds" / "round_2" / "profile", "two")
    workspace = tmp_path / "destination"

    imported = _split_import(source, workspace, reanalyze=True)

    manifest = validate_profile_manifest(workspace / "rounds" / "round_1" / "profile")
    assert manifest["capture_id"] == "two"
    assert imported.profile
    assert not (workspace / reuse.REUSE_DIRNAME / reuse.PRIOR_ANALYSIS_DIRNAME).exists()


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_legacy_raw_captures_support_reanalysis_without_findings(tmp_path, layout):
    source = tmp_path / "source"
    original = source if layout == "flat" else source / "rounds" / "round_2" / "analysis"
    _write(original / "server_ncu.ncu-rep", "raw counters")
    workspace = tmp_path / "destination"

    imported = _split_import(source, workspace, reanalyze=True)

    profile = workspace / "rounds" / "round_1" / "profile"
    manifest = validate_profile_manifest(profile, require_raw=True)
    assert imported.profile
    assert manifest["methods"]["ncu"]["status"] == "captured"
    assert manifest["runtime"]["import_path"].startswith("unavailable:")
    assert (original / "server_ncu.ncu-rep").read_text() == "raw counters"
    assert not (original / PROFILE_MANIFEST_NAME).exists()


def test_split_legacy_import_separates_reports_and_derivations(tmp_path):
    source = _perf_analyze_workspace(tmp_path / "source")
    _write(source / "nsys_analysis" / "summary.json", "old decomposition")
    workspace = tmp_path / "destination"

    imported = _split_import(source, workspace)

    round_dir = workspace / "rounds" / "round_1"
    assert imported.findings and imported.profile
    assert (round_dir / "profile" / "server_nsys.nsys-rep").is_file()
    assert not (round_dir / "analysis" / "server_nsys.nsys-rep").exists()
    assert (round_dir / "analysis" / "nsys_analysis" / "summary.json").is_file()
    assert (round_dir / "analysis" / reuse.FINDINGS_NAME).is_file()


def test_reanalysis_rejects_prose_even_when_baseline_is_available(tmp_path):
    source = tmp_path / "source"
    _write(source / reuse.BASELINE_REPORT_NAME)
    _write(source / reuse.FINDINGS_NAME)
    _write(source / "nsys_stats.txt", "aggregated summary")

    with pytest.raises(reuse.ReuseError, match="requires usable raw captures"):
        _split_import(source, tmp_path / "destination", reanalyze=True)
    assert not (tmp_path / "destination").exists()


def test_unavailable_only_manifest_can_be_imported_but_not_reanalyzed(tmp_path):
    source = _capture(tmp_path / "source")
    manifest_path = source / PROFILE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["methods"]["nsys"] = {"status": "unavailable", "reason": "No GPU"}
    manifest_path.write_text(json.dumps(manifest))

    assert _split_import(source, tmp_path / "destination").profile
    with pytest.raises(reuse.ReuseError, match="requires usable raw captures"):
        _split_import(source, tmp_path / "reanalyze", reanalyze=True)


def test_chained_imports_rebase_analysis_identity_to_destination_profile(tmp_path):
    source = tmp_path / "source"
    _capture(source / "rounds" / "round_3" / "profile", "third-capture")
    analysis = source / "rounds" / "round_3" / "analysis"
    _write(analysis / reuse.FINDINGS_NAME, "third round findings")
    original_identity = (
        "schema_version: 1\nanalysis_id: round_3\ncapture_id: third-capture\n"
        "profile_dir: rounds/round_3/profile\n"
    )
    _write(analysis / "analysis_manifest.yaml", original_identity)
    first = tmp_path / "first-import"
    second = tmp_path / "second-import"

    _split_import(source, first)
    found = reuse.discover(first)

    assert found.profile_dir == first / "rounds" / "round_1" / "profile"
    identity = yaml.safe_load(
        (first / "rounds" / "round_1" / "analysis" / "analysis_manifest.yaml").read_text()
    )
    assert identity["capture_id"] == "third-capture"
    assert identity["profile_dir"] == "rounds/round_1/profile"
    assert identity["source_analysis_dir"] == str(analysis)
    imported = _split_import(first, second)
    assert imported.findings and imported.profile
    assert reuse.discover(second).profile_dir == second / "rounds" / "round_1" / "profile"
    assert (analysis / "analysis_manifest.yaml").read_text() == original_identity
