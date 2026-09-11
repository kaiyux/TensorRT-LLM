"""Checks for the profiler's independently reusable capture manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_flow.workflows.perf_optimize.profile import (
    PROFILE_MANIFEST_NAME,
    PROFILE_REPORT_NAME,
    ProfileError,
    validate_profile_manifest,
)


def _manifest(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    (root / "server_nsys.nsys-rep").write_bytes(b"raw capture")
    return {
        "schema_version": 1,
        "capture_id": "round_1-abc123",
        "runtime": {
            "serve_command": "trtllm-serve model",
            "benchmark_command": "trtllm-bench --concurrency 8",
            "config": "{backend: pytorch}",
            "build": "abc123; clean checkout",
            "import_path": "/repo/tensorrt_llm/__init__.py",
        },
        "methods": {
            "nsys": {
                "status": "captured",
                "command": "nsys profile --output server_nsys ...",
                "artifacts": ["server_nsys.nsys-rep"],
            },
            "ncu": {"status": "unavailable", "reason": "ncu is not installed"},
        },
    }


def _save(root: Path, manifest: dict) -> None:
    (root / PROFILE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_complete_capture_allows_justified_partial_tool_availability(tmp_path):
    manifest = _manifest(tmp_path)
    _save(tmp_path, manifest)

    assert (
        validate_profile_manifest(tmp_path, required_methods=["nsys", "ncu"], require_raw=True)
        == manifest
    )


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_fresh_capture_requires_nonempty_profiler_report_but_legacy_reuse_does_not(
    tmp_path, content
):
    manifest = _manifest(tmp_path)
    _save(tmp_path, manifest)
    if content is not None:
        (tmp_path / PROFILE_REPORT_NAME).write_text(content, encoding="utf-8")

    validate_profile_manifest(tmp_path, require_raw=True)
    with pytest.raises(ProfileError, match="profiler_report.md"):
        validate_profile_manifest(tmp_path, require_report=True)


def test_fresh_capture_accepts_report_and_rejects_report_symlink_escape(tmp_path):
    root = tmp_path / "profile"
    manifest = _manifest(root)
    _save(root, manifest)
    report = root / PROFILE_REPORT_NAME
    report.write_text("# Capture coverage\n", encoding="utf-8")
    validate_profile_manifest(root, require_report=True)

    report.rename(tmp_path / "external.md")
    report.symlink_to(tmp_path / "external.md")
    with pytest.raises(ProfileError, match="escapes the capture directory"):
        validate_profile_manifest(root, require_report=True)


def test_all_unavailable_completes_stage_but_cannot_be_reanalyzed(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["methods"]["nsys"] = {"status": "unavailable", "reason": "No GPU on host"}
    _save(tmp_path, manifest)

    validate_profile_manifest(tmp_path, required_methods=["nsys", "ncu"])
    with pytest.raises(ProfileError, match="requires usable raw captures"):
        validate_profile_manifest(tmp_path, require_raw=True)


@pytest.mark.parametrize("reference", ["../outside.nsys-rep", "/tmp/outside.nsys-rep", "", None])
def test_manifest_rejects_unsafe_artifact_references(tmp_path, reference):
    manifest = _manifest(tmp_path)
    manifest["methods"]["nsys"]["artifacts"] = [reference]
    _save(tmp_path, manifest)

    with pytest.raises(ProfileError, match="relative paths|inside the capture directory"):
        validate_profile_manifest(tmp_path)


def test_manifest_rejects_symlink_escape(tmp_path):
    root = tmp_path / "profile"
    manifest = _manifest(root)
    external = tmp_path / "external.nsys-rep"
    external.write_bytes(b"different capture")
    (root / "escape.nsys-rep").symlink_to(external)
    manifest["methods"]["nsys"]["artifacts"] = ["escape.nsys-rep"]
    _save(root, manifest)

    with pytest.raises(ProfileError, match="escapes the capture directory"):
        validate_profile_manifest(root)


@pytest.mark.parametrize("missing", [True, False])
def test_captured_report_must_exist_and_be_nonempty(tmp_path, missing):
    manifest = _manifest(tmp_path)
    capture = tmp_path / "server_nsys.nsys-rep"
    capture.unlink() if missing else capture.write_bytes(b"")
    _save(tmp_path, manifest)

    with pytest.raises(ProfileError, match="missing or empty"):
        validate_profile_manifest(tmp_path)


@pytest.mark.parametrize("status", ["pending", "failed", "running", None])
def test_partial_manifest_is_not_a_complete_capture(tmp_path, status):
    manifest = _manifest(tmp_path)
    manifest["methods"]["ncu"]["status"] = status
    _save(tmp_path, manifest)

    with pytest.raises(ProfileError, match="incomplete status"):
        validate_profile_manifest(tmp_path)


def test_capture_requires_status_for_each_configured_method(tmp_path):
    manifest = _manifest(tmp_path)
    del manifest["methods"]["ncu"]
    _save(tmp_path, manifest)

    with pytest.raises(ProfileError, match="missing configured methods.*ncu"):
        validate_profile_manifest(tmp_path, required_methods=["nsys", "ncu"])


def test_unavailable_status_requires_reason(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["methods"]["ncu"]["reason"] = " "
    _save(tmp_path, manifest)

    with pytest.raises(ProfileError, match="requires a reason"):
        validate_profile_manifest(tmp_path)


def test_captured_status_requires_command(tmp_path):
    manifest = _manifest(tmp_path)
    del manifest["methods"]["nsys"]["command"]
    _save(tmp_path, manifest)

    with pytest.raises(ProfileError, match="requires its command"):
        validate_profile_manifest(tmp_path)


@pytest.mark.parametrize(
    "name", ["serve_command", "benchmark_command", "config", "build", "import_path"]
)
def test_runtime_provenance_is_required(tmp_path, name):
    manifest = _manifest(tmp_path)
    del manifest["runtime"][name]
    _save(tmp_path, manifest)

    with pytest.raises(ProfileError, match="Profile runtime requires"):
        validate_profile_manifest(tmp_path)


@pytest.mark.parametrize("content", ["", "[", "[]", '{"schema_version": 2}'])
def test_malformed_or_unsupported_manifest_is_rejected(tmp_path, content):
    (tmp_path / PROFILE_MANIFEST_NAME).write_text(content, encoding="utf-8")

    with pytest.raises(ProfileError):
        validate_profile_manifest(tmp_path)


def test_derived_summary_alone_is_not_a_capture(tmp_path):
    manifest = _manifest(tmp_path)
    (tmp_path / "nsys_stats.txt").write_text("kernel totals", encoding="utf-8")
    manifest["methods"]["nsys"]["artifacts"] = ["nsys_stats.txt"]
    _save(tmp_path, manifest)

    with pytest.raises(ProfileError, match="lacks a raw report or export"):
        validate_profile_manifest(tmp_path)


@pytest.mark.parametrize("method,filename", [("nsys", "capture.sqlite"), ("ncu", "metrics.csv")])
def test_nonempty_offline_exports_support_reanalysis(tmp_path, method, filename):
    manifest = _manifest(tmp_path)
    (tmp_path / filename).write_bytes(b"exported evidence")
    manifest["methods"] = {
        method: {"status": "captured", "command": "profile/export ...", "artifacts": [filename]}
    }
    _save(tmp_path, manifest)

    validate_profile_manifest(tmp_path, require_raw=True)
