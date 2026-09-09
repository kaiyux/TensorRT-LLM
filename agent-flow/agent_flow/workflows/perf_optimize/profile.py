"""Validate independently checkpointed profiler captures.

The profiler writes ``profile_manifest.json`` last. A captured method lists
its raw reports or offline exports, capture command, and any supporting
artifacts. A method that could not run records an explicit reason instead.
The manifest identifies the runtime whose immutable evidence the analyzer
will interpret, independently of any particular analysis of that evidence.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROFILE_MANIFEST_NAME = "profile_manifest.json"
PROFILE_SCHEMA_VERSION = 1
_METHOD_REPORT_SUFFIXES = {"nsys": (".nsys-rep", ".sqlite"), "ncu": (".ncu-rep", ".csv")}
_RUNTIME_FIELDS = ("serve_command", "benchmark_command", "config", "build", "import_path")


class ProfileError(ValueError):
    """Raised when a capture is incomplete or its artifact references are unsafe."""


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _artifact_path(root: Path, reference: object) -> Path:
    if not _nonempty_string(reference):
        raise ProfileError("Profile artifact references must be nonempty relative paths")
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts:
        raise ProfileError(f"Profile artifact must stay inside the capture directory: {reference}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ProfileError(f"Profile artifact escapes the capture directory: {reference}")
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ProfileError(f"Profile artifact is missing or empty: {reference}")
    return path


def validate_profile_manifest(
    profile_dir: Path,
    *,
    required_methods: Sequence[str] | None = None,
    require_raw: bool = False,
) -> dict[str, Any]:
    """Read and validate a completed profiler manifest and its artifacts.

    Args:
        profile_dir: Directory containing the manifest and capture artifacts.
        required_methods: Configured methods, each of which must record a result.
        require_raw: Require at least one reusable report or offline export.

    Returns:
        The parsed manifest, including any additional provenance or coverage fields.

    Raises:
        ProfileError: The manifest is incomplete, malformed, unsafe, or lacks raw
            evidence when ``require_raw`` is true. Explicitly justified unavailable
            methods otherwise count as completed profiling work.
    """
    profile_dir = Path(profile_dir)
    manifest_path = profile_dir / PROFILE_MANIFEST_NAME
    try:
        if not manifest_path.resolve().is_relative_to(profile_dir.resolve()):
            raise ProfileError(f"Profile manifest escapes the capture directory: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Cannot read profile manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ProfileError(f"Profile manifest must be an object: {manifest_path}")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ProfileError(f"Unsupported profile manifest schema_version in {manifest_path}")
    if not _nonempty_string(manifest.get("capture_id")):
        raise ProfileError("Profile manifest requires a nonempty capture_id")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or any(
        not _nonempty_string(runtime.get(name)) for name in _RUNTIME_FIELDS
    ):
        raise ProfileError(f"Profile runtime requires nonempty strings for {_RUNTIME_FIELDS}")
    methods = manifest.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ProfileError("Profile manifest requires a nonempty methods object")
    if required_methods is not None:
        missing = set(required_methods) - methods.keys()
        if missing:
            raise ProfileError(f"Profile manifest is missing configured methods: {sorted(missing)}")
    captured = False
    try:
        provenance_artifacts = manifest.get("artifacts", [])
        if not isinstance(provenance_artifacts, list):
            raise ProfileError("Profile manifest artifacts must be a list")
        for reference in provenance_artifacts:
            _artifact_path(profile_dir, reference)
        for method, result in methods.items():
            if method not in _METHOD_REPORT_SUFFIXES or not isinstance(result, dict):
                raise ProfileError(f"Unsupported or malformed profile method: {method}")
            status = result.get("status")
            if status == "unavailable":
                if not _nonempty_string(result.get("reason")):
                    raise ProfileError(f"Unavailable profile method {method} requires a reason")
                artifacts = result.get("artifacts", [])
            elif status == "captured":
                if not _nonempty_string(result.get("command")):
                    raise ProfileError(f"Captured profile method {method} requires its command")
                artifacts = result.get("artifacts")
                if not isinstance(artifacts, list) or not artifacts:
                    raise ProfileError(f"Captured profile method {method} requires artifacts")
            else:
                raise ProfileError(f"Profile method {method} has incomplete status {status!r}")
            if not isinstance(artifacts, list):
                raise ProfileError(f"Profile method {method} artifacts must be a list")
            paths = [_artifact_path(profile_dir, reference) for reference in artifacts]
            if status == "captured":
                if not any(path.suffix in _METHOD_REPORT_SUFFIXES[method] for path in paths):
                    raise ProfileError(
                        f"Captured profile method {method} lacks a raw report or export"
                    )
                captured = True
    except OSError as exc:
        raise ProfileError(f"Cannot inspect profile artifacts in {profile_dir}: {exc}") from exc
    if require_raw and not captured:
        raise ProfileError("Reanalysis requires usable raw captures; all methods are unavailable")
    return manifest


__all__ = [
    "PROFILE_MANIFEST_NAME",
    "PROFILE_SCHEMA_VERSION",
    "ProfileError",
    "validate_profile_manifest",
]
