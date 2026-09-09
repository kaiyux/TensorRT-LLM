"""Import a previous run's captures and analysis into a perf-optimize workspace.

``--reuse-analysis`` imports the latest evidence-backed findings for a
plan-only analyzer turn. ``--reanalyze`` instead selects the latest usable
capture, even when its analyzer never finished, and preserves any matching
old findings under ``reused_analysis/prior_analysis/``. Fresh derivations
are written into the new campaign's analysis directory. Captures and
findings from different runtime states are never silently paired.

Both workflow layouts and legacy combined analysis directories are supported::

    perf-analyze            perf-optimize
    ------------            -------------
    benchmark_results.md    baseline/benchmark_results.md
    profile_findings.md     rounds/round_<n>/analysis/profile_findings.md
    *.nsys-rep, *.ncu-rep   rounds/round_<n>/profile/profile_manifest.json
    sol_projection.md       sol_projection.md
    sol_work/               sol_work/
    (none)                  roadmap.yaml

The baseline and SOL projection retain their independent import behavior.
A source roadmap is reference material in ``reused_analysis/``; its campaign
state is never imported as the new campaign's live roadmap.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .profile import PROFILE_MANIFEST_NAME, ProfileError, validate_profile_manifest

BASELINE_REPORT_NAME = "benchmark_results.md"
FINDINGS_NAME = "profile_findings.md"
SOL_PROJECTION_NAME = "sol_projection.md"
SOL_WORK_DIRNAME = "sol_work"
ROADMAP_NAME = "roadmap.yaml"
KERNEL_LEDGER_NAME = "kernel_ledger.yaml"

# Where the import parks the provenance record and the source campaign's
# roadmap (prior art the plan-only analyzer reads, never the live ledger).
REUSE_DIRNAME = "reused_analysis"
MANIFEST_NAME = "manifest.md"
PRIOR_ROADMAP_NAME = "prior_roadmap.yaml"
PRIOR_ANALYSIS_DIRNAME = "prior_analysis"

# Sibling artifacts copied alongside the baseline report: the result
# JSONs the evaluator's full-metric diff reads out of the reference
# directory (see ``PerfOptimizeWorkflow._reference_result_dir``).
_BENCHMARK_FILE_GLOBS = ("*.json",)
_BENCHMARK_DIR_GLOBS = ("concurrency_*",)

# Sibling artifacts copied alongside the profile findings: the traces the
# evaluator's kernel comparison and the reporter's before/after read,
# plus the machine-readable analysis products. Deliberately an allowlist
# — a perf-analyze source keeps these next to its reports and checkpoint,
# so copying the whole directory would drag that run's state and report
# files in as well. ``*.sqlite`` is excluded on purpose: it is a
# multi-GB, regenerable export of the ``.nsys-rep`` next to it.
_ANALYSIS_FILE_GLOBS = (
    "*.nsys-rep",
    "*.ncu-rep",
    "nsys_stats*.txt",
    "*ncu*.txt",
    "*ncu*.csv",
    "*ncu*.md",
    KERNEL_LEDGER_NAME,
    "regions.json",
    "sol.json",
)
_ANALYSIS_DIR_GLOBS = ("nsys_analysis*",)

_ROUND_DIR_RE = re.compile(r"round_(\d+)")


class ReuseError(ValueError):
    """Raised when a ``--reuse-analysis`` source cannot be imported."""


def _nonempty_file(path: Path) -> bool:
    """True iff ``path`` is a file holding non-whitespace content.

    Both workflows pre-create blank managed files (``roadmap.yaml``,
    ``sol_projection.md``) on a fresh run, so existence alone would
    "import" an empty placeholder over a real artifact's absence.
    """
    try:
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())
    except OSError:
        return False


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if _nonempty_file(candidate):
            return candidate
    return None


# What makes a round's ``analysis/`` a profile rather than a replan note:
# an artifact from any supported profiler. ``profile.methods`` may omit
# nsys, so ncu-only rounds count too. A perf-optimize round that opened
# replan-only writes findings without capturing any of these.
_PROFILED_ROUND_FILE_GLOBS = ("*.nsys-rep", "nsys_stats*.txt", "*.ncu-rep")


def _has_trace(analysis_dir: Path) -> bool:
    """True iff ``analysis_dir`` holds a capture, not just prose."""
    return any(
        path.is_file() and path.stat().st_size > 0
        for pattern in _PROFILED_ROUND_FILE_GLOBS
        for path in analysis_dir.glob(pattern)
    )


def _valid_profile(profile_dir: Path, *, require_raw: bool = False) -> bool:
    try:
        validate_profile_manifest(profile_dir, require_raw=require_raw)
    except ProfileError:
        return False
    return True


def _profile_for_findings(source: Path, findings: Path) -> Path | None:
    """Resolve evidence from this analysis, without borrowing another round's capture."""
    identity = findings.parent / "analysis_manifest.yaml"
    if identity.is_file():
        try:
            record = yaml.safe_load(identity.read_text(encoding="utf-8"))
            reference = record.get("profile_dir") if isinstance(record, dict) else None
            if not isinstance(reference, str) or not reference.strip():
                return None
            candidate = source / reference
            if not candidate.resolve().is_relative_to(source.resolve()):
                return None
            manifest = validate_profile_manifest(candidate)
            if record.get("capture_id") != manifest["capture_id"]:
                return None
            return candidate
        except (OSError, UnicodeError, yaml.YAMLError, ProfileError):
            return None
    candidate = findings.parent.parent / "profile"
    if candidate.resolve().is_relative_to(source.resolve()) and _valid_profile(candidate):
        return candidate
    if _valid_profile(findings.parent):
        return findings.parent
    if not (findings.parent / PROFILE_MANIFEST_NAME).exists() and _has_trace(findings.parent):
        return findings.parent
    return None


def _round_findings(source: Path) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    for path in source.glob(f"rounds/round_*/analysis/{FINDINGS_NAME}"):
        match = _ROUND_DIR_RE.fullmatch(path.parent.parent.name)
        if match and _nonempty_file(path):
            candidates.append((int(match.group(1)), path))
    return candidates


def _latest_profile(source: Path, *, require_raw: bool = False) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in source.glob(f"rounds/round_*/profile/{PROFILE_MANIFEST_NAME}"):
        match = _ROUND_DIR_RE.fullmatch(path.parent.parent.name)
        if match and _valid_profile(path.parent, require_raw=require_raw):
            candidates.append((int(match.group(1)), path.parent))
    # Legacy campaigns predate the manifest and keep raw captures in analysis/.
    for candidate in source.glob("rounds/round_*/analysis"):
        match = _ROUND_DIR_RE.fullmatch(candidate.parent.name)
        if not match:
            continue
        if not (candidate / PROFILE_MANIFEST_NAME).exists() and _has_raw_capture(candidate):
            candidates.append((int(match.group(1)), candidate))
    if candidates:
        return max(candidates)[1]
    for candidate in (source, source / "profile"):
        if _valid_profile(candidate, require_raw=require_raw):
            return candidate
    if not (source / PROFILE_MANIFEST_NAME).exists() and _has_raw_capture(source):
        return source
    return None


def _has_raw_capture(profile_dir: Path) -> bool:
    return any(
        path.is_file() and path.stat().st_size > 0
        for pattern in ("*.nsys-rep", "*.ncu-rep", "*.sqlite", "*ncu*.csv")
        for path in profile_dir.glob(pattern)
    )


def latest_round_findings(source: Path) -> Path | None:
    """Newest *profiling* ``rounds/round_<n>/analysis/profile_findings.md``.

    Rounds are ranked numerically so ``round_10`` outranks ``round_9``
    (mirroring the reporter's kernel-ledger lookup), and the newest round
    that actually profiled wins — not simply the newest round. A
    perf-optimize campaign opens a round **replan-only** when its standing
    runtime profile remains current, and such a round writes a short
    replan note into its ``analysis/`` with no traces beside it; a plateau
    campaign typically *ends* on one. Importing that note would seed the
    new run with a pointer to a directory it does not have and no traces
    at all, when the earlier profiling round holds the real evidence.

    Falls back to the numerically newest findings when no round carries a
    trace — an import of prose beats importing nothing.
    """
    candidates = _round_findings(source)
    if not candidates:
        return None
    profiled = [
        (number, path)
        for number, path in candidates
        if _profile_for_findings(source, path) is not None
    ]
    return max(profiled or candidates)[1]


@dataclass(frozen=True)
class DiscoveredAnalysis:
    """What a ``--reuse-analysis`` source actually offers.

    Every field is ``None`` when the source does not carry that artifact
    — the import copies what it found and says so.
    """

    source: Path
    baseline_report: Path | None = None
    findings: Path | None = None
    sol_projection: Path | None = None
    sol_work: Path | None = None
    prior_roadmap: Path | None = None
    profile_dir: Path | None = None

    @property
    def kernel_ledger(self) -> Path | None:
        """The findings' sibling ``kernel_ledger.yaml``, when present."""
        if self.findings is None:
            return None
        ledger = self.findings.parent / KERNEL_LEDGER_NAME
        return ledger if _nonempty_file(ledger) else None

    @property
    def is_empty(self) -> bool:
        """True when no baseline, findings, or complete capture is reusable."""
        return self.baseline_report is None and self.findings is None and self.profile_dir is None


def discover(source: str | Path, *, reanalyze: bool = False) -> DiscoveredAnalysis:
    """Locate reusable artifacts in ``source`` (either workspace layout).

    ``reanalyze`` selects captures independently of completed findings.
    The default selects the newest evidence-backed findings and their own
    capture, falling back to independent captures when no findings exist.

    Raises :class:`ReuseError` when ``source`` is not a directory; an
    existing directory with nothing reusable comes back with
    ``is_empty`` set, so the caller can report what it probed for.
    """
    root = Path(source).expanduser()
    if not root.is_dir():
        raise ReuseError(f"--reuse-analysis source is not a directory: {root}")
    findings = latest_round_findings(root) or _first_existing(root / FINDINGS_NAME)
    profile_dir = _profile_for_findings(root, findings) if findings is not None else None
    if reanalyze:
        profile_dir = _latest_profile(root, require_raw=True)
        candidates = _round_findings(root)
        flat_findings = _first_existing(root / FINDINGS_NAME)
        if flat_findings is not None:
            candidates.append((0, flat_findings))
        matching = [
            (number, path)
            for number, path in candidates
            if profile_dir is not None and _profile_for_findings(root, path) == profile_dir
        ]
        findings = max(matching)[1] if matching else None
    elif findings is None:
        profile_dir = _latest_profile(root)
    sol_work = root / SOL_WORK_DIRNAME
    return DiscoveredAnalysis(
        source=root,
        baseline_report=_first_existing(
            root / "baseline" / BASELINE_REPORT_NAME, root / BASELINE_REPORT_NAME
        ),
        findings=findings,
        sol_projection=_first_existing(root / SOL_PROJECTION_NAME),
        sol_work=sol_work if sol_work.is_dir() and any(sol_work.iterdir()) else None,
        prior_roadmap=_first_existing(root / ROADMAP_NAME),
        profile_dir=profile_dir,
    )


@dataclass
class ImportedAnalysis:
    """What :func:`import_analysis` copied into the new workspace."""

    source: Path
    baseline_report: bool = False
    findings: bool = False
    sol_projection: bool = False
    sol_work: bool = False
    kernel_ledger: bool = False
    prior_roadmap: bool = False
    profile: bool = False
    reanalyze: bool = False
    # ``(source, destination)`` pairs, in copy order — the manifest body.
    copied: list[tuple[Path, Path]] = field(default_factory=list)
    manifest_path: Path | None = None

    def summary(self) -> str:
        """One-line human summary of what the import brought in."""
        parts = [
            name
            for name, present in (
                ("baseline benchmark", self.baseline_report),
                ("profile findings", self.findings),
                ("profile captures", self.profile),
                ("SOL projection", self.sol_projection),
                ("kernel ledger", self.kernel_ledger),
                ("prior roadmap (as reference)", self.prior_roadmap),
            )
            if present
        ]
        return ", ".join(parts) if parts else "nothing"


def _copy_file(src: Path, dst: Path, imported: ImportedAnalysis) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    imported.copied.append((src, dst))


def _copy_siblings(
    anchor: Path,
    dst_dir: Path,
    file_globs: tuple[str, ...],
    dir_globs: tuple[str, ...],
    imported: ImportedAnalysis,
) -> None:
    """Copy ``anchor``'s allowlisted sibling artifacts into ``dst_dir``.

    ``anchor`` is the artifact whose directory is being harvested (the
    baseline report / the profile findings); it is skipped itself, having
    already been copied to its canonical destination name. Dotfiles are
    skipped too: ``pathlib`` globs match them (unlike a shell), so
    ``*.json`` would otherwise import the source run's
    ``.perf_analyze_state.json`` checkpoint.
    """
    src_dir = anchor.parent
    for pattern in file_globs:
        for path in sorted(src_dir.glob(pattern)):
            if path.is_file() and path != anchor and not path.name.startswith("."):
                _copy_file(path, dst_dir / path.name, imported)
    for pattern in dir_globs:
        for path in sorted(src_dir.glob(pattern)):
            if path.is_dir():
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(path, dst_dir / path.name, dirs_exist_ok=True)
                imported.copied.append((path, dst_dir / path.name))


def _copy_profile(src: Path, dst: Path, imported: ImportedAnalysis) -> None:
    """Copy validated captures, preserving every manifest-relative reference."""
    manifest_path = src / PROFILE_MANIFEST_NAME
    if manifest_path.exists():
        manifest = validate_profile_manifest(src)
        references = {
            reference
            for result in manifest["methods"].values()
            for reference in result.get("artifacts", [])
        }
        references.update(manifest.get("artifacts", []))
        for reference in sorted(references):
            _copy_file(src / reference, dst / reference, imported)
        _copy_file(manifest_path, dst / PROFILE_MANIFEST_NAME, imported)
    else:
        # Give legacy captures a portable identity without inventing provenance
        # the old workflow did not record. Source paths remain in the runtime.
        methods = {}
        patterns = {"nsys": ("*.nsys-rep", "*.sqlite"), "ncu": ("*.ncu-rep", "*ncu*.csv")}
        for method, globs in patterns.items():
            artifacts = sorted(
                {
                    path.name
                    for pattern in globs
                    for path in src.glob(pattern)
                    if path.is_file() and path.stat().st_size > 0
                }
            )
            if artifacts:
                for reference in artifacts:
                    path = src / reference
                    if not path.resolve().is_relative_to(src.resolve()):
                        raise ReuseError(f"Legacy capture escapes its directory: {path}")
                    _copy_file(path, dst / reference, imported)
                methods[method] = {
                    "status": "captured",
                    "command": "unavailable: legacy capture command was not recorded",
                    "artifacts": artifacts,
                }
            else:
                methods[method] = {
                    "status": "unavailable",
                    "reason": "No reusable raw capture for this method in the legacy source",
                }
        unavailable = "unavailable: provenance was not recorded in this legacy capture"
        manifest = {
            "schema_version": 1,
            "capture_id": f"legacy:{src.resolve()}",
            "runtime": {
                "serve_command": unavailable,
                "benchmark_command": unavailable,
                "config": unavailable,
                "build": unavailable,
                "import_path": unavailable,
                "source": str(src.resolve()),
            },
            "methods": methods,
        }
        dst.mkdir(parents=True, exist_ok=True)
        (dst / PROFILE_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    validate_profile_manifest(dst)
    imported.profile = True


def _render_manifest(imported: ImportedAnalysis, workspace: Path) -> str:
    lines = [
        "# Reused analysis",
        "",
        f"This campaign was launched with `--reuse-analysis {imported.source}`.",
        "",
        "The artifacts below were copied from that run instead of being",
        "re-derived. Round 1's analyzer interprets the copied captures"
        if imported.reanalyze
        else "re-derived. Round 1's analyzer plans from the imported findings",
        "without launching a server or profiler. Prior analysis is reference",
        "only; reanalysis writes fresh derivations in this workspace. No",
        "capture is taken by the analyzer. Every",
        "measurement they contain describes the source run's system — the",
        "baseline numbers this campaign's gains are computed against were",
        "measured there, not here.",
        "",
        "| artifact | source | destination |",
        "| --- | --- | --- |",
    ]
    for src, dst in imported.copied:
        try:
            shown_dst = str(dst.relative_to(workspace))
        except ValueError:
            shown_dst = str(dst)
        lines.append(f"| `{dst.name}` | `{src}` | `{shown_dst}` |")
    if not imported.copied:
        lines.append("| _(nothing)_ | | |")
    lines.append("")
    return "\n".join(lines)


def import_analysis(
    discovered: DiscoveredAnalysis,
    *,
    workspace: Path,
    baseline_dir: Path,
    analysis_dir: Path,
    sol_projection_path: Path,
    sol_work_dir: Path,
    reuse_dir: Path,
    profile_dir: Path | None = None,
    reanalyze: bool = False,
) -> ImportedAnalysis:
    """Copy ``discovered``'s artifacts into a fresh perf-optimize workspace.

    Destinations are the workspace's canonical paths, so every downstream
    stage reads imported artifacts exactly where it reads freshly
    produced ones: the baseline report and its result JSONs land in
    ``baseline/``, the findings in round 1's ``analysis/`` and captures in
    ``profile_dir`` when supplied, the projection at ``sol_projection.md`` (with
    ``sol_work/`` beside it). The source roadmap is parked in
    ``reused_analysis/`` as prior art — never as the live ledger.

    ``reanalyze`` requires raw evidence and leaves any previous derivations
    in ``reused_analysis/prior_analysis/`` for reference. Legacy callers that
    omit ``profile_dir`` keep the combined analysis directory layout.

    Raises :class:`ReuseError` when nothing is reusable or reanalysis has no
    usable raw capture.
    """
    if reanalyze:
        discovered = discover(discovered.source, reanalyze=True)
        if discovered.profile_dir is None:
            raise ReuseError(
                f"--reanalyze requires usable raw captures in {discovered.source}; "
                "profile findings or unavailable-method notes alone cannot be reanalyzed"
            )
        profile_dir = profile_dir or analysis_dir.parent / "profile"
    if discovered.is_empty:
        raise ReuseError(
            f"no reusable analysis found in {discovered.source}. Looked for a "
            f"baseline report (baseline/{BASELINE_REPORT_NAME} or "
            f"{BASELINE_REPORT_NAME}) and profile findings "
            f"(rounds/round_<n>/analysis/{FINDINGS_NAME} or {FINDINGS_NAME}); "
            f"or a valid {PROFILE_MANIFEST_NAME}; "
            f"pass a perf-analyze or perf-optimize workspace that completed at "
            f"least one of those stages."
        )

    imported = ImportedAnalysis(source=discovered.source, reanalyze=reanalyze)

    if discovered.baseline_report is not None:
        _copy_file(discovered.baseline_report, baseline_dir / BASELINE_REPORT_NAME, imported)
        _copy_siblings(
            discovered.baseline_report,
            baseline_dir,
            _BENCHMARK_FILE_GLOBS,
            _BENCHMARK_DIR_GLOBS,
            imported,
        )
        imported.baseline_report = True

    if discovered.findings is not None:
        findings_dir = reuse_dir / PRIOR_ANALYSIS_DIRNAME if reanalyze else analysis_dir
        _copy_file(discovered.findings, findings_dir / FINDINGS_NAME, imported)
        file_globs = _ANALYSIS_FILE_GLOBS
        if profile_dir is not None:
            file_globs = tuple(
                pattern for pattern in file_globs if pattern not in ("*.nsys-rep", "*.ncu-rep")
            )
        _copy_siblings(
            discovered.findings,
            findings_dir,
            file_globs,
            _ANALYSIS_DIR_GLOBS,
            imported,
        )
        imported.findings = not reanalyze
        imported.kernel_ledger = not reanalyze and discovered.kernel_ledger is not None

    if discovered.profile_dir is not None:
        target = profile_dir or analysis_dir
        has_manifest = (discovered.profile_dir / PROFILE_MANIFEST_NAME).exists()
        if profile_dir is None and not has_manifest and discovered.findings is not None:
            # Preserve the original import layout and its exclusion of large
            # regenerable sqlite exports for callers that did not opt into split dirs.
            imported.profile = _has_raw_capture(discovered.profile_dir)
        elif has_manifest or _has_raw_capture(discovered.profile_dir):
            _copy_profile(discovered.profile_dir, target, imported)
        if imported.profile and imported.findings and (target / PROFILE_MANIFEST_NAME).is_file():
            capture = validate_profile_manifest(target)
            analysis_identity = {
                "schema_version": 1,
                "analysis_id": f"{analysis_dir.parent.name}-imported",
                "capture_id": capture["capture_id"],
                "profile_dir": str(target.resolve().relative_to(workspace.resolve())),
                "imported": True,
                "source_analysis_dir": str(discovered.findings.parent.resolve()),
            }
            # The imported findings retain their capture identity, but their
            # source round's directory names do not belong to this workspace.
            (analysis_dir / "analysis_manifest.yaml").write_text(
                yaml.safe_dump(analysis_identity, sort_keys=False), encoding="utf-8"
            )

    if discovered.sol_projection is not None:
        _copy_file(discovered.sol_projection, sol_projection_path, imported)
        imported.sol_projection = True

    if discovered.sol_work is not None:
        sol_work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(discovered.sol_work, sol_work_dir, dirs_exist_ok=True)
        imported.copied.append((discovered.sol_work, sol_work_dir))
        imported.sol_work = True

    if discovered.prior_roadmap is not None:
        _copy_file(discovered.prior_roadmap, reuse_dir / PRIOR_ROADMAP_NAME, imported)
        imported.prior_roadmap = True

    manifest_path = reuse_dir / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_render_manifest(imported, workspace), encoding="utf-8")
    imported.manifest_path = manifest_path
    return imported


__all__ = [
    "BASELINE_REPORT_NAME",
    "FINDINGS_NAME",
    "KERNEL_LEDGER_NAME",
    "MANIFEST_NAME",
    "PRIOR_ROADMAP_NAME",
    "PRIOR_ANALYSIS_DIRNAME",
    "REUSE_DIRNAME",
    "DiscoveredAnalysis",
    "ImportedAnalysis",
    "ReuseError",
    "discover",
    "import_analysis",
    "latest_round_findings",
]
