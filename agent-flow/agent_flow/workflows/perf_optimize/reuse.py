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
    analysis.md             rounds/round_<n>/analysis/analysis.md
    *.nsys-rep, *.ncu-rep   rounds/round_<n>/profile/profile_manifest.json
    sol_projection.md       sol_projection.md
    sol_work/               sol_work/
    (none)                  roadmap.yaml

The baseline and SOL projection retain their independent import behavior.
A source roadmap and kernel ledger are reference material in
``reused_analysis/``. The analyzer authors the new campaign's live roadmap,
kernel ledger, and model revision history.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agent_flow.workflows.perf_analyze.performance_model import MODEL_FILENAME

from .profile import (
    PROFILE_MANIFEST_NAME,
    PROFILE_REPORT_NAME,
    ProfileError,
    validate_profile_manifest,
)

BASELINE_REPORT_NAME = "benchmark_results.md"
FINDINGS_NAME = "analysis.md"
LEGACY_FINDINGS_NAME = "profile_findings.md"
SOL_PROJECTION_NAME = "sol_projection.md"
SOL_WORK_DIRNAME = "sol_work"
ROADMAP_NAME = "roadmap.yaml"
KERNEL_LEDGER_NAME = "kernel_ledger.yaml"

# Where the import parks the provenance record and the source campaign's
# roadmap (prior art the plan-only analyzer reads, never the live ledger).
REUSE_DIRNAME = "reused_analysis"
MANIFEST_NAME = "manifest.md"
PRIOR_ROADMAP_NAME = "prior_roadmap.yaml"
PRIOR_KERNEL_LEDGER_NAME = "kernel_ledger.yaml"
PRIOR_PERFORMANCE_MODEL_NAME = "prior_performance_model.yaml"
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
    "regions.json",
    "sol.json",
    "taxonomy.json",
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
    # Split-layout analyses get their identity only after the analyzer gate
    # succeeds. A completed sibling capture does not complete partial findings.
    candidate = findings.parent.parent / "profile"
    if (
        candidate != findings.parent
        and candidate.resolve().is_relative_to(source.resolve())
        and candidate.exists()
    ):
        return None
    if _valid_profile(findings.parent):
        return findings.parent
    if not (findings.parent / PROFILE_MANIFEST_NAME).exists() and _has_trace(findings.parent):
        return findings.parent
    return None


def _round_findings(source: Path) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    for directory in source.glob("rounds/round_*/analysis"):
        match = _ROUND_DIR_RE.fullmatch(directory.parent.name)
        path = _first_existing(directory / FINDINGS_NAME, directory / LEGACY_FINDINGS_NAME)
        if match and path is not None:
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
    """Newest *profiling* ``rounds/round_<n>/analysis/analysis.md``.

    Rounds are ranked numerically so ``round_10`` outranks ``round_9``
    (mirroring the reporter's kernel-ledger lookup), and the newest round
    that actually profiled wins — not simply the newest round. A
    perf-optimize campaign opens a round **replan-only** when its standing
    runtime profile remains current, and such a round writes a short
    replan note into its ``analysis/`` with no traces beside it; a plateau
    campaign typically *ends* on one. Importing that note would seed the
    new run with a pointer to a directory it does not have and no traces
    at all, when the earlier profiling round holds the real evidence.

    Legacy campaigns without split capture directories or analysis manifests
    retain their prose-only fallback. Modern analyses require completion identity.
    """
    candidates = _round_findings(source)
    if not candidates:
        return None
    profiled = [
        (number, path)
        for number, path in candidates
        if _profile_for_findings(source, path) is not None
    ]
    if profiled:
        return max(profiled)[1]
    split_layout = any(source.glob("rounds/round_*/profile")) or any(
        source.glob("rounds/round_*/analysis/analysis_manifest.yaml")
    )
    return None if split_layout else max(candidates)[1]


def _profile_for_ledger(source: Path, ledger: Path) -> Path | None:
    """Resolve a ledger's capture without inferring it from round order.

    An analysis manifest is authoritative. Replan ledgers without one can
    name the standing profile's analysis artifacts in ``source``; follow
    that existing path within the source workspace to its capture. A
    source that is prose or unavailable cannot establish that relationship.
    """
    if (ledger.parent / "analysis_manifest.yaml").exists():
        return _profile_for_findings(source, ledger)
    try:
        data = yaml.safe_load(ledger.read_text(encoding="utf-8"))
        reference = data.get("source") if isinstance(data, dict) else None
        if isinstance(reference, str) and reference.strip():
            artifact = (source / reference).resolve()
            root = source.resolve()
            if artifact.exists() and artifact.is_relative_to(root):
                directory = artifact if artifact.is_dir() else artifact.parent
                for candidate in (directory, *directory.parents):
                    if not candidate.is_relative_to(root):
                        break
                    profile = _profile_for_findings(source, candidate / FINDINGS_NAME)
                    if profile is not None:
                        return profile
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    return _profile_for_findings(source, ledger)


def _latest_kernel_ledger(
    source: Path, findings: Path | None, profile_dir: Path | None
) -> Path | None:
    """Newest ledger for the selected capture, with findings-sibling fallback.

    A newer replan may update the model without producing full findings.
    Import that knowledge only when its provenance identifies the selected
    capture; unrelated or unresolvable newer ledgers cannot supersede the
    selected findings' own ledger.
    """
    sibling = _first_existing(findings.parent / KERNEL_LEDGER_NAME) if findings else None
    if profile_dir is None:
        return sibling
    candidates: list[tuple[int, Path]] = []
    for ledger in source.glob(f"rounds/round_*/analysis/{KERNEL_LEDGER_NAME}"):
        match = _ROUND_DIR_RE.fullmatch(ledger.parent.parent.name)
        if not match or not _nonempty_file(ledger):
            continue
        profile = _profile_for_ledger(source, ledger)
        if ledger == sibling or (
            profile is not None and profile.resolve() == profile_dir.resolve()
        ):
            candidates.append((int(match.group(1)), ledger))
    return max(candidates)[1] if candidates else sibling


def _model_builds(path: Path) -> dict[int | None, str] | None:
    """Read explicit per-point build identities without adopting a prior model."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        points = data.get("points") if isinstance(data, dict) else None
        if not isinstance(points, list) or not points:
            return None
        builds = {}
        for point in points:
            conditions = point.get("operating_point") if isinstance(point, dict) else None
            build = conditions.get("build") if isinstance(conditions, dict) else None
            if not (isinstance(build, str) and build.strip() or isinstance(build, dict) and build):
                return None
            concurrency = point.get("concurrency")
            if concurrency is not None and (type(concurrency) is not int or concurrency <= 0):
                return None
            if concurrency in builds:
                return None
            builds[concurrency] = json.dumps(build, sort_keys=True, separators=(",", ":"))
        return builds
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, ValueError):
        return None


def _latest_performance_model(
    source: Path, findings: Path | None, profile_dir: Path | None
) -> tuple[Path | None, str]:
    """Select a corrected model only within the chosen analysis/capture scope.

    A newer replan can supersede the selected analysis only when its capture
    provenance resolves to the same directory and its build identities match.
    Without both proofs, retain the selected analysis's model as prior art.
    """
    sibling = _first_existing(findings.parent / MODEL_FILENAME) if findings else None
    final_model = source / "final_verification" / "analysis" / MODEL_FILENAME
    if _nonempty_file(final_model):
        try:
            identity = yaml.safe_load(
                (final_model.parent / "analysis_manifest.yaml").read_text(encoding="utf-8")
            )
            if (
                isinstance(identity, dict)
                and identity.get("mode") == "final_reconciliation"
                and identity.get("capture_id") is None
                and identity.get("profile_dir") is None
                and identity.get("capture_unavailable")
            ):
                return final_model, (
                    "Final reconciled model without reusable capture provenance. "
                    "Final measurements come from QA; inherited findings retain their "
                    "source scope. Import as prior reference, not a captured model."
                )
        except (OSError, UnicodeError, yaml.YAMLError):
            pass
    scope = (
        "Selected analysis's model; source capture/build conditions must be checked "
        "against the new campaign before adopting any bound."
        if sibling is not None
        else ""
    )
    if profile_dir is None:
        return sibling, scope
    builds = _model_builds(sibling) if sibling is not None else None
    capture_build = None
    if builds is None:
        try:
            capture = validate_profile_manifest(profile_dir)
            capture_build = json.dumps(capture["runtime"]["build"], sort_keys=True)
        except ProfileError:
            return sibling, scope

    def matching_builds(path: Path) -> bool:
        candidate_builds = _model_builds(path)
        if candidate_builds is None:
            return False
        if builds is not None:
            return candidate_builds == builds
        return all(build == capture_build for build in candidate_builds.values())

    candidates = []
    for path in source.glob(f"rounds/round_*/analysis/{MODEL_FILENAME}"):
        match = _ROUND_DIR_RE.fullmatch(path.parent.parent.name)
        if not match or not _nonempty_file(path):
            continue
        profile = _profile_for_ledger(source, path)
        if path == sibling or (
            profile is not None
            and profile.resolve() == profile_dir.resolve()
            and matching_builds(path)
        ):
            candidates.append((int(match.group(1)), path))
    selected = max(candidates)[1] if candidates else sibling
    if selected is not None and selected != sibling:
        scope = (
            "Latest model with provenance matching the selected capture and explicit "
            "build identities; source operating conditions still require a campaign fit check."
        )
    if (
        _nonempty_file(final_model)
        and (final_model.parent / "analysis_manifest.yaml").is_file()
        and _profile_for_ledger(source, final_model) == profile_dir
    ):
        selected = final_model
        scope = (
            "Final reconciled model with matching selected capture and build identities; "
            "source workload and timing conditions still require a campaign fit check."
            if matching_builds(final_model)
            else "Final reconciled model: final build differs from the selected capture "
            "or its build identity is unproven. Inherited capture measurements describe "
            "the earlier build; import this as scoped prior reference, not a captured "
            "model of the final runtime."
        )
    return selected, scope


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
    performance_model: Path | None = None
    performance_model_scope: str = ""

    @property
    def kernel_ledger(self) -> Path | None:
        """Latest kernel/model knowledge attributable to the selected capture."""
        return _latest_kernel_ledger(self.source, self.findings, self.profile_dir)

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
    findings = latest_round_findings(root) or _first_existing(
        root / FINDINGS_NAME, root / LEGACY_FINDINGS_NAME
    )
    profile_dir = _profile_for_findings(root, findings) if findings is not None else None
    if reanalyze:
        profile_dir = _latest_profile(root, require_raw=True)
        candidates = _round_findings(root)
        flat_findings = _first_existing(root / FINDINGS_NAME, root / LEGACY_FINDINGS_NAME)
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
    prior_model, model_scope = _latest_performance_model(root, findings, profile_dir)
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
        performance_model=prior_model,
        performance_model_scope=model_scope,
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
    performance_model: bool = False
    performance_model_scope: str = ""
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
                ("kernel ledger (as reference)", self.kernel_ledger),
                ("performance model (as reference)", self.performance_model),
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
        if _nonempty_file(src / PROFILE_REPORT_NAME):
            validate_profile_manifest(src, require_report=True)
            references.add(PROFILE_REPORT_NAME)
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
        report = src / PROFILE_REPORT_NAME
        if _nonempty_file(report):
            if not report.resolve().is_relative_to(src.resolve()):
                raise ReuseError(f"Profiler report escapes its directory: {report}")
            _copy_file(report, dst / PROFILE_REPORT_NAME, imported)
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
        "measurement they contain describes the source run's system.",
        "A usable imported baseline is retained; a missing or unusable baseline",
        "is measured by this campaign. The benchmarker progress entry records",
        "any new baseline, whose report replaces the imported report.",
        "",
        "The imported kernel ledger and its model revision history are prior",
        "art only. Resolve its evidence relative to its original source",
        "directory below. The analyzer writes a fresh kernel ledger with",
        "this campaign's roadmap references and model revision history.",
        "",
        "The prior performance model is read-only reference, never the new campaign's",
        "current performance_model.yaml. Preserve its corrected assumptions rather than",
        "silently reverting to the original SOL projection. Resolve cited evidence in the",
        "source workspace; the analyzer must write and validate a fresh current model.",
        imported.performance_model_scope
        if imported.performance_model
        else "No prior performance model was available in the selected source analysis.",
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
    ``sol_work/`` beside it). The source roadmap and kernel ledger are parked
    in ``reused_analysis/`` as prior art, including their original history.
    The analyzer authors this campaign's live ledger in either reuse mode.

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

    prior_ledger = discovered.kernel_ledger
    if prior_ledger is not None:
        _copy_file(prior_ledger, reuse_dir / PRIOR_KERNEL_LEDGER_NAME, imported)
        imported.kernel_ledger = True

    if discovered.performance_model is not None:
        _copy_file(
            discovered.performance_model,
            reuse_dir / PRIOR_PERFORMANCE_MODEL_NAME,
            imported,
        )
        imported.performance_model = True
        imported.performance_model_scope = discovered.performance_model_scope

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
    "PRIOR_KERNEL_LEDGER_NAME",
    "PRIOR_PERFORMANCE_MODEL_NAME",
    "PRIOR_ANALYSIS_DIRNAME",
    "REUSE_DIRNAME",
    "DiscoveredAnalysis",
    "ImportedAnalysis",
    "ReuseError",
    "discover",
    "import_analysis",
    "latest_round_findings",
]
