from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any, Iterator

import yaml


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a unique same-directory staging path and durably replace ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        if not temporary.is_file():
            raise RuntimeError(
                f"atomic writer did not produce a regular file: {temporary}"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(text)


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    _atomic_text(
        path, json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def atomic_yaml(path: Path, document: dict[str, Any]) -> None:
    _atomic_text(path, yaml.safe_dump(document, sort_keys=True))


def file_size(path: Path) -> int:
    """Read file size for detecting missing or truncated output files."""
    return Path(path).stat().st_size


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.environ.get("FFQI_SOURCE_COMMIT", "UNKNOWN")


def environment_manifest() -> str:
    import torch

    lines = [
        f"hostname={platform.node()}",
        f"python={platform.python_version()}",
        f"torch={torch.__version__}",
        f"cuda={torch.version.cuda}",
        f"cuda_available={torch.cuda.is_available()}",
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}",
        f"slurm_job_id={os.environ.get('SLURM_JOB_ID')}",
        f"slurm_array_task_id={os.environ.get('SLURM_ARRAY_TASK_ID')}",
    ]
    return "\n".join(lines) + "\n"


def enforce_resolved_document(path: Path, document: dict[str, Any]) -> None:
    text = yaml.safe_dump(document, sort_keys=True)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(
            f"resolved document changed inside existing run directory: {path}"
        )
    atomic_yaml(path, document)


def archive_incomplete_attempt_on_source_change(
    run_root: Path, current_source_commit: str
) -> Path | None:
    """Preserve a partial attempt before resuming it with a different code commit."""
    run_root = Path(run_root)
    if (run_root / "_SUCCESS").is_file():
        return None
    resolved_path = run_root / "config_resolved.yaml"
    if not resolved_path.is_file():
        return None
    resolved = yaml.safe_load(resolved_path.read_text())
    previous_source_commit = str(resolved.get("source_commit", "UNKNOWN"))
    status_path = run_root / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    failed_recoverable = status.get("state") == "FAILED_RECOVERABLE"
    source_changed = previous_source_commit != current_source_commit
    if not source_changed and not failed_recoverable:
        return None
    attempts_root = run_root / "failed_attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt_root = attempts_root / (
        f"attempt-{time.time_ns()}-{previous_source_commit[:12]}"
    )
    attempt_root.mkdir()
    moved = []
    for path in sorted(run_root.iterdir(), key=lambda item: item.name):
        if path.name in {".execution.lock", "failed_attempts"}:
            continue
        os.replace(path, attempt_root / path.name)
        moved.append(path.name)
    atomic_json(
        attempt_root / "archive_receipt.json",
        {
            "schema_version": 1,
            "reason": (
                "failed_recoverable_clean_restart"
                if failed_recoverable
                else "incomplete_attempt_source_commit_changed"
            ),
            "previous_source_commit": previous_source_commit,
            "current_source_commit": current_source_commit,
            "moved_top_level_entries": moved,
            "scientific_config_changed": False,
            "failed_attempt_deleted": False,
        },
    )
    return attempt_root


def write_run_artifact_manifest(
    run_root: Path, relative_paths: list[str]
) -> dict[str, Any]:
    records = []
    for relative in sorted(set(relative_paths)):
        path = run_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"required run artifact is missing or a symlink: {relative}"
            )
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
            }
        )
    document = {
        "schema_version": 1,
        "artifact_count": len(records),
        "artifacts": records,
    }
    atomic_json(run_root / "run_artifact_manifest.json", document)
    return document


def verify_run_artifact_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "run_artifact_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("run artifact manifest is missing or a symlink")
    document = json.loads(manifest_path.read_text())
    records = document.get("artifacts")
    if not isinstance(records, list) or len(records) != int(
        document.get("artifact_count", -1)
    ):
        raise RuntimeError("run artifact manifest inventory is invalid")
    seen = set()
    for record in records:
        relative = record["path"]
        if relative in seen:
            raise RuntimeError("duplicate path in run artifact manifest")
        seen.add(relative)
        path = run_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"manifest artifact is missing or a symlink: {relative}")
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"manifest artifact verification failed: {relative}")
    return document
