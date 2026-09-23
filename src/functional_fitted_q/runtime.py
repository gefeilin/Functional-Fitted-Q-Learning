"""Portable project paths and execution policy; no machine-specific directories."""

from pathlib import Path
import os
import socket


def project_root(start=None):
    paths = [Path(start).resolve()] if start else [Path.cwd(), Path(__file__).resolve()]
    for candidate in paths:
        for parent in (candidate, *candidate.parents):
            if (parent / "project_config.json").is_file():
                return parent
    raise FileNotFoundError("Run from the package or supply its project root")


def require_execution_host():
    """Biowulf login nodes may only submit/manage allocations."""
    host = socket.getfqdn().lower()
    if "biowulf" in host and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use an allocated Biowulf compute node")


def configure_runtime():
    # Set deterministic CUDA settings before importing torch.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(name, "4")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(project_root() / "outputs/.cache/matplotlib")
    )


def deterministic_device(device="cuda"):
    require_execution_host()
    configure_runtime()
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Full training requires a CUDA GPU; use smoke for a CPU example"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    return torch.device(device)
