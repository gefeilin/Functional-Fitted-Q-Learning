"""Execute the analysis notebook using this interpreter at the project root."""

from pathlib import Path
import json
import os
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

root = Path(__file__).resolve().parents[1]
cache = root / "outputs/.cache"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("IPYTHONDIR", str(cache / "ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(cache / "jupyter"))
os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
path = root / "examples/paper_results.ipynb"
notebook = nbformat.read(path, as_version=4)
name = notebook.metadata.kernelspec.name
if notebook.metadata.kernelspec.language != "python":
    raise ValueError("This runner expects the documented Python notebook")
with tempfile.TemporaryDirectory(prefix="kernel-", dir=cache) as directory:
    spec = Path(directory) / name
    spec.mkdir()
    (spec / "kernel.json").write_text(
        json.dumps(
            dict(
                argv=[
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                display_name="FFQI analysis Python",
                language="python",
            )
        )
    )
    manager = KernelManager(
        kernel_name=name, kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory])
    )
    client = NotebookClient(
        notebook, km=manager, timeout=180, resources={"metadata": {"path": str(root)}}
    )
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
nbformat.write(notebook, path)
code = [c for c in notebook.cells if c.cell_type == "code"]
if any(o.output_type == "error" for c in code for o in c.outputs):
    raise AssertionError("Notebook contains error output")
print(f"Executed {len(code)} code cells: {path.relative_to(root)}")
