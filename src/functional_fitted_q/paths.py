from pathlib import Path


def output_paths(base_dir: Path, run_tag: str, horizon: int, gamma: float, size: int) -> tuple[Path, Path]:
    # Preserve the original result naming convention so downstream notebooks remain easy to adapt.
    save_path = base_dir / "saved_models" / run_tag / f"h{horizon}_g{gamma}_s{size}"
    result_path = base_dir / "simulation_results_test" / f"CV_{run_tag}_result"
    save_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)
    return save_path, result_path
