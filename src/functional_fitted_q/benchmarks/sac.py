import argparse

import numpy as np
from d3rlpy.algos import SACConfig

from .benchmark_runner import BenchmarkSpec, run_benchmark_experiment

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/benchmarks")
    return parser

def create_sac(params, gamma, batch_size, device):
    return SACConfig(
        actor_learning_rate=float(params["lr_actor"]),
        critic_learning_rate=float(params["lr_critic"]),
        temp_learning_rate=float(params["lr_alpha"]),
        batch_size=batch_size,
        tau=float(params["tau"]),
        gamma=gamma,
    ).create(device=device)

def main():
    args = build_parser().parse_args()
    spec = BenchmarkSpec(
        algo_name="SAC",
        run_tag="20250819_2",
        result_tag="CV_20250819_result_2",
        batch_size=256,
        param_grid={
            "tau": np.array([1e-3, 5e-3, 1e-2], dtype=np.float32),
            "lr_actor": np.logspace(-5, -2, num=3),
            "lr_critic": np.logspace(-5, -2, num=3),
            "lr_alpha": np.logspace(-5, -2, num=3),
        },
        use_noop_logger=True,
    )
    run_benchmark_experiment(args, spec, create_sac)


if __name__ == "__main__":
    main()
