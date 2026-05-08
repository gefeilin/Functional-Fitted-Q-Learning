import argparse

import numpy as np
from d3rlpy.algos import TD3Config

from .benchmark_runner import BenchmarkSpec, run_benchmark_experiment

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/benchmarks")
    return parser

def create_td3(params, gamma, batch_size, device):
    return TD3Config(
        actor_learning_rate=float(params["lr_actor"]),
        critic_learning_rate=float(params["lr_critic"]),
        batch_size=batch_size,
        tau=float(params["tau"]),
        gamma=gamma,
    ).create(device=device)

def main():
    args = build_parser().parse_args()
    spec = BenchmarkSpec(
        algo_name="TD3",
        run_tag="20250819",
        result_tag="CV_20250819_result",
        batch_size=256,
        param_grid={
            "tau": np.array([1e-3, 5e-3, 1e-2], dtype=np.float32),
            "lr_actor": np.logspace(-5, -2, num=3),
            "lr_critic": np.logspace(-5, -2, num=3),
        },
    )
    run_benchmark_experiment(args, spec, create_td3)


if __name__ == "__main__":
    main()
