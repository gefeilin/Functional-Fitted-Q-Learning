import argparse

import numpy as np
from d3rlpy.algos import BCQConfig

from .benchmark_runner import BenchmarkSpec, run_benchmark_experiment

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/benchmarks")
    return parser

def create_bcq(params, gamma, batch_size, device):
    return BCQConfig(
        actor_learning_rate=float(params["lr_actor_critic"]),
        critic_learning_rate=float(params["lr_actor_critic"]),
        batch_size=batch_size,
        beta=float(params["beta"]),
        action_flexibility=float(params["action_flexibility"]),
        lam=float(params["lam"]),
        tau=float(params["tau"]),
        gamma=gamma,
    ).create(device=device)

def main():
    args = build_parser().parse_args()
    spec = BenchmarkSpec(
        algo_name="BCQ",
        run_tag="20250819_2",
        result_tag="CV_20250819_2_result",
        batch_size=32,
        param_grid={
            "tau": np.array([1e-3, 5e-3, 1e-2], dtype=np.float32),
            "lr_actor_critic": np.logspace(-5, -2, num=3),
            "beta": np.array([0.1, 0.5, 1.0], dtype=np.float32),
            "action_flexibility": np.array([0.05, 0.1, 0.2], dtype=np.float32),
            "lam": np.array([0.5, 0.75], dtype=np.float32),
        },
        use_noop_logger=True,
    )
    run_benchmark_experiment(args, spec, create_bcq)


if __name__ == "__main__":
    main()
