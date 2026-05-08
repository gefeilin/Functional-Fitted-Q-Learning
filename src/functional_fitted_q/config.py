from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    # Centralize paper-facing hyperparameters so each run can be reconstructed from one object.
    horizon: int
    gamma: float
    size: int
    iteration: int
    run_tag: str = "20250719"
    num_basis_knots: int = 20
    spline_degree: int = 5
    num_action_grid_points: int = 100
    lambda_grid_size: int = 10
    lambda_min: float = 1e-5
    lambda_max: float = 1e-1
    num_data_subjects: int = 100
    num_cv_splits: int = 3
    fqi_iterations: int = 10
    training_iterations: int = 2000
    fqe_simulations: int = 1000
    fqe_cycles: int = 20
    fqe_eval_states: int = 100
    fqe_train_expectation_samples: int = 1000
    fqe_eval_expectation_samples: int = 1000
    fqe_clamp_targets: bool = True
    learning_rate: float = 0.01
    early_stopping_patience: int = 50
    early_stopping_delta: float = 1e-4

    @property
    def discount(self) -> float:
        return self.gamma

    @property
    def q_hat_upper(self) -> float:
        return 1 / (1 - self.gamma)

    @property
    def gpu_id(self) -> int:
        return self.iteration % 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        return cls(**data)
