"""Subject-grouped cross-validation for the fixed-feature Nyström KRR critic.

The Bellman target is frozen within each FQI iteration.  Entire subjects are
held out together, predictions are un-clipped, and the selected ridge is then
refit on the complete design.  The implementation works from sufficient
statistics so that an 81-point grid does not require 81 explicit refits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import torch

from .nystrom_functional_critic import NystromFunctionalKRR, NystromKRRConfig


@dataclass(frozen=True)
class GroupedRidgeCVConfig:
    lambda_grid: tuple[float, ...]
    folds: int = 5
    fold_seed: int = 200000
    criterion: str = "mean_subject_unclipped_squared_prediction_error"
    tie_rule: str = "largest_lambda_among_exact_minimizers"

    def __post_init__(self) -> None:
        grid = np.asarray(self.lambda_grid, dtype=np.float64)
        if (
            grid.ndim != 1
            or len(grid) < 2
            or not np.isfinite(grid).all()
            or np.any(grid <= 0)
            or not np.all(np.diff(grid) > 0)
        ):
            raise ValueError(
                "grouped CV needs a finite positive increasing lambda grid"
            )
        if type(self.folds) is not int or self.folds < 2:
            raise ValueError("grouped CV needs at least two folds")
        if type(self.fold_seed) is not int or self.fold_seed < 0:
            raise ValueError("grouped CV fold_seed must be a nonnegative integer")
        if self.criterion != "mean_subject_unclipped_squared_prediction_error":
            raise ValueError("unsupported grouped CV criterion")
        if self.tie_rule != "largest_lambda_among_exact_minimizers":
            raise ValueError("unsupported grouped CV tie rule")


def grouped_cv_config_from_kernel(kernel: dict) -> GroupedRidgeCVConfig:
    if (
        kernel.get("ridge_selection") != "grouped_5fold_cv"
        or kernel.get("ridge") is not None
    ):
        raise ValueError(
            "strict grouped CV requires ridge_selection=grouped_5fold_cv and ridge=null"
        )
    specification = kernel["grouped_cv"]
    low = float(specification["log10_min"])
    high = float(specification["log10_max"])
    count = specification["grid_points"]
    folds = specification["folds"]
    fold_seed = specification["fold_seed"]
    if (
        type(count) is not int
        or count < 2
        or not np.isfinite([low, high]).all()
        or low >= high
    ):
        raise ValueError("invalid log-spaced grouped CV grid")
    return GroupedRidgeCVConfig(
        tuple(float(value) for value in np.logspace(low, high, count)),
        folds=folds,
        fold_seed=fold_seed,
    )


def _symmetric(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.T) / 2.0


def _stable_eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(_symmetric(matrix))
    scale = torch.clamp(eigenvalues.abs().max(), min=torch.finfo(matrix.dtype).tiny)
    if torch.any(eigenvalues < -1.0e-10 * scale):
        raise RuntimeError(
            "grouped CV Gram matrix is not numerically positive semidefinite"
        )
    return eigenvalues.clamp_min(0.0), eigenvectors


class GroupedRidgeCVWorkspace:
    """Reusable five-fold subject split and ridge sufficient statistics."""

    def __init__(
        self,
        features: torch.Tensor,
        subject_ids: np.ndarray,
        config: GroupedRidgeCVConfig,
    ) -> None:
        """Precompute a subject-disjoint fold design reused at every ridge solve."""
        if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
            raise ValueError("grouped CV needs a nonempty two-dimensional design")
        if features.dtype != torch.float64 or not torch.isfinite(features).all():
            raise ValueError("grouped CV design must be finite float64")
        groups = np.asarray(subject_ids)
        if groups.ndim != 1 or len(groups) != len(features):
            raise ValueError("one subject identifier is required per training row")
        if groups.dtype.kind not in "iu" or not np.isfinite(groups).all():
            raise ValueError("subject identifiers must be finite integers")
        unique, counts = np.unique(groups, return_counts=True)
        if len(unique) < config.folds:
            raise ValueError("fewer subjects than grouped CV folds")
        # The frozen designs use balanced longitudinal panels. This equality makes
        # row MSE exactly equal to the requested subject-macro MSE.
        if not np.all(counts == counts[0]):
            raise ValueError("grouped CV requires equal rows per subject")

        rng = np.random.default_rng(config.fold_seed)
        permuted = rng.permutation(unique)
        fold_groups = tuple(
            np.sort(part) for part in np.array_split(permuted, config.folds)
        )
        if max(map(len, fold_groups)) - min(map(len, fold_groups)) > 1:
            raise RuntimeError("grouped CV subject folds are unexpectedly imbalanced")
        row_fold = np.full(len(groups), -1, dtype=np.int16)
        group_to_fold: dict[int, int] = {}
        for fold_index, identifiers in enumerate(fold_groups):
            row_fold[np.isin(groups, identifiers)] = fold_index
            for identifier in identifiers:
                group_to_fold[int(identifier)] = fold_index
        if np.any(row_fold < 0) or len(group_to_fold) != len(unique):
            raise RuntimeError("grouped CV fold assignment is incomplete")

        self.n = int(len(features))
        self.features = features
        self.config = config
        self.device = features.device
        self.grid = torch.as_tensor(
            config.lambda_grid, dtype=torch.float64, device=self.device
        )
        self.groups = groups.astype(np.int64, copy=True)
        self.row_fold = row_fold
        self.fold_groups = fold_groups
        self.rows_per_subject = int(counts[0])
        assignment = np.column_stack(
            [
                np.sort(unique).astype("<i8"),
                np.array([group_to_fold[int(v)] for v in np.sort(unique)], dtype="<i8"),
            ]
        )
        self.subject_fold_assignment = assignment.tolist()

        with torch.no_grad():
            total_gram = _symmetric(features.T @ features)
            self.full_eigenvalues, self.full_eigenvectors = _stable_eigh(total_gram)
            self.fold_statistics = []
            for fold_index in range(config.folds):
                valid_rows_np = np.flatnonzero(row_fold == fold_index).astype(np.int64)
                valid_rows = torch.as_tensor(valid_rows_np, device=self.device)
                valid_features = features[valid_rows]
                valid_gram = _symmetric(valid_features.T @ valid_features)
                train_gram = _symmetric(total_gram - valid_gram)
                train_eigenvalues, train_eigenvectors = _stable_eigh(train_gram)
                self.fold_statistics.append(
                    {
                        "fold": fold_index,
                        "valid_rows": valid_rows,
                        "n_valid": int(len(valid_rows_np)),
                        "n_train": self.n - int(len(valid_rows_np)),
                        "n_valid_subjects": int(len(fold_groups[fold_index])),
                        "train_eigenvalues": train_eigenvalues,
                        "train_eigenvectors": train_eigenvectors,
                        "valid_gram_rotated": _symmetric(
                            train_eigenvectors.T @ valid_gram @ train_eigenvectors
                        ),
                    }
                )

    def fold_receipt(self) -> dict:
        return {
            "folds": self.config.folds,
            "fold_seed": self.config.fold_seed,
            "n_subjects": int(
                len(self.fold_groups) and sum(map(len, self.fold_groups))
            ),
            "rows_per_subject": self.rows_per_subject,
            "subject_counts_by_fold": [int(len(values)) for values in self.fold_groups],
            "row_counts_by_fold": [
                int(item["n_valid"]) for item in self.fold_statistics
            ],
            "subject_fold_assignment": self.subject_fold_assignment,
            "all_rows_of_each_subject_in_one_fold": True,
            "folds_reused_across_fqi_iterations": True,
        }

    def solve(self, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Select ridge by grouped validation and refit on the full design."""
        if target.shape != (self.n,) or target.dtype != torch.float64:
            raise ValueError("invalid grouped CV target")
        if not torch.isfinite(target).all():
            raise ValueError("grouped CV target must be finite")
        with torch.no_grad():
            total_right = self.features.T @ target
            fold_mse = []
            fold_rows = []
            for item in self.fold_statistics:
                valid_rows = item["valid_rows"]
                valid_target = target[valid_rows]
                valid_features = self.features[valid_rows]
                valid_right = valid_features.T @ valid_target
                train_right = total_right - valid_right
                vectors = item["train_eigenvectors"]
                eigenvalues = item["train_eigenvalues"]
                projected_train_right = vectors.T @ train_right
                coefficients_rotated = projected_train_right[None, :] / (
                    eigenvalues[None, :] + item["n_train"] * self.grid[:, None]
                )
                projected_valid_right = vectors.T @ valid_right
                linear = torch.sum(
                    coefficients_rotated * projected_valid_right[None, :], dim=1
                )
                quadratic = torch.einsum(
                    "gi,ij,gj->g",
                    coefficients_rotated,
                    item["valid_gram_rotated"],
                    coefficients_rotated,
                )
                sse = valid_target.square().sum() - 2.0 * linear + quadratic
                tolerance = 1.0e-10 * torch.clamp(valid_target.square().sum(), min=1.0)
                if torch.any(sse < -tolerance) or not torch.isfinite(sse).all():
                    raise RuntimeError("invalid grouped CV validation residual")
                mse = sse.clamp_min(0.0) / item["n_valid"]
                fold_mse.append(mse)
                fold_rows.append(
                    {
                        "fold": int(item["fold"]),
                        "n_train": int(item["n_train"]),
                        "n_valid": int(item["n_valid"]),
                        "n_valid_subjects": int(item["n_valid_subjects"]),
                    }
                )
            fold_mse_tensor = torch.stack(fold_mse, dim=1)
            # Weight by held-out subject counts. Balanced rows within subject make
            # this exactly a subject-macro average without overweighting time rows.
            weights = torch.as_tensor(
                [row["n_valid_subjects"] for row in fold_rows],
                dtype=torch.float64,
                device=self.device,
            )
            scores = (fold_mse_tensor * weights[None, :]).sum(dim=1) / weights.sum()
            if not torch.isfinite(scores).all():
                raise RuntimeError(
                    "nonfinite grouped CV score; no fixed-ridge fallback"
                )
            score_array = scores.cpu().numpy()
            # Reverse argmin implements the larger-ridge tie rule. This differs
            # from policy-return tuning, which prefers the smaller coefficient.
            selected = len(score_array) - 1 - int(np.argmin(score_array[::-1]))
            selected_lambda = float(self.config.lambda_grid[selected])
            projected_full_right = self.full_eigenvectors.T @ total_right
            coefficient = self.full_eigenvectors @ (
                projected_full_right
                / (self.full_eigenvalues + self.n * selected_lambda)
            )

        fold_mse_array = fold_mse_tensor.cpu().numpy()
        candidates = [
            {
                "lambda": float(value),
                "grouped_cv_mse": float(score_array[index]),
                "fold_mse": [float(v) for v in fold_mse_array[index]],
                "selected": index == selected,
            }
            for index, value in enumerate(self.config.lambda_grid)
        ]
        receipt = {
            "schema_version": 1,
            "criterion": self.config.criterion,
            "penalty_convention": "mean_squared_error_plus_lambda_times_coefficient_norm_squared",
            "normal_equation_diagonal": self.n * selected_lambda,
            "n": self.n,
            "selected_index": selected,
            "selected_lambda": selected_lambda,
            "selected_grouped_cv_mse": float(score_array[selected]),
            "selected_fold_mse": [float(v) for v in fold_mse_array[selected]],
            "selected_at_grid_boundary": selected in (0, len(candidates) - 1),
            "tie_rule": self.config.tie_rule,
            "candidate_count": len(candidates),
            "target_frozen_across_candidates": True,
            "prediction_clipping_in_cv": False,
            "return_used_for_selection": False,
            "fold_rows": fold_rows,
            "fold_assignment": self.fold_receipt(),
            "training_row_subsampling": False,
            "candidates": candidates,
        }
        return coefficient.detach(), receipt


class GroupedCVNystromFunctionalKRR(NystromFunctionalKRR):
    solver_identity = "deterministic_landmark_nystrom_subject_grouped_5fold_cv"

    def __init__(
        self,
        config: NystromKRRConfig,
        grouped_cv_config: GroupedRidgeCVConfig,
        subject_ids: np.ndarray,
        value_max: float = 20.0,
    ) -> None:
        if config.ridge is not None:
            raise ValueError("grouped CV critic requires ridge=None")
        super().__init__(config, value_max=value_max)
        self.grouped_cv_config = grouped_cv_config
        self.subject_ids = np.asarray(subject_ids).copy()
        self._grouped_cv_workspace: GroupedRidgeCVWorkspace | None = None

    def _prepare_target_solver(self) -> dict:
        started = time.time()
        self._grouped_cv_workspace = GroupedRidgeCVWorkspace(
            self._design_features, self.subject_ids, self.grouped_cv_config
        )
        return {
            "regularization_diagonal": None,
            "factor_reuse": False,
            "grouped_cv_sufficient_statistics_reuse": True,
            "grouped_cv_setup_wall_time_sec": time.time() - started,
            "ridge_selection": {
                "method": "subject_grouped_5fold_cv_each_fqi_iteration",
                **asdict(self.grouped_cv_config),
                "fold_assignment": self._grouped_cv_workspace.fold_receipt(),
            },
        }

    def solve_targets(self, targets: np.ndarray) -> "GroupedCVNystromFunctionalKRR":
        if self._grouped_cv_workspace is None or self.fit_metadata is None:
            raise RuntimeError("grouped CV Nyström design is not fitted")
        target = torch.as_tensor(
            np.array(targets, dtype=np.float64, copy=True),
            dtype=torch.float64,
            device=self.device,
        )
        started = time.time()
        self.alpha, receipt = self._grouped_cv_workspace.solve(target)
        right = self._design_features.T @ target
        ridge = len(target) * receipt["selected_lambda"]
        residual = self._design_features.T @ (
            self._design_features @ self.alpha - target
        )
        residual += ridge * self.alpha
        relative = float(
            (
                torch.linalg.vector_norm(residual)
                / torch.clamp(
                    torch.linalg.vector_norm(right), min=torch.finfo(torch.float64).eps
                )
            ).cpu()
        )
        if not np.isfinite(relative) or relative > 1.0e-5:
            raise RuntimeError(
                f"invalid grouped CV selected-fit normal-equation residual: {relative}"
            )
        self.fit_metadata.update(
            {
                "target_solve_count": int(self.fit_metadata["target_solve_count"]) + 1,
                "latest_target_solve_wall_time_sec": time.time() - started,
                "latest_relative_linear_solve_residual": relative,
                "relative_linear_solve_residual": relative,
                "regularization_diagonal": ridge,
                "grouped_cv_selection": receipt,
            }
        )
        return self
