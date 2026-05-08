import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from .config import ExperimentConfig
from .kernels import EarlyStopping, KRR, KRRlamda_gcv_selector, Maximize_QHat


logger = logging.getLogger(__name__)


class QHat_c(nn.Module):
    def __init__(self, krr_model, given_state, q_hat_upper_bound, device):
        super(QHat_c, self).__init__()
        self.krr_model = krr_model
        self.given_state = given_state.to(device)
        self.device = device
        self.q_hat_upper_bound = q_hat_upper_bound

    def forward(self, action):
        action = action.to(self.device)
        action = torch.clamp(action, min=-2, max=2)
        q_hat = self.krr_model.predict(self.given_state, action)
        q_hat = torch.clamp(q_hat, max=self.q_hat_upper_bound)
        return q_hat


def max_Q_c(
    model=None,
    given_state=None,
    max_iterations=1000,
    early_stop=None,
    lr=0.001,
    decay_rate=0.9,
    q_hat_upper_bound=9.56,
    discount=0.4,
    device=None,
):
    # Optimize a scalar action by gradient ascent on the fitted Q surrogate.
    q_hat_model = QHat_c(model, given_state, q_hat_upper_bound, device).to(device)
    best_q_hat = -float("inf")
    best_action = None
    action = nn.Parameter((torch.rand(1, 1, device=device) * 4.0) - 2.0)

    optimizer = optim.Adam([{"params": [action], "lr": lr}])
    scheduler = ExponentialLR(optimizer, gamma=decay_rate)
    q_hat_values = []

    if early_stop is not None:
        early_stop.reset()

    for iteration in range(max_iterations):
        optimizer.zero_grad()
        q_hat = q_hat_model(action)
        q_hat_values.append(q_hat.item())
        loss = -q_hat
        loss.backward()
        optimizer.step()
        scheduler.step()

        if q_hat.item() > best_q_hat:
            best_q_hat = q_hat.item()
            best_action = action.clone().detach()

        if early_stop is not None:
            early_stop(loss)
            if early_stop.early_stop:
                logger.info("Scalar action optimization stopped early at iteration %s", iteration)
                break

    return best_q_hat, best_action, q_hat_values


class Fitted_Q_Iteration_c:
    def __init__(
        self,
        dataset,
        discount=0.4,
        lr=0.01,
        model=KRR,
        max_iterations=2000,
        decay_rate=0.9,
        q_hat_upper_bound=9.56,
        early_stop=None,
        device=None,
    ):
        self.device = device
        self.dataset = dataset
        self.lr = lr
        self.max_iterations = max_iterations
        self.gcv_max_iterations = max(1, min(1000, self.max_iterations))
        self.discount = discount
        self.state = torch.tensor(np.stack(self.dataset["State"].to_numpy()), dtype=torch.float32, device=self.device)
        self.action = torch.tensor(self.dataset["Continuous"].to_numpy(), dtype=torch.float32, device=self.device).unsqueeze(-1)
        self.reward = torch.tensor(self.dataset["Reward"].to_numpy(), dtype=torch.float32, device=self.device).unsqueeze(-1)
        self.next_state = torch.tensor(np.stack(self.dataset["Next_State"].to_numpy()), dtype=torch.float32, device=self.device)
        self.y_target = self.reward.clone().detach()
        self.time = self.dataset["Time"].values
        self.q_function = model(state=self.state, action=self.action, device=self.device)
        self.model = model
        self.early_stop = early_stop
        self.decay_rate = decay_rate
        self.q_hat_upper_bound = q_hat_upper_bound
        self.initial_fit()

    def initial_fit(self):
        _, gcv_lam, _ = KRRlamda_gcv_selector(
            self.state,
            self.action,
            self.y_target,
            early_stop=self.early_stop,
            max_iterations=self.max_iterations,
            device=self.device,
        )
        logger.info("Initial scalar-action KRR lambda: %s", gcv_lam)
        self.q_function.fit(self.y_target, lamda=gcv_lam)

    def update_q_function(self):
        # Alternate Bellman target updates with fresh kernel refits on the whole dataset.
        y_target_clone = self.y_target.clone()
        reward_clone = self.reward.clone()
        for _ in tqdm(range(self.time.max()), desc="Updating Q Function", unit="Iteration"):
            for r in range(y_target_clone.shape[0]):
                q_max, _, _ = max_Q_c(
                    model=self.q_function,
                    given_state=self.next_state[r].unsqueeze(0),
                    lr=self.lr,
                    max_iterations=self.max_iterations,
                    decay_rate=self.decay_rate,
                    q_hat_upper_bound=self.q_hat_upper_bound,
                    early_stop=self.early_stop,
                    discount=self.discount,
                    device=self.device,
                )
                y_target_clone[r] = reward_clone[r] + self.discount * q_max

            _, gcv_lam, _ = KRRlamda_gcv_selector(
                self.state,
                self.action,
                y_target_clone,
                early_stop=self.early_stop,
                max_iterations=self.max_iterations,
                device=self.device,
            )
            self.q_function.fit(y_target_clone, lamda=gcv_lam)

    def get_policy(self, state, max_iterations=2000):
        _, new_action, _ = max_Q_c(
            model=self.q_function,
            given_state=state.unsqueeze(0),
            lr=self.lr,
            max_iterations=max_iterations,
            q_hat_upper_bound=self.q_hat_upper_bound,
            early_stop=self.early_stop,
            discount=self.discount,
            device=self.device,
        )
        return new_action


class Fitted_Q_Iteration:
    def __init__(
        self,
        dataset,
        discount=0.4,
        bspline_basis=None,
        R_matrix=None,
        lr=0.01,
        model=KRR,
        max_fqi_iteration=10,
        max_iterations=2000,
        early_stop=None,
        device=None,
        lambda_spline=0,
        q_hat_upper_bound=9.56,
    ):
        self.device = device
        self.dataset = dataset
        self.lr = lr
        self.model = model
        self.max_iterations = max_iterations
        self.gcv_max_iterations = max(1, min(1000, self.max_iterations))
        self.discount = discount
        self.state = torch.tensor(np.stack(self.dataset["State"].to_numpy()), dtype=torch.float32, device=self.device)
        self.action = torch.tensor(np.stack(self.dataset["FunctionalData"].to_numpy()), dtype=torch.float32, device=self.device)
        self.reward = torch.tensor(self.dataset["Reward"].to_numpy(), dtype=torch.float32, device=self.device).unsqueeze(-1)
        self.next_state = torch.tensor(np.stack(self.dataset["Next_State"].to_numpy()), dtype=torch.float32, device=self.device)
        self.y_target = self.reward.clone().detach()
        self.q_function = self.model(state=self.state, action=self.action, device=self.device)
        self.basis = bspline_basis.to(self.device)
        self.R_matrix = R_matrix.to(self.device)
        self.early_stop = early_stop
        self.lambda_spline = lambda_spline
        self.max_fqi_iteration = max_fqi_iteration
        self.q_hat_upper_bound = q_hat_upper_bound
        self.initial_fit()

    def initial_fit(self):
        initial_gcv_start = time.time()
        _, gcv_lam, _ = KRRlamda_gcv_selector(
            self.state,
            self.action,
            self.y_target,
            early_stop=EarlyStopping(50, 0.00001),
            max_iterations=self.gcv_max_iterations,
            device=self.device,
        )
        initial_gcv_end = time.time()

        initial_fit_start = time.time()
        self.q_function.fit(self.y_target, lamda=gcv_lam)
        initial_fit_end = time.time()

        logger.info(
            "Initial functional FQI fit completed: GCV %.2fs, KRR fit %.2fs",
            initial_gcv_end - initial_gcv_start,
            initial_fit_end - initial_fit_start,
        )

    def update_q_function(self):
        # At each FQI step, update targets using the current greedy functional policy.
        y_target_clone = self.y_target.clone()
        reward_clone = self.reward.clone()

        for iteration in tqdm(range(self.max_fqi_iteration), desc="Updating Q Function", unit="Iteration"):
            iter_start = time.time()
            target_start = time.time()
            if iteration == 0:
                mean_action = self.action.mean(dim=0)
                optimal_action = mean_action.clone().unsqueeze(0).repeat(self.next_state.shape[0], 1)
                q_hat = self.q_function.predict(self.next_state, optimal_action)
                q_hat = torch.clamp(q_hat, max=self.q_hat_upper_bound)
                y_target_clone = reward_clone + self.discount * q_hat
            else:
                optimal_action = self.basis.T @ optimal_action_coefficients.T
                optimal_action = self.next_state @ optimal_action.T
                optimal_action = torch.clamp(optimal_action, min=-2.0, max=2.0)
                q_hat = self.q_function.predict(self.next_state, optimal_action)
                q_hat = torch.clamp(q_hat, max=self.q_hat_upper_bound)
                y_target_clone = reward_clone + self.discount * q_hat
            target_end = time.time()

            self.q_function = self.model(state=self.state, action=self.action, device=self.device)
            gcv_start = time.time()
            _, gcv_lam, _ = KRRlamda_gcv_selector(
                self.state,
                self.action,
                y_target_clone,
                early_stop=EarlyStopping(50, 0.00001),
                max_iterations=self.gcv_max_iterations,
                device=self.device,
            )
            gcv_end = time.time()

            fit_start = time.time()
            self.q_function.fit(y_target_clone, lamda=gcv_lam)
            fit_end = time.time()

            policy_start = time.time()
            optimal_action_coefficients = Maximize_QHat(
                self.q_function,
                self.next_state,
                self.basis,
                self.R_matrix,
                self.q_hat_upper_bound,
                device=self.device,
            ).maximize(
                max_interation=self.max_iterations,
                smooth_lambda=self.lambda_spline,
                lr=self.lr,
                decay_rate=0.9,
                early_stop=self.early_stop,
            )
            if iteration == self.max_fqi_iteration - 1:
                self.optimal_action_coefficients = optimal_action_coefficients
            policy_end = time.time()
            iter_end = time.time()

            logger.info(
                "FQI iteration %s completed: target %.2fs, GCV %.2fs, KRR fit %.2fs, policy %.2fs, total %.2fs",
                iteration + 1,
                target_end - target_start,
                gcv_end - gcv_start,
                fit_end - fit_start,
                policy_end - policy_start,
                iter_end - iter_start,
            )

    def get_policy(self, state):
        optimal_action = self.basis.T @ self.optimal_action_coefficients.T
        optimal_action = state @ optimal_action.T
        optimal_action = torch.clamp(optimal_action, min=-2.0, max=2.0)
        return optimal_action

    def get_device(self):
        return self.device

    def save(self, file_path):
        checkpoint = {
            "checkpoint_type": "functional_fqi_full",
            "q_function": {
                "alpha": self.q_function.alpha,
                "state": self.q_function.state,
                "action": self.q_function.action,
            },
            "optimal_action_coefficients": getattr(self, "optimal_action_coefficients", None),
            "basis": self.basis,
            "R_matrix": self.R_matrix,
            "discount": self.discount,
            "lambda_spline": self.lambda_spline,
            "q_hat_upper_bound": self.q_hat_upper_bound,
            "experiment_config": (
                self.experiment_config.to_dict() if hasattr(self, "experiment_config") and self.experiment_config is not None else None
            ),
        }
        torch.save(checkpoint, file_path)
        logger.info("Model saved to %s", file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)

        # Compatibility: minimal checkpoints stored only the KRR state.
        if "q_function" not in checkpoint:
            self.q_function.load(file_path)
            return

        q_function_state = checkpoint["q_function"]
        self.q_function.alpha = q_function_state["alpha"].to(self.device)
        self.q_function.state = q_function_state["state"].to(self.device)
        self.q_function.action = q_function_state["action"].to(self.device)

        optimal_action_coefficients = checkpoint.get("optimal_action_coefficients")
        if optimal_action_coefficients is not None:
            self.optimal_action_coefficients = optimal_action_coefficients.to(self.device)

        if "basis" in checkpoint and checkpoint["basis"] is not None:
            self.basis = checkpoint["basis"].to(self.device)
        if "R_matrix" in checkpoint and checkpoint["R_matrix"] is not None:
            self.R_matrix = checkpoint["R_matrix"].to(self.device)

        logger.info("Model loaded from %s", file_path)

    @classmethod
    def from_checkpoint(
        cls,
        file_path,
        *,
        device="cpu",
        model=KRR,
        early_stop=None,
        dataset=None,
        bspline_basis=None,
        R_matrix=None,
        lr=None,
        max_fqi_iteration=None,
        max_iterations=None,
    ):
        checkpoint = torch.load(file_path, map_location=device)

        if "q_function" not in checkpoint:
            raise ValueError(
                "This checkpoint only stores the minimal KRR state. "
                "Instantiate Fitted_Q_Iteration first and then call load() for backward compatibility."
            )

        config_data = checkpoint.get("experiment_config")
        experiment_config = ExperimentConfig.from_dict(config_data) if config_data is not None else None

        if bspline_basis is None or R_matrix is None:
            if checkpoint.get("basis") is not None and checkpoint.get("R_matrix") is not None:
                bspline_basis = checkpoint["basis"]
                R_matrix = checkpoint["R_matrix"]
            elif experiment_config is not None:
                from .runner import build_bspline_basis

                bspline_basis, R_matrix = build_bspline_basis(experiment_config, torch.device(device))
            else:
                raise ValueError("bspline_basis and R_matrix are required when the checkpoint does not contain enough metadata.")

        fitted_q = cls.__new__(cls)
        fitted_q.device = torch.device(device)
        fitted_q.dataset = dataset
        fitted_q.lr = lr if lr is not None else (experiment_config.learning_rate if experiment_config is not None else 0.01)
        fitted_q.model = model
        fitted_q.max_iterations = (
            max_iterations if max_iterations is not None else (experiment_config.training_iterations if experiment_config is not None else 2000)
        )
        fitted_q.discount = checkpoint.get("discount", experiment_config.discount if experiment_config is not None else 0.4)
        fitted_q.lambda_spline = checkpoint.get("lambda_spline", 0)
        fitted_q.max_fqi_iteration = (
            max_fqi_iteration if max_fqi_iteration is not None else (experiment_config.fqi_iterations if experiment_config is not None else 10)
        )
        fitted_q.q_hat_upper_bound = checkpoint.get(
            "q_hat_upper_bound",
            experiment_config.q_hat_upper if experiment_config is not None else 9.56,
        )
        fitted_q.early_stop = early_stop
        fitted_q.experiment_config = experiment_config
        fitted_q.basis = bspline_basis.to(fitted_q.device)
        fitted_q.R_matrix = R_matrix.to(fitted_q.device)

        q_function_state = checkpoint["q_function"]
        fitted_q.q_function = model(
            state=q_function_state["state"].to(fitted_q.device),
            action=q_function_state["action"].to(fitted_q.device),
            device=fitted_q.device,
        )
        fitted_q.q_function.alpha = q_function_state["alpha"].to(fitted_q.device)

        optimal_action_coefficients = checkpoint.get("optimal_action_coefficients")
        fitted_q.optimal_action_coefficients = (
            optimal_action_coefficients.to(fitted_q.device) if optimal_action_coefficients is not None else None
        )

        return fitted_q


class Fitted_q_evaluation:
    def __init__(
        self,
        dataset,
        discount=0.4,
        model=None,
        device="cpu",
        policy=None,
        bound=None,
        policy_actions=None,
        num_iterations=20,
        gcv_max_iterations=1000,
    ):
        self.discount = discount
        self.dataset = dataset
        self.model = model
        self.device = device
        self.state = torch.tensor(np.stack(self.dataset["State"].to_numpy()), dtype=torch.float32, device=self.device)
        self.action = torch.tensor(np.stack(self.dataset["FunctionalData"].to_numpy()), dtype=torch.float32, device=self.device)
        self.reward = torch.tensor(self.dataset["Reward"].to_numpy(), dtype=torch.float32, device=self.device).unsqueeze(-1)
        self.next_state = torch.tensor(np.stack(self.dataset["Next_State"].to_numpy()), dtype=torch.float32, device=self.device)
        self.y_target = self.reward.clone().detach()
        self.time = self.dataset["Time"].values
        self.q_function = model(state=self.state, action=self.action, device=self.device)
        self.policy = policy
        self.policy_actions = policy_actions.to(self.device) if policy_actions is not None else None
        self.num_iterations = num_iterations
        self.gcv_max_iterations = max(1, int(gcv_max_iterations))
        self.model = model
        self.initial_fit()
        self.bound = bound

    def _apply_bounds(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.bound is None:
            return tensor
        return torch.clamp(tensor, self.bound[0], self.bound[1])

    def _predict_policy_expectation(self, state, policy_actions):
        if policy_actions.dim() == 2:
            return self.q_function.predict(state, policy_actions).to(self.device)

        if policy_actions.dim() != 3:
            raise ValueError("policy_actions must have shape [num_states, action_dim] or [num_states, num_samples, action_dim].")

        num_states, num_samples, action_dim = policy_actions.shape
        output_dim = self.q_function.alpha.shape[-1]
        prediction_sum = torch.zeros((num_states, output_dim), dtype=self.q_function.alpha.dtype, device=self.device)

        # Avoid materializing a (num_states * num_samples) x num_train cdist matrix all at once.
        state_kernel = self.q_function.rbf_kernel(state, self.q_function.state)
        max_chunk_size = max(1, 16_000_000 // max(1, num_states * self.q_function.state.shape[0]))

        for sample_start in range(0, num_samples, max_chunk_size):
            sample_end = min(sample_start + max_chunk_size, num_samples)
            action_chunk = policy_actions[:, sample_start:sample_end, :]
            flat_actions = action_chunk.reshape(num_states * (sample_end - sample_start), action_dim)
            action_kernel = self.q_function.rbf_kernel(flat_actions, self.q_function.action).reshape(
                num_states,
                sample_end - sample_start,
                -1,
            )
            combined_kernel = state_kernel.unsqueeze(1) * action_kernel
            chunk_predictions = torch.matmul(combined_kernel, self.q_function.alpha)
            prediction_sum += chunk_predictions.sum(dim=1)

        return prediction_sum / num_samples

    def initial_fit(self):
        initial_gcv_start = time.time()
        _, gcv_lam, _ = KRRlamda_gcv_selector(
            self.state,
            self.action,
            self.y_target,
            early_stop=EarlyStopping(50, 0.00001),
            max_iterations=self.gcv_max_iterations,
            device=self.device,
        )
        initial_gcv_end = time.time()

        initial_fit_start = time.time()
        self.q_function.fit(self.y_target, lamda=gcv_lam)
        initial_fit_end = time.time()

        logger.info(
            "Initial functional FQE fit completed: GCV %.2fs, KRR fit %.2fs",
            initial_gcv_end - initial_gcv_start,
            initial_fit_end - initial_fit_start,
        )

    def update_q_function(self):
        # FQE keeps the policy fixed and iterates only the value update.
        y_target_clone = self.y_target.clone()
        reward_clone = self.reward.clone()

        if self.policy_actions is None:
            logger.info("Generating policy action list")
            optimal_action_start = time.time()
            self.policy_actions = self.policy(self.next_state).to(self.device)
            optimal_action_end = time.time()
            logger.info("Generated policy action list in %.2fs", optimal_action_end - optimal_action_start)
        else:
            logger.info("Reusing cached policy action list")

        for iteration in tqdm(range(self.num_iterations), desc="Updating Q Function", unit="Iteration"):
            pred_q_start = time.time()
            pred_q = self._predict_policy_expectation(self.next_state, self.policy_actions)
            pred_q_end = time.time()

            y_target_clone = reward_clone + self.discount * pred_q
            y_target_clone = self._apply_bounds(y_target_clone)

            gcv_start = time.time()
            _, gcv_lam, _ = KRRlamda_gcv_selector(
                self.state,
                self.action,
                y_target_clone,
                early_stop=EarlyStopping(50, 0.00001),
                max_iterations=self.gcv_max_iterations,
                device=self.device,
            )
            gcv_end = time.time()
            fit_start = time.time()
            self.q_function.fit(y_target_clone, lamda=gcv_lam)
            fit_end = time.time()

            logger.info(
                "FQE iteration %s completed: predict %.2fs, GCV %.2fs, KRR fit %.2fs",
                iteration + 1,
                pred_q_end - pred_q_start,
                gcv_end - gcv_start,
                fit_end - fit_start,
            )

    def predict(self, new_state, new_action):
        prediction = self._predict_policy_expectation(new_state, new_action)
        prediction = self._apply_bounds(prediction)
        return prediction


__all__ = [
    "Fitted_Q_Iteration",
    "Fitted_Q_Iteration_c",
    "Fitted_q_evaluation",
    "QHat_c",
    "max_Q_c",
]
