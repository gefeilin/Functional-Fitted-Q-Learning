import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from d3rlpy.dataset import MDPDataset
from tqdm import tqdm


logger = logging.getLogger(__name__)


class KRRlamda_gcv(nn.Module):
    def __init__(self, state, action, y_target, device):
        super().__init__()
        self.state = state.to(device)
        self.action = action.to(device)
        self.y_target = y_target.to(device)
        self.device = device

    def rbf_kernel(self, x1, x2):
        dist = torch.cdist(x1, x2)
        h = torch.median(torch.cdist(x2, x2)).clamp(min=1e-6)
        return torch.exp(-(dist**2) / (2 * h**2))

    def I_A_lamda(self, lamda):
        x = self.rbf_kernel(self.state, self.state) * self.rbf_kernel(self.action, self.action)
        x_t = x.t()
        xtx = torch.mm(x_t, x)
        identity = torch.eye(x.size(1), device=self.device) * x.size(1) * lamda
        inv = torch.linalg.pinv(xtx + identity)
        a_hat = torch.mm(x, torch.mm(inv, x_t))
        return torch.eye(x.size(1), device=self.device) - a_hat

    def forward(self, lamda):
        lamda = lamda.to(self.device)
        i_minus_a = self.I_A_lamda(lamda)
        norm = torch.norm(torch.mm(i_minus_a, self.y_target), p=2) / i_minus_a.size(1)
        trace = (torch.trace(i_minus_a) / i_minus_a.size(1)) ** 2
        return norm / trace


def KRRlamda_gcv_selector(state=None, action=None, y_target=None, max_iterations=1000, early_stop=None, lr=0.01, device="cuda"):
    gcv_model = KRRlamda_gcv(state=state, action=action, y_target=y_target, device=device).to(device)
    best_gcv = float("inf")
    best_lamda = None
    log_lamda = nn.Parameter(torch.tensor([-9.0], dtype=torch.float32, requires_grad=True).to(device))
    optimizer = optim.Adam([{"params": [log_lamda], "lr": lr}])
    gcv_scores = []

    if early_stop is not None:
        early_stop.reset()

    for _ in range(max_iterations):
        optimizer.zero_grad()
        lamda = torch.exp(log_lamda)
        gcv = gcv_model(lamda)
        gcv_scores.append(gcv.item())
        if not torch.isfinite(gcv):
            continue
        gcv.backward()
        optimizer.step()
        if gcv.item() < best_gcv:
            best_gcv = gcv.item()
            best_lamda = lamda.clone().detach()
        if early_stop is not None:
            early_stop(gcv)
            if early_stop.early_stop:
                break

    if best_lamda is None:
        best_lamda = torch.exp(log_lamda.detach())
        best_gcv = gcv_scores[-1] if gcv_scores else float("inf")
    return best_gcv, best_lamda, gcv_scores


class KRR:
    def __init__(self, kernel="rbf", state=None, action=None, device="cpu"):
        self.alpha = None
        self.kernel = kernel
        self.device = device
        self.state = state.to(device) if state is not None else None
        self.action = action.to(device) if action is not None else None

    def rbf_kernel(self, x1, x2):
        dist = torch.cdist(x1, x2)
        h = torch.median(torch.cdist(x2, x2)).clamp(min=1e-6)
        return torch.exp(-(dist**2) / (2 * h**2))

    def fit(self, target=None, lamda=0.01, weight=None):
        self.target = target.to(self.device) if target is not None else None
        if self.kernel != "rbf":
            raise ValueError("Unsupported kernel")
        k_mat = self.rbf_kernel(self.state, self.state) * self.rbf_kernel(self.action, self.action)
        n_samples = k_mat.shape[0]
        if weight is None:
            k_mat = k_mat + n_samples * lamda * torch.eye(n_samples, device=self.device)
            self.alpha = torch.linalg.solve(k_mat, self.target)
            return
        weight = weight.to(self.device)
        weight_m = weight * torch.eye(n_samples, device=self.device)
        weight_m = torch.matmul(weight_m, k_mat)
        weight_m = weight_m + lamda * torch.eye(n_samples, device=self.device)
        self.alpha = torch.linalg.solve(weight_m, torch.matmul(weight_m, self.target))

    def predict(self, new_state, new_action):
        new_state = new_state.to(self.device)
        new_action = new_action.to(self.device)
        if self.kernel != "rbf":
            raise ValueError("Unsupported kernel")
        kernel_mat = self.rbf_kernel(new_state, self.state) * self.rbf_kernel(new_action, self.action)
        return torch.matmul(kernel_mat, self.alpha)

    def get_coefficients(self):
        return self.alpha

    def get_train_state_action(self):
        return self.state, self.action

    def save(self, file_path):
        torch.save({"alpha": self.alpha, "state": self.state, "action": self.action}, file_path)
        logger.info("Model saved to %s", file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.alpha = checkpoint["alpha"]
        self.state = checkpoint["state"]
        self.action = checkpoint["action"]
        logger.info("Model loaded from %s", file_path)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False


class Fitted_q_evaluation:
    def __init__(self, dataset, discount=0.4, model=None, device="cpu", policy=None, bound=None):
        self.discount = discount
        self.dataset = dataset
        self.device = device
        self.state = torch.tensor(self.dataset["State"].tolist(), dtype=torch.float32, device=self.device)
        self.action = torch.tensor(self.dataset["Continuous"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        self.reward = torch.tensor(self.dataset["Reward"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        self.next_state = torch.tensor(self.dataset["Next_State"].tolist(), dtype=torch.float32, device=self.device)
        self.y_target = self.reward.clone().detach()
        self.time = self.dataset["Time"].values
        self.q_function = model(state=self.state, action=self.action, device=self.device)
        self.policy = policy
        self.bound = bound
        self.initial_fit()

    def initial_fit(self):
        _, gcv_lam, _ = KRRlamda_gcv_selector(
            self.state,
            self.action,
            self.y_target,
            early_stop=EarlyStopping(100, 0.0001),
            max_iterations=1000,
            device=self.device,
        )
        self.q_function.fit(self.y_target, lamda=gcv_lam)

    def update_q_function(self):
        y_target_clone = self.y_target.clone()
        reward_clone = self.reward.clone()
        n_samples = self.next_state.shape[0]
        n_action_samples = 100
        next_state_expanded = self.next_state.unsqueeze(1).repeat(1, n_action_samples, 1).reshape(-1, self.next_state.shape[1])
        batch_size = 32
        actions_list = []
        for idx in range(0, next_state_expanded.shape[0], batch_size):
            batch = next_state_expanded[idx : idx + batch_size]
            with torch.no_grad():
                actions_list.append(self.policy(batch))
        sampled_actions = torch.cat(actions_list, dim=0)
        sampled_actions_matrix = sampled_actions.view(n_samples, n_action_samples)
        for _ in tqdm(range(self.time.max()), desc="Updating Q Function", unit="Iteration"):
            for row_idx in range(y_target_clone.shape[0]):
                sampled_actions_for_row = sampled_actions_matrix[row_idx, :].unsqueeze(1)
                q_values = self.q_function.predict(
                    self.next_state[row_idx].unsqueeze(0).repeat(n_action_samples, 1),
                    sampled_actions_for_row,
                )
                expectation = q_values.mean(dim=0)
                y_target_clone[row_idx] = reward_clone[row_idx] + self.discount * expectation
            y_target_clone = torch.clamp(y_target_clone, self.bound[0], self.bound[1])
            _, gcv_lam, _ = KRRlamda_gcv_selector(
                self.state,
                self.action,
                y_target_clone,
                early_stop=EarlyStopping(100, 0.0001),
                max_iterations=1000,
                device=self.device,
            )
            self.q_function.fit(y_target_clone, lamda=gcv_lam)

    def predict(self, new_state, new_action):
        prediction = self.q_function.predict(new_state, new_action)
        return torch.clamp(prediction, self.bound[0], self.bound[1])


def dataset_for_d3rlpy(df):
    state = np.stack(df["State"].to_numpy()).astype(np.float32)
    action = np.stack(df["Continuous"].to_numpy()).astype(np.float32).reshape(-1, 1)
    reward = np.stack(df["Reward"].to_numpy()).astype(np.float32)
    terminals = np.stack(df["Status"].to_numpy()).astype(np.bool_)
    return MDPDataset(state, action, reward, terminals)
