import logging

import torch
import torch.nn as nn
import torch.optim as optim


logger = logging.getLogger(__name__)


class KRRlamda_gcv(nn.Module):
    def __init__(self, state, action, y_target, device):
        super(KRRlamda_gcv, self).__init__()
        self.state = state.to(device)
        self.action = action.to(device)
        self.y_target = y_target.to(device)
        self.device = device

    def rbf_kernel(self, X1, X2):
        dist = torch.cdist(X1, X2)
        h = torch.median(torch.cdist(X2, X2))
        h = torch.clamp(h, min=1e-6)
        return torch.exp(-dist**2 / (2 * h**2))

    def I_A_lamda(self, lamda):
        # Build the hat-matrix complement that appears in the generalized CV objective.
        X = self.rbf_kernel(self.state, self.state) * self.rbf_kernel(self.action, self.action)
        X_t = X.t()
        XtX = torch.mm(X_t, X)
        # Keep the identity matrix on the same device as the kernel matrix.
        I = torch.eye(X.size(1), device=self.device) * X.size(1) * lamda
        XtX_plus_nlambdaI = XtX + I
        inv = torch.linalg.pinv(XtX_plus_nlambdaI)
        A = torch.mm(X, torch.mm(inv, X_t))
        I_A = torch.eye(X.size(1), device=self.device) - A
        return I_A

    def forward(self, lamda):
        lamda = lamda.to(self.device)
        I_A_lamda = self.I_A_lamda(lamda)
        norm = torch.norm(torch.mm(I_A_lamda, self.y_target), p=2) / I_A_lamda.size(1)
        trace = (torch.trace(I_A_lamda) / I_A_lamda.size(1)) ** 2
        return norm / trace


def KRRlamda_gcv_selector(state=None, action=None, y_target=None, max_iterations=1000, early_stop=None, lr=0.01, device="cuda"):
    # Optimize lambda in log-space so the regularization parameter stays positive.
    gcv_model = KRRlamda_gcv(state=state, action=action, y_target=y_target, device=device).to(device)
    best_gcv = float("inf")
    best_lamda = None
    log_lamda = nn.Parameter(torch.tensor([-9.0], dtype=torch.float32, requires_grad=True).to(device))
    fallback_lamda = torch.exp(log_lamda.detach()).clone()
    optimizer = optim.Adam([{"params": [log_lamda], "lr": lr}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.9,
        patience=10,
        threshold=1e-4,
        min_lr=1e-6,
    )
    gcv_scores = []

    if early_stop is not None:
        early_stop.reset()

    for iteration in range(max_iterations):
        optimizer.zero_grad()
        lamda = torch.exp(log_lamda)
        gcv = gcv_model(lamda)
        gcv_scores.append(gcv.item())

        if not torch.isfinite(gcv):
            continue

        gcv.backward()
        optimizer.step()
        scheduler.step(gcv.item())

        if gcv.item() < best_gcv:
            best_gcv = gcv.item()
            best_lamda = lamda.clone().detach()

        if early_stop is not None:
            early_stop(gcv)
            if early_stop.early_stop:
                logger.info("GCV stopped early at iteration %s", iteration)
                break
            if iteration == max_iterations - 1:
                logger.info("GCV reached maximum iterations without early stopping")

    if best_lamda is None:
        best_lamda = fallback_lamda

    return best_gcv, best_lamda, gcv_scores


class KRR:
    def __init__(self, kernel="rbf", state=None, action=None, device="cpu"):
        self.alpha = None
        self.kernel = kernel
        self.device = device
        self.state = state.to(device) if state is not None else None
        self.action = action.to(device) if action is not None else None

    def rbf_kernel(self, X1, X2):
        dist = torch.cdist(X1, X2)
        h = torch.median(torch.cdist(X2, X2))
        h = torch.clamp(h, min=1e-6)
        return torch.exp(-dist**2 / (2 * h**2))

    def fit(self, target=None, lamda=0.01, weight=None):
        self.target = target.to(self.device) if target is not None else None
        if self.kernel == "rbf":
            K = self.rbf_kernel(self.state, self.state) * self.rbf_kernel(self.action, self.action)
        else:
            raise ValueError("Unsupported kernel")

        # Solve the kernel ridge regression system directly on the chosen device.
        n = K.shape[0]
        if weight is None:
            K += n * lamda * torch.eye(n, device=self.device)
            self.alpha = torch.linalg.solve(K, self.target)
        else:
            self.weight = weight.to(self.device)
            weight_m = self.weight * torch.eye(n, device=self.device)
            weight_m = torch.matmul(weight_m, K)
            weight_m += lamda * torch.eye(n, device=self.device)
            self.alpha = torch.linalg.solve(weight_m, torch.matmul(weight_m, self.target))

    def predict(self, new_state, new_action):
        new_state = new_state.to(self.device)
        new_action = new_action.to(self.device)

        if self.kernel == "rbf":
            K = self.rbf_kernel(new_state, self.state) * self.rbf_kernel(new_action, self.action)
        else:
            raise ValueError("Unsupported kernel")
        return torch.matmul(K, self.alpha)

    def get_coefficients(self):
        return self.alpha

    def get_train_state_action(self):
        return self.state, self.action

    def save(self, file_path):
        save_dict = {
            "alpha": self.alpha,
            "state": self.state,
            "action": self.action,
        }
        torch.save(save_dict, file_path)
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


class Maximize_QHat(nn.Module):
    def __init__(self, krr_model, given_state, basis, R_matrix, q_hat_upper_bound, device):
        super(Maximize_QHat, self).__init__()
        self.krr_model = krr_model
        self.given_state = given_state.to(device)
        self.basis = basis.to(device)
        self.R_matrix = R_matrix.to(device)
        self.device = device
        self.q_hat_upper_bound = q_hat_upper_bound
        self.Sigma_hat = torch.cov(self.given_state.T)

    def forward(self, action_coefficients):
        # Map spline coefficients to an action trajectory and score it with the current Q estimate.
        action_coefficients = action_coefficients.to(self.device)
        action_temp = self.basis.T @ action_coefficients.T
        action_temp = self.given_state @ action_temp.T
        action_temp = torch.clamp(action_temp, min=-2.0, max=2.0)
        q_hat = self.krr_model.predict(self.given_state, action_temp)
        q_hat = torch.clamp(q_hat, max=self.q_hat_upper_bound)
        return q_hat.mean(dim=0, keepdim=True)

    def maximize(self, max_interation=1000, smooth_lambda=0.1, lr=0.001, decay_rate=0.9, early_stop=None):
        # Optimize over spline coefficients instead of enumerating candidate trajectories.
        action_coefficients = nn.Parameter(
            torch.randn(self.given_state.shape[1], self.basis.shape[0], device=self.device, requires_grad=True)
        )
        best_loss = float("inf")
        best_coefficients = None
        fallback_coefficients = action_coefficients.detach().clone()
        optimizer = torch.optim.Adam([action_coefficients], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.99,
            patience=10,
            threshold=1e-4,
            min_lr=1e-6,
        )

        if early_stop is not None:
            early_stop.reset()
        for iteration in range(max_interation):
            optimizer.zero_grad()
            crc_sigma = action_coefficients @ self.R_matrix @ action_coefficients.T @ self.Sigma_hat
            penalty = torch.trace(crc_sigma).reshape(1, 1)
            q_hat = self.forward(action_coefficients)
            loss = -q_hat + smooth_lambda * penalty
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_coefficients = action_coefficients.clone().detach()

            if early_stop is not None:
                early_stop(loss)
                if early_stop.early_stop:
                    logger.info("Policy optimization stopped early at iteration %s", iteration)
                    break
            if iteration == max_interation - 1:
                logger.info("Policy optimization reached maximum iterations without early stopping")
        if best_coefficients is None:
            best_coefficients = fallback_coefficients
        return best_coefficients


__all__ = [
    "EarlyStopping",
    "KRR",
    "KRRlamda_gcv",
    "KRRlamda_gcv_selector",
    "Maximize_QHat",
]
