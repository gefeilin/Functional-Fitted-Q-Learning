import torch

from functional_fitted_q.kernels import KRR


def test_krr_fit_predict_is_finite():
    state = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    action = torch.tensor([[0.0, 0.1], [0.2, 0.3], [0.1, 0.2]], dtype=torch.float32)
    target = torch.tensor([[1.0], [0.5], [0.25]], dtype=torch.float32)

    model = KRR(state=state, action=action, device="cpu")
    model.fit(target=target, lamda=torch.tensor([0.01]))
    prediction = model.predict(state, action)

    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
