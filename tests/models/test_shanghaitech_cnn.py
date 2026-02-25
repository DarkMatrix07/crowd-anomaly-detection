import torch

from src.models.shanghaitech_cnn import build_model, tolerance_accuracy


def test_build_tiny_model_output_shape() -> None:
    model = build_model(model_name="tiny", pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1)


def test_tolerance_accuracy_bounds() -> None:
    y_true = torch.tensor([100.0, 200.0, 300.0])
    y_pred = torch.tensor([110.0, 500.0, 310.0])
    acc = tolerance_accuracy(y_true.numpy(), y_pred.numpy())
    assert 0.0 <= acc <= 1.0
