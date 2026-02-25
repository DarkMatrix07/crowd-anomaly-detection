import torch

from src.models.baseline import Conv3DBaseline


def test_conv3d_baseline_output_shape() -> None:
    model = Conv3DBaseline(in_channels=3)
    x = torch.randn(2, 3, 8, 32, 32)

    logits = model(x)

    assert logits.shape == (2, 1)
