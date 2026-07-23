import torch
import torch.nn.functional as F
import pytest

from ml.model import VisMLP


def test_forward_returns_correct_shape_and_dtype():
    model = VisMLP(max_vis=8, npix=4, hidden_sizes=(16,))
    x = torch.randn(1, 8, 5)
    out = model(x)
    assert out.shape == (1, 16)
    assert out.dtype == torch.float32


def test_forward_handles_multiple_batch_sizes():
    model = VisMLP(max_vis=8, npix=4, hidden_sizes=(16,))
    for batch_size in (1, 4):
        x = torch.randn(batch_size, 8, 5)
        out = model(x)
        assert out.shape == (batch_size, 16)


def test_gradients_flow_after_backward():
    model = VisMLP(max_vis=8, npix=4, hidden_sizes=(16,))
    x = torch.randn(2, 8, 5)
    target = torch.randn(2, 16)

    loss = F.mse_loss(model(x), target)
    loss.backward()

    grads = [p.grad for p in model.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_empty_hidden_sizes_raises_value_error():
    with pytest.raises(ValueError, match="hidden_sizes"):
        VisMLP(max_vis=8, npix=4, hidden_sizes=())


def test_forward_is_deterministic_in_eval_mode():
    torch.manual_seed(0)
    model = VisMLP(max_vis=8, npix=4, hidden_sizes=(16,))
    model.eval()
    x = torch.randn(1, 8, 5)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.equal(out1, out2)
