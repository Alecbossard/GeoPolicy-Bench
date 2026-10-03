import pytest
import torch
from geopolicy.policies import PointCondition
from geopolicy.v3.models import RoutedPointCondition


@pytest.mark.parametrize("prior", ["chroma40", False])
def test_routed_encoder_preserves_existing_computation(prior):
    torch.manual_seed(7)
    original = PointCondition(color_prior=prior)
    routed = RoutedPointCondition(color_prior=prior)
    routed.load_state_dict(original.state_dict())
    batch = dict(
        points=torch.rand(2, 20, 6),
        point_mask=torch.rand(2, 20) > 0.3,
        state=torch.randn(2, 23),
        instruction=torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
    )
    with torch.no_grad():
        expected = original(batch)
        actual, moments = routed(batch)
    assert torch.equal(actual, expected)
    assert moments.shape == (2, 4, 6)
    batch["point_mask"].zero_()
    with torch.no_grad():
        _, moments = routed(batch)
    assert torch.equal(moments, torch.zeros_like(moments))
