import torch
from geopolicy.policies import PointCondition


def test_invalid_point_values_and_missing_view_do_not_change_condition():
    torch.set_num_threads(2)
    torch.manual_seed(4)
    model = PointCondition(color_prior="chroma40").eval()
    points = torch.rand(2, 512, 6)
    mask = torch.zeros(2, 512, dtype=torch.bool)
    mask[0, :32] = True
    batch = {
        "points": points,
        "point_mask": mask,
        "state": torch.zeros(2, 23),
        "instruction": torch.tensor([[1.0, 0, 1, 0], [0.0, 1, 0, 1]]),
    }
    with torch.inference_mode():
        expected = model(batch)
    altered = points.clone()
    altered[~mask] = torch.tensor([1000.0, 1000.0, 1000.0, 0.9, 0.9, 0.9])
    with torch.inference_mode():
        actual = model(dict(batch, points=altered))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
