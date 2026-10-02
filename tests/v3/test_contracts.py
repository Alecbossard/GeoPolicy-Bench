import numpy as np
import torch
from geopolicy.v2.metrics import StableWindow
from geopolicy.v3.metrics import physical_sample
from geopolicy.v3.models import DirectBC


def test_physical_dwell_distinguishes_empty_hand_closure_from_object_motion():
    cfg = dict(maximum_linear_speed_m_s=.02, maximum_angular_speed_rad_s=.25)
    assert not physical_sample(False, True, True, False, 0, 0, cfg)
    assert physical_sample(True, True, True, False, 0, 0, cfg)
    assert not physical_sample(True, True, True, True, 0, 0, cfg)
    assert not physical_sample(True, True, True, False, .021, 0, cfg)
    window = StableWindow(1.)
    for i in range(20):
        assert not window.update(i / 20, True)
    assert window.update(1., True)
    interrupted = StableWindow(1.)
    for i in range(21):
        interrupted.update(i / 20, i != 10)
    assert not interrupted.success


def test_binary_gripper_uses_physical_sign_with_nonzero_normalization():
    torch.manual_seed(1)
    net = DirectBC(binary_gripper=True)
    batch = dict(state=torch.zeros(1, 23), instruction=torch.tensor([[1., 0., 1., 0.]]),
                 points=torch.zeros(1, 8, 6), point_mask=torch.ones(1, 8, dtype=torch.bool),
                 action_mean=torch.tensor([0., 0., 0., 0., 0., 0., .3]),
                 action_std=torch.tensor([1., 1., 1., 1., 1., 1., .8]))
    with torch.no_grad():
        net.gripper.weight.zero_()
        net.gripper.bias.fill_(-2.)
        prediction = net.predict(batch)
    physical = prediction * batch["action_std"] + batch["action_mean"]
    assert torch.allclose(physical[:, :, 6], torch.full((1, 8), -1.))
    batch.update(action=torch.zeros(1, 8, 7), physical_action=torch.ones(1, 8, 7),
                 action_mask=torch.ones(1, 8, dtype=torch.bool))
    loss, _ = net.loss(batch)
    loss.backward()
    assert net.head[-1].weight.grad.reshape(8, 7, -1)[:, 6].abs().sum() > 0
