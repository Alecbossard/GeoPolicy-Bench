import numpy as np
import torch
from torch import nn
from geopolicy.v2.models import ActionLoss


def test_gripper_classifier_decodes_to_physical_sign_with_nonzero_mean():
    class Decoder(nn.Module, ActionLoss):
        pass

    decoder = Decoder()
    norm = {
        "action_mean": np.array([0] * 6 + [0.7], np.float32),
        "action_std": np.array([1] * 6 + [0.9], np.float32),
    }
    decoder.install_normalization(norm)
    normalized = decoder.normalized_gripper(torch.tensor([-0.1, 0.1]))
    physical = normalized * decoder.action_std[6] + decoder.action_mean[6]
    torch.testing.assert_close(physical, torch.tensor([-1.0, 1.0]), rtol=0, atol=1e-7)
