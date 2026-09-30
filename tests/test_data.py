from geopolicy.data import scene_split
import pytest


def test_scene_splits_have_disjoint_boundaries():
    assert scene_split(99999) == "train"
    assert scene_split(100000) == "validation"
    assert scene_split(199999) == "validation"
    assert scene_split(200000) == "test"


@pytest.mark.simulation
def test_counterfactual_instruction_does_not_resample_scene():
    # Rendering-independent simulation test; target IDs change while geometry repeats.
    import numpy as np
    from geopolicy.environment import SelectPlace

    env = SelectPlace()
    env.reset_scene(121, object_id=0, goal_id=0)
    a = env.object_positions().copy()
    g = env.goal_positions.copy()
    text = env.instruction
    env.reset_scene(121, object_id=1, goal_id=1)
    np.testing.assert_allclose(a, env.object_positions(), atol=1e-12)
    np.testing.assert_allclose(g, env.goal_positions, atol=1e-12)
    assert text != env.instruction
    env.close()
