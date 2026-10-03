"""A diagnostic or a partial/test result must never authorize a broader study."""
import copy
import unittest
from geopolicy.v3.pipeline import validation_counts


class ValidationGateTests(unittest.TestCase):
    def setUp(self):
        self.result = dict(
            identity=dict(first=110200, episodes=20, reference=False,
                          raw_weights=False, checkpoint_sha256="checkpoint"),
            rollouts=[dict(scene_seed=scene, training_seed=0,
                           diagnostic_oracle=False, overfit_diagnostic=False,
                           checkpoint_sha256="checkpoint", physical_success=True,
                           strict_v2_success=True) for scene in range(110200, 110220)])

    def test_complete_unassisted_validation_is_counted(self):
        self.assertEqual(validation_counts(self.result, 0, 110200), dict(physical=20, strict=20))

    def test_oracle_overfit_wrong_seed_and_partial_results_are_rejected(self):
        for key, value in [("diagnostic_oracle", True), ("overfit_diagnostic", True),
                           ("training_seed", 2), ("checkpoint_sha256", "other")]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.result)
                changed["rollouts"][0][key] = value
                with self.assertRaises(AssertionError):
                    validation_counts(changed, 0, 110200)
        self.result["rollouts"].pop()
        with self.assertRaises(AssertionError):
            validation_counts(self.result, 0, 110200)

    def test_training_and_reserved_test_cannot_be_gate_inputs(self):
        for first in (10000, 400000):
            changed = copy.deepcopy(self.result)
            changed["identity"]["first"] = first
            for offset, row in enumerate(changed["rollouts"]):
                row["scene_seed"] = first + offset
            with self.assertRaises(AssertionError):
                validation_counts(changed, 0, first)


if __name__ == "__main__":
    unittest.main()
