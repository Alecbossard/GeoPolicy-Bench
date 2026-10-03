import unittest
from geopolicy.v3.resources import check_readiness


class ResourcePreflightTests(unittest.TestCase):
    def test_low_commit_is_rejected_even_with_cool_idle_gpu(self):
        snapshot = dict(free_commit_bytes=4 * 1024**3, free_disk_bytes=23 * 1024**3,
                        gpu_free_mib=6800, gpu_temperature_c=43)
        runtime = dict(minimum_free_commit_before_heavy_job_gib=6, minimum_free_gpu_mib=1024)
        in_job = dict(minimum_free_disk_gib=10, maximum_temperature_c=80)
        self.assertEqual(check_readiness(snapshot, runtime, in_job),
                         ["insufficient Windows free commit before startup"])
        snapshot["free_commit_bytes"] = 7 * 1024**3
        self.assertEqual(check_readiness(snapshot, runtime, in_job), [])


if __name__ == "__main__":
    unittest.main()
