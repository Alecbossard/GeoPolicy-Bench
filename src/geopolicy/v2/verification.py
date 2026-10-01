"""Scientific contract checks and continuation of a real main-run checkpoint."""

import json
import subprocess
import sys
from pathlib import Path
from geopolicy.io import save_json
from .config import file_hash
from .preservation import preserve_v1


def verify(recipe):
    report = {"v1": preserve_v1()}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "not gpu"], capture_output=True, text=True
    )
    report["pytest"] = dict(returncode=result.returncode, output=result.stdout + result.stderr)
    assert result.returncode == 0, report["pytest"]["output"]
    import h5py
    import numpy as np
    import torch

    original = json.loads(Path("configs/dataset_manifest.json").read_text())
    assert all(file_hash(e["path"]) == e["sha256"] for e in original["episodes"])
    report["original_hdf5_hashes_verified"] = len(original["episodes"])
    augmented = json.loads(Path(recipe["dataset_manifest"]).read_text())
    old = {r["episode_id"]: r for r in original["episodes"]}
    for record in augmented["episodes"]:
        assert file_hash(record["path"]) == record["sha256"]
        source = old[record["episode_id"]]
        with h5py.File(source["path"]) as a, h5py.File(record["path"]) as b:
            n = len(a["state"])
            keys = []
            a.visititems(lambda k, v: keys.append(k) if isinstance(v, h5py.Dataset) else None)
            for key in keys:
                if "/final_" in key:
                    camera, terminal_key = key.split("/")
                    np.testing.assert_array_equal(
                        a[key][:], b[camera + "/" + terminal_key.removeprefix("final_")][n]
                    )
                else:
                    np.testing.assert_array_equal(a[key][:], b[key][:n])
            assert len(b["state"]) == n + recipe["data_extension"]["post_release_steps"]
    report["augmented_prefixes_array_exact"] = len(augmented["episodes"])
    from .input_checks import check_inputs

    report["live_recorded_inputs"] = check_inputs(recipe)
    from .act_oracle import diagnose_components

    report["act_validation_component_interventions"] = diagnose_components()
    plan = json.loads(Path("configs/v2/main_plan.json").read_text())
    for job in plan["jobs"]:
        manifest = json.loads((Path(job["checkpoint"]).parent / "manifest.json").read_text())
        assert manifest["training_episode_ids"] == augmented["selected_train_ids"]
        assert manifest["validation_episode_ids"] == augmented["selected_validation_ids"]
        assert manifest["config"]["updates"] == recipe["training"]["updates"]
        assert manifest["config"]["batch_size"] == recipe["training"]["batch_size"]
    report["matched_main_runs_verified"] = len(plan["jobs"])
    # Continue 7500 -> 8000, using the actual dataset, AdamW, scheduler and EMA.
    job = next(j for j in plan["jobs"] if j["name"] == "fusion_prior" and j["seed"] == 0)
    root = Path(job["checkpoint"]).parent
    resume_out = Path("artifacts/v2/verification/resumed_real_run")
    cfg = json.loads(Path(job["run_config"]).read_text())
    from .training import train

    train(recipe, cfg, resume_out, root / "resume_probe.pt")
    expected = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
    actual = torch.load(resume_out / "latest.pt", map_location="cpu", weights_only=False)

    def exact(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                exact(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                exact(x, y)
        else:
            assert a == b

    for key in ["model", "optimizer", "scheduler", "extra", "config", "normalization", "random"]:
        exact(expected[key], actual[key])
    for key in ["update", "best_validation"]:
        exact(expected["progress"][key], actual["progress"][key])
    report["real_500_update_resume_bit_exact"] = True
    report["resume_source"] = str(root / "resume_probe.pt")
    save_json("results/v2/pretest_verification.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report
