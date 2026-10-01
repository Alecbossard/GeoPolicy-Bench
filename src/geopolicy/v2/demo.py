"""Small local EMA-checkpoint bundle and exact replay of disclosed final-test scenes."""

import json
import shutil
import zipfile
from pathlib import Path
from geopolicy.io import save_json
from .config import file_hash, json_hash


def build_demo(recipe):
    import torch
    from .evaluation import evaluate

    root = Path("artifacts/v2/demo")
    root.mkdir(parents=True, exist_ok=True)
    # Predeclared demo policy: fusion with prior, training seed 0. Select the
    # earliest nominal success and earliest nominal failure, not the best seed.
    source = Path("artifacts/v2/main_runs/fusion_prior_s0/best.pt")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    compact = dict(
        model=saved["extra"]["ema"],
        config=saved["config"],
        normalization=saved["normalization"],
        progress=saved["progress"],
    )
    torch.save(compact, root / "policy.pt")
    restored = torch.load(root / "policy.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(v, restored["model"][k]) for k, v in saved["extra"]["ema"].items())
    compact_sha = file_hash(root / "policy.pt")
    # A compact checkpoint has different serialization/hash but exactly the
    # already frozen EMA parameters. Register this equivalence separately.
    equivalence = dict(
        source=str(source),
        source_sha256=file_hash(source),
        compact_sha256=compact_sha,
        ema_tensor_exact=True,
    )
    save_json("configs/v2/demo_equivalence.json", equivalence)
    rows = json.loads(Path("artifacts/v2/test/fusion_prior_s0/rollouts.json").read_text())
    nominal = sorted(
        [r for r in rows if r["condition"] == "nominal"], key=lambda r: r["scene_seed"]
    )
    selected = []
    for success in [True, False]:
        matches = [r for r in nominal if r["stable_success"] == success]
        if matches:
            selected.append(matches[0])
    save_json(root / "recipe.json", recipe)
    save_json(root / "expected_rollouts.json", selected)
    metadata = dict(
        checkpoint_sha256=compact_sha,
        recipe_sha256=json_hash(recipe),
        source_checkpoint_sha256=equivalence["source_sha256"],
        selected_scene_rule="Earliest nominal stable success and earliest nominal failure for predeclared fusion-prior seed 0; no selected success if none exists.",
        requires_training_dataset=False,
        expected_rollouts=selected,
    )
    save_json(root / "bundle.json", metadata)
    # For scoring, use the original frozen file; the compact model is then
    # independently replayed and compared by replay_demo in the clean env.
    for record in selected:
        out = f"results/v2/demo/{'success' if record['stable_success'] else 'failure'}"
        repeated = evaluate(
            recipe,
            source,
            out,
            record["scene_seed"],
            1,
            [record["condition"]],
            videos=1,
            frozen=True,
            demo_record=record,
        )[0]
        assert repeated["stable_success"] == record["stable_success"]
        assert repeated["steps"] == record["steps"]
        import numpy as np

        np.testing.assert_array_equal(repeated["first_action"], record["first_action"])
    (root / "README.md").write_text(
        "# Local demo\n\nFrom the project root, run `python -m geopolicy.v2 demo --bundle artifacts/v2/demo --out artifacts/v2/demo_reproduced`.\n\n"
        "Contains a trusted local compact EMA checkpoint, configuration, hashes and expected raw rollouts. No dataset download or training is needed. Videos show disclosed final-test examples selected by the stated rule; their success is not an aggregate performance claim.\n",
        encoding="utf8",
    )
    with zipfile.ZipFile("artifacts/v2/demo_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.iterdir()):
            archive.write(path, path.relative_to(root))
    return metadata


def replay_demo(bundle, out):
    from .evaluation import evaluate

    root = Path(bundle)
    meta = json.loads((root / "bundle.json").read_text())
    recipe = json.loads((root / "recipe.json").read_text())
    assert file_hash(root / "policy.pt") == meta["checkpoint_sha256"]
    assert json_hash(recipe) == meta["recipe_sha256"]
    reports = []
    for record in meta["expected_rollouts"]:
        expected = dict(record)
        expected["checkpoint_sha256"] = meta["checkpoint_sha256"]
        label = "success" if record["stable_success"] else "failure"
        rows = evaluate(
            recipe,
            root / "policy.pt",
            Path(out) / label,
            record["scene_seed"],
            1,
            [record["condition"]],
            videos=1,
            frozen=True,
            demo_record=expected,
        )
        actual = rows[0]
        import numpy as np

        np.testing.assert_array_equal(actual["first_action"], record["first_action"])
        assert actual["stable_success"] == record["stable_success"]
        assert actual["steps"] == record["steps"]
        assert (
            actual["initial_rgbd_robot_camera_pose_sha256"]
            == record["initial_rgbd_robot_camera_pose_sha256"]
        )
        reports.append(
            dict(
                scene_seed=record["scene_seed"],
                stable_success=record["stable_success"],
                first_action_exact=True,
                sensor_hash_exact=True,
                steps_exact=True,
            )
        )
    save_json(Path(out) / "verification.json", dict(verified=True, episodes=reports))
    print(json.dumps(reports, indent=2))
