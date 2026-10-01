"""Small, named hypotheses instead of hardcoded one-off training scripts."""

from .config import file_hash, json_hash

PILOTS = {
    "diffusion_suffix_only": {
        "mode": "diffusion",
        "view": "fusion",
        "history": 1,
        "binary_gripper": False,
        "color_prior": "chroma40",
    },
    "diffusion_binary": {
        "mode": "diffusion",
        "view": "fusion",
        "history": 1,
        "binary_gripper": True,
        "color_prior": "chroma40",
    },
    "diffusion_history": {
        "mode": "diffusion",
        "view": "fusion",
        "history": 4,
        "binary_gripper": True,
        "color_prior": "chroma40",
    },
    "act_prior_only": {
        "mode": "act_rgb_prior",
        "view": "fusion",
        "history": 1,
        "binary_gripper": False,
        "color_prior": False,
    },
    "act_history_binary": {
        "mode": "act_rgb_prior",
        "view": "fusion",
        "history": 4,
        "binary_gripper": True,
        "color_prior": False,
    },
    "act_point": {
        "mode": "act_point",
        "view": "fusion",
        "history": 4,
        "binary_gripper": True,
        "color_prior": "chroma40",
    },
}


def run_config(recipe, name, seed=0, pilot=True, overrides=None):
    config = dict(PILOTS[name])
    if overrides:
        config.update(overrides)
    t = recipe["training"]
    config.update(
        version=2,
        name=name,
        seed=seed,
        updates=t["pilot_updates"] if pilot else t["updates"],
        batch_size=t["batch_size"],
        horizon=t["horizon"],
        execute_steps=t["execute_steps"],
        inference_steps=t["inference_steps"],
        point_budget=t["point_budget"],
        rgb_resolution=64,
        normalization_path=recipe["data_extension"]["normalization_path"],
        training_recipe=dict(t),
        dataset_manifest_sha256=file_hash(recipe["dataset_manifest"]),
        normalization_sha256=file_hash(recipe["data_extension"]["normalization_path"]),
        recipe_sha256=json_hash(recipe),
    )
    return config
