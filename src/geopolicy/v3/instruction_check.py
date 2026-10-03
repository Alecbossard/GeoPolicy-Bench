"""Initial-state instruction counterfactual; oracle commands are diagnostic only."""

import json
import numpy as np
import torch
from .common import ROOT, sha, write
from .models import build
from .stages import environment, stage_live
from geopolicy.sensors import student_state


def instruction_check():
    torch.set_num_threads(2)
    models = {}
    for recipe in ("continued_history4", "continued_state1"):
        checkpoint = (
            ROOT / f"artifacts/v3/runs/bc_two_objects_one_goal_{recipe}80_s0/best.pt"
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build(saved["config"]).eval()
        model.load_state_dict(saved["extra"]["ema"])
        models[recipe] = (model, saved, sha(checkpoint))
    env = environment("two_objects_one_goal")
    rows = []
    try:
        for scene in range(111100, 111120):
            obs = env.reset_scene(scene)
            history = [student_state(obs).copy()]
            reference = []
            predictions = {name: [] for name in models}
            for object_id in (0, 1):
                env.target_object = object_id
                reference.append(env.reference_action().copy())
                for name, (model, saved, _) in models.items():
                    batch = stage_live(
                        env, obs, history, saved["config"], saved["normalization"]
                    )
                    with torch.inference_mode():
                        predicted = model.predict(batch)[0, 0].numpy()
                    norm = saved["normalization"]
                    predictions[name].append(
                        np.clip(
                            predicted * norm["action_std"] + norm["action_mean"], -1, 1
                        )
                    )
            rows.append(
                dict(
                    scene_seed=scene,
                    diagnostic_reference_first_actions=np.asarray(reference).tolist(),
                    learned_first_actions={
                        k: np.asarray(v).tolist() for k, v in predictions.items()
                    },
                )
            )
    finally:
        env.close()
    summary = {}
    for name in models:
        learned = np.asarray([r["learned_first_actions"][name] for r in rows])
        reference = np.asarray([r["diagnostic_reference_first_actions"] for r in rows])
        delta = learned[:, 1, :3] - learned[:, 0, :3]
        reference_delta = reference[:, 1, :3] - reference[:, 0, :3]
        summary[name] = dict(
            mean_instruction_motion_change=float(np.linalg.norm(delta, axis=1).mean()),
            diagnostic_reference_mean_change=float(
                np.linalg.norm(reference_delta, axis=1).mean()
            ),
            first_y_direction_matches=int(
                ((learned[:, :, 1] > 0) == (reference[:, :, 1] > 0)).sum()
            ),
            y_commands=40,
            checkpoint_sha256=models[name][2],
        )
    result = dict(
        task="two_objects_one_goal",
        scenes=[111100, 111119],
        diagnostic=True,
        reference_used_in_student_actions=False,
        rows=rows,
        summary=summary,
        limitation="Initial-state response only; oracle action magnitude is a reference, not the uniquely correct learned command. No unique representation or optimization cause is established.",
    )
    write(ROOT / "results/v3/instruction_diagnosis.json", result)
    print(json.dumps(summary), flush=True)
    return result
