"""Separate actual actor changes from critic changes after PPO."""

import json
from pathlib import Path
import torch
from stable_baselines3 import PPO

torch.set_num_threads(2)
bc = PPO.load("artifacts/teacher_pilot/bc_initial.zip", device="cpu").policy.state_dict()
rows = []
for label, path in [
    ("selected_2048", "artifacts/teacher_selected.zip"),
    ("continuation_34816", "artifacts/teacher_main/latest.zip"),
]:
    updated = PPO.load(path, device="cpu").policy.state_dict()
    actor = []
    critic = []
    for key in bc:
        difference = (updated[key] - bc[key]).float().flatten()
        if key.startswith(("mlp_extractor.policy_net", "action_net")) or key == "log_std":
            actor.append(difference)
        else:
            critic.append(difference)
    rows.append(
        {
            "checkpoint": path,
            "label": label,
            "actor_parameter_delta_l2_from_bc": float(torch.cat(actor).norm()),
            "critic_other_parameter_delta_l2_from_bc": float(torch.cat(critic).norm()),
            "log_std_delta_l2": float((updated["log_std"] - bc["log_std"]).norm()),
        }
    )
report = {
    "updates": rows,
    "provenance": "BC actor initialization then SB3 on-policy PPO; actor/critic separated",
    "end_to_end_task_sequencing_claimed": False,
}
Path("results/teacher_update_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
