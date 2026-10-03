# Bullet CV proposée — aucune modification du CV

- Built and diagnosed a closed-loop imitation benchmark in MuJoCo/robosuite;
  achieved 98% stable single-object placements (147/150 test rollouts,
  three training seeds), with controlled camera/prior ablations, negative
  multi-object results and a checkpoint demo reproducing the complete action/physics trace.

Périmètre à conserver si cette bullet est utilisée : simulation, un objet/un bac,
50 scènes test distinctes répétées sur trois seeds. Aucune revendication de
grounding multicible, robot réel, gain général de fusion, PPO ou SmolVLA.
Faits : [résultats](../../results/v3/final_summary.json), [protocole](../../configs/v3/final_protocol.json),
[replay](../../artifacts/v3/demo/replay/verification.json).
