# V4 — index des huit captures préspécifiées

Les huit clips sont fixés par le plan des jobs avant l'accès au test :
seed 0, première scène réservée 500000, conditions nominale et caméra fixe
absente pour chacun des quatre groupes. Aucun clip n'est choisi en fonction
de sa réussite ; les échecs restent présentés. Une seule scène illustrée
par clip ne constitue pas une estimation de performance.

Ces captures montrent le **RGB physique brut du simulateur**, avec les deux
caméras affichées. La corruption est appliquée à la représentation de points
après acquisition, crop, voxelisation et échantillonnage. L'image RGB affichée
reste donc disponible à l'écran lorsque les points de la caméra fixe sont
retirés de l'entrée de la policy. Ces clips ne montrent pas une caméra
physiquement débranchée ni l'entrée perturbée fournie au modèle.

La [démo principale](../../artifacts/v4/demo/reproduced/demo.mp4) affiche aussi la représentation effective XYZRGB :
RGB brut en haut, points retenus en bas. Elle rend visible la suppression
de l'entrée fixe. Les verdicts ci-dessous proviennent des rollouts et traces
JSON complets, plutôt que de l'apparence du clip.

- **Fixe, sans augmentation — Nominal :** [vidéo](../../artifacts/v4/evaluations/final_fixed_clean_s0_nominal/scene_500000.mp4), physique **réussi**, strict V2 **réussi**. [Résultat brut](../../results/v4/test/final_fixed_clean_s0_nominal.json). Chemin : `artifacts/v4/evaluations/final_fixed_clean_s0_nominal/scene_500000.mp4`. SHA256 : `9f114a9de1268c432feca282d58f7bf8e2547f6884ab118177604bdccb9bcef4`.
- **Fixe, sans augmentation — Caméra fixe absente :** [vidéo](../../artifacts/v4/evaluations/final_fixed_clean_s0_absent/scene_500000.mp4), physique **échoué**, strict V2 **échoué**. [Résultat brut](../../results/v4/test/final_fixed_clean_s0_absent.json). Chemin : `artifacts/v4/evaluations/final_fixed_clean_s0_absent/scene_500000.mp4`. SHA256 : `2d8faa38cdbeab0ab4b37075059d7d46ef68b585fb50e199f8000314075b0ea0`.
- **Fixe, avec augmentations — Nominal :** [vidéo](../../artifacts/v4/evaluations/final_fixed_aug_s0_nominal/scene_500000.mp4), physique **réussi**, strict V2 **réussi**. [Résultat brut](../../results/v4/test/final_fixed_aug_s0_nominal.json). Chemin : `artifacts/v4/evaluations/final_fixed_aug_s0_nominal/scene_500000.mp4`. SHA256 : `1657e4e0bcdabbee4617036f25881042a40127e2e4ac063a771544fc35f4dcd3`.
- **Fixe, avec augmentations — Caméra fixe absente :** [vidéo](../../artifacts/v4/evaluations/final_fixed_aug_s0_absent/scene_500000.mp4), physique **échoué**, strict V2 **échoué**. [Résultat brut](../../results/v4/test/final_fixed_aug_s0_absent.json). Chemin : `artifacts/v4/evaluations/final_fixed_aug_s0_absent/scene_500000.mp4`. SHA256 : `5e250e713031de12ff20c2965b6d0db6cd844c912dd3b8d6e80b0bd090f85bfd`.
- **Fusion, sans augmentation — Nominal :** [vidéo](../../artifacts/v4/evaluations/final_fusion_clean_s0_nominal/scene_500000.mp4), physique **réussi**, strict V2 **réussi**. [Résultat brut](../../results/v4/test/final_fusion_clean_s0_nominal.json). Chemin : `artifacts/v4/evaluations/final_fusion_clean_s0_nominal/scene_500000.mp4`. SHA256 : `47ac34dc0c404c2078b1966f077803d404e4461eaaa3853716454e4beb570ffb`.
- **Fusion, sans augmentation — Caméra fixe absente :** [vidéo](../../artifacts/v4/evaluations/final_fusion_clean_s0_absent/scene_500000.mp4), physique **réussi**, strict V2 **réussi**. [Résultat brut](../../results/v4/test/final_fusion_clean_s0_absent.json). Chemin : `artifacts/v4/evaluations/final_fusion_clean_s0_absent/scene_500000.mp4`. SHA256 : `730d810c489c4cee3958df7c3ebee272fee53f930d2ccecfddecbf02bf23f62f`.
- **Fusion, avec augmentations — Nominal :** [vidéo](../../artifacts/v4/evaluations/final_fusion_aug_s0_nominal/scene_500000.mp4), physique **réussi**, strict V2 **réussi**. [Résultat brut](../../results/v4/test/final_fusion_aug_s0_nominal.json). Chemin : `artifacts/v4/evaluations/final_fusion_aug_s0_nominal/scene_500000.mp4`. SHA256 : `fa2477dd742699877d61b3780fa947e8d93b15e3c3a7d8587f43df14a96a8b23`.
- **Fusion, avec augmentations — Caméra fixe absente :** [vidéo](../../artifacts/v4/evaluations/final_fusion_aug_s0_absent/scene_500000.mp4), physique **réussi**, strict V2 **réussi**. [Résultat brut](../../results/v4/test/final_fusion_aug_s0_absent.json). Chemin : `artifacts/v4/evaluations/final_fusion_aug_s0_absent/scene_500000.mp4`. SHA256 : `138be1526c1cc910c90ee6052be74ed73bc13e2811cd8589ce2ee2e7f010326f`.

[Rapport et résultats agrégés](report.md) · [Reproduction](reproduction.md).
