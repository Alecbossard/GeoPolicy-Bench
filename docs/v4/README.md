# GeoPolicy-Bench V4 — robustesse de policies RGB-D

**Question :** caméra fixe ou fusion fixe+poignet résistent-elles à l'occultation,
à l'absence de caméra, au bruit de profondeur et aux points manquants, et les
augmentations aident-elles sans sacrifier la performance nominale ?

**Résultats exécutés :** un cube, un bac, consigne fixe ; 20 nouvelles scènes ×
3 seeds, mêmes 79 démonstrations retenues sur 80 collectes, données/actions/budgets
appariés. Les valeurs altérées moyennent neuf perturbations synthétiques fixes.

| Recette | Nominal | Neuf altérations |
|---|---:|---:|
| Fixe, sans augmentation | 56/60 (93.3 %) | 272/540 (50.4 %) |
| Fixe, avec augmentations | 54/60 (90.0 %) | 271/540 (50.2 %) |
| Fusion, sans augmentation | 60/60 (100.0 %) | 483/540 (89.4 %) |
| Fusion, avec augmentations | 58/60 (96.7 %) | 470/540 (87.0 %) |

Placement physique stable après libération ; les deux critères coïncident dans ce test.
La fusion sans augmentation conserve le meilleur taux observé sur ce banc.
La recette d'augmentation n'apporte pas de gain global démontré : les intervalles
appariés de ses deux contrastes moyens recouvrent zéro. Détails dans le rapport.

[Vidéo de démonstration locale](../../artifacts/v4/demo/reproduced/demo.mp4) · [Checkpoint local](../../artifacts/v4/demo/checkpoint.pt)

[Huit captures préspécifiées avec verdicts réels](videos.md) : RGB brut du
simulateur ; la démo principale montre aussi les points fournis à la policy.

![Courbes de robustesse physique](figures/robustness_physical.png)

[Rapport et intervalles](report.md) · [Reproduction](reproduction.md) ·
[Résultats bruts](../../results/v4/test/) · [Code V4](../../src/geopolicy/v4/) ·
[Paramètres centraux](../../configs/v4/plan.json).

**QC :** les représentations initiales propres et perturbées des **deux caméras**
sont identiques entre variantes **avant sélection des vues**. Fixe et fusion
consomment ensuite des sous-ensembles différents ; leurs observations et
trajectoires ultérieures peuvent diverger avec leurs actions.
Une variation d'un niveau sur une composante RGB brute est conservée et
[documentée dans le rapport](report.md), avec tous les résultats originaux.

Pour rejouer la démo depuis le projet :

```powershell
.venv\Scripts\python.exe scripts/v4/demo.py replay
```

Le checkpoint compact et le [bundle autonome](../../artifacts/v4/demo_bundle.zip)
se passent des données d'entraînement. Reconstruction des résultats :
`python scripts/v4/report.py`.

**Limites :** perturbations après échantillonnage, RGB-D simulé/calibration idéale,
prior couleur manuel, objets connus et tâche simple. Trois seeds et 20 scènes limitent
les conclusions ; aucun transfert vers un robot réel ou gain universel de fusion
n'est revendiqué. Les échecs et les versions V1/V2/V3 sont conservés séparément.
