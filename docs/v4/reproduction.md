# Reproduire GeoPolicy-Bench V4

Depuis `C:\Users\Alec\Documents\ChatGPT\GeoPolicy-Bench`, utiliser l'environnement
local validé et une seule tâche lourde à la fois sur la RTX 4060 Laptop 8 Go.
Les versions exactes et limites de ressources sont consignées dans le protocole
gelé et le [plan](../../configs/v4/plan.json). Les checkpoints d'optimisation servent
à reprendre les calculs ; le checkpoint compact de démonstration sert à l'inférence.

## Vérifier les résultats conservés

```powershell
.venv\Scripts\python.exe scripts/v4/report.py
```

Cette commande ne lance aucune simulation ni entraînement. Elle exige les 2 400
rollouts test et le QC `initial_sensor_qc.json`, vérifie l'appariement de chaque
entrée effective initiale, puis reconstruit
CSV, synthèse, courbes et documentation. Une erreur de complétude ou d'identité
interrompt la génération ; elle n'invente, ne filtre et ne remplace aucun résultat.
Le QC peut être recalculé avec `python scripts/v4/sensor_qc.py`. Il conserve
l'exception RGB brute et exige l'égalité exacte de toutes les entrées du modèle.

## Démo avec checkpoint local, sans données d'entraînement

```powershell
.venv\Scripts\python.exe scripts/v4/demo.py replay
```

Sorties : `artifacts/v4/demo/reproduced/` (MP4, GIF, trace, métriques et
vérification exacte des observations initiales, actions et physique). Le modèle
est fusion augmentée, seed 0, scène 500000, caméra fixe absente. Cette scène est
la première du test et a été choisie avant son observation, succès ou échec.
La vidéo distingue RGB physique du simulateur et représentation XYZRGB réellement
fournie à la policy. Le bundle local `artifacts/v4/demo_bundle.zip` comprend le
checkpoint compact, la source, les paramètres et les traces attendues. Après
extraction dans un dossier indépendant, depuis ce dossier :

```powershell
python scripts/v4/demo.py replay
```

Utiliser Python 3.11.9 et les versions verrouillées dans `requirements-lock.txt`.
La livraison a aussi exécuté ce replay avec une copie isolée de la source et
un second environnement local épinglé ; voir `delivery_verification.json`.

## Reprise et reproduction des calculs

Les phases sérielles conservent leurs logs et index terminés sous `artifacts/v4/jobs/`.
Une phase déjà terminée ne relance pas ses cellules. Après interruption :

```powershell
.venv\Scripts\python.exe scripts/v4/run_jobs.py configs/v4/main_train_jobs.json
.venv\Scripts\python.exe scripts/v4/run_jobs.py configs/v4/confirmation_jobs.json
.venv\Scripts\python.exe scripts/v4/run_jobs.py configs/v4/final_jobs.json
```

Le contrôleur ajoute `--resume` lorsqu'un checkpoint existe, sauvegarde chaque
rollout et vérifie les limites du PC avant chaque processus lourd. Le plafond
temporel global est enregistré dans `artifacts/v4/budget.json` ; son expiration
arrête les nouvelles cellules et conserve les résultats. Un nouveau budget ne
doit pas être ajouté silencieusement pour prolonger l'étude livrée.

Dans une copie indépendante du projet, l'entraînement exige les 79 HDF5 train
retenus **et les 10 HDF5 de validation enregistrée**, leurs manifests sous
`configs/v3/`, ainsi que `artifacts/v3/runs/bc_continued_history480_s0/best.pt`
pour reprendre exactement la normalisation V3. La démo compacte se passe de
ces données et de ce checkpoint de normalisation. Sans anciens runs V4 dans
cette copie, une cellule d'entraînement au même budget se lance ainsi :

```powershell
.venv\Scripts\python.exe -m geopolicy.v4 preflight
.venv\Scripts\python.exe -m geopolicy.v4 train --scope main --view fusion --augmented --seed 0
```

Omettre `--augmented` pour le contrôle et choisir `--view fixed` pour la caméra
fixe. Les seeds sont 0, 1 et 2. Les paramètres centraux restent ceux du plan gelé.
Pour une reproduction complète du test connu, conserver les 12 checkpoints et le protocole
gelé dans cette copie, sans fichiers de résultats/index V4 antérieurs, puis utiliser
`configs/v4/final_jobs.json`. Toute modification de recette requiert un autre test.

## Contrats d'une reproduction

Conserver les SHA des 79 démonstrations V3, le même ordre d'épisodes et RNG de
batches, le même budget et la normalisation V3. Une augmentation a son propre RNG
reprenable ; elle ne doit pas changer les tirages d'actions démontrées. Conserver
les observations initiales propres/altérées des deux caméras avant sélection de vue.
Les scènes 500000–500019 sont désormais connues : elles servent au replay, plus au
choix d'une nouvelle amélioration. Toute recette ultérieure a besoin d'un autre test.

Ne pas regénérer le protocole après accès au test. Choisir de nouveaux espaces V4
pour une reproduction distincte ; ne pas remplacer les bruts ni poids livrés.
Vérifier les commandes de calcul/démo dans le journal de livraison et la CLI V4
avant de lancer un entraînement ; les résultats de ce rapport ne prouvent pas
une reproduction sur un autre matériel ou une égalité bit à bit interplateforme.

[Vidéo de démonstration locale](../../artifacts/v4/demo/reproduced/demo.mp4) · [Checkpoint local](../../artifacts/v4/demo/checkpoint.pt)

[Rapport](report.md) · [Synthèse JSON](../../results/v4/summary.json) ·
[CSV complet](../../results/v4/rollouts.csv). Aucun artefact V1/V2/V3 n'est réécrit
par ce générateur ; aucun CV n'est modifié.
