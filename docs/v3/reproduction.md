# Reproduire GeoPolicy-Bench V3 en local

Dossier : `C:\Users\Alec\Documents\ChatGPT\GeoPolicy-Bench`.
Toutes les commandes partent de ce dossier, avec un seul worker lourd à la fois.
Les artefacts V1/V2 sont conservés ; aucune publication V3 n'est requise.

## Démo courte avec le checkpoint local

```powershell
Set-Location 'C:\Users\Alec\Documents\ChatGPT\GeoPolicy-Bench'
.venv\Scripts\python.exe scripts/v3/replay_demo.py
```

Entrées : source, [protocole gelé](../../configs/v3/final_protocol.json),
[checkpoint compact](../../artifacts/v3/demo/checkpoint.pt) et
[preuve d'équivalence](../../configs/v3/demo_equivalence.json).
Aucun dataset ou checkpoint historique n'est nécessaire à cette démo.
Sorties : `artifacts/v3/demo/replay/demo.mp4`, images de début/fin et
`verification.json`. La commande exige les mêmes versions et vérifie exactement
les capteurs initiaux, toutes les actions/mesures physiques et les scores de la
première scène test préspécifiée 400000, seed d'entraînement 0.

Le [bundle portable local](../../artifacts/v3/portable_demo/) contient sa propre
source figée, les configurations, le checkpoint et son script. La vérification
dans le second environnement utilise explicitement cette copie source :

```powershell
.venv-repro\Scripts\python.exe artifacts/v3/portable_demo/replay_demo.py
```

Le script place le `src` du bundle en tête des imports, plutôt que le package
installé dans l'environnement. Il écrit sous `artifacts/v3/portable_demo/replay/`.
Cette égalité a été vérifiée sur ce PC ; elle n'est pas garantie bit à bit sur
une autre plateforme, un autre backend graphique ou des versions différentes.
La vidéo montre un exemple préspécifié, pas une estimation de performance.

## Environnement et paramètres

Versions vérifiées : Windows, Python 3.11.9, Torch 2.7.1+cu128, NumPy 1.26.4,
MuJoCo 3.3.7, robosuite 1.5.2, h5py 3.14.0. Les autres dépendances sont dans
le [lock historique](../../requirements-lock.txt), inchangé.
Les environnements existants `.venv` et `.venv-repro` sont utilisables.
Pour une installation distincte dans un **nouveau** dossier d'environnement :

```powershell
uv venv --python 3.11.9 .venv-v3
uv pip install --python .venv-v3\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-v3\Scripts\python.exe -r requirements-lock.txt
uv pip install --python .venv-v3\Scripts\python.exe --no-deps -e .
.venv-v3\Scripts\python.exe scripts/v3/replay_demo.py
```

Ce guide n'installe aucun pilote et ne modifie aucun paramètre système.
La reconstruction d'un nouvel environnement `.venv-v3` n'a pas été exécutée
pour cette livraison ; le replay indépendant utilise `.venv-repro` déjà présent.

| Fichier | Rôle |
|---|---|
| [plan.json](../../configs/v3/plan.json) | Budgets, dimensions, scènes, horizon, stabilité et limites pendant calcul |
| [runtime_limits.json](../../configs/v3/runtime_limits.json) | Garde-fou avant import Torch : commit libre ≥5,5 Gio |
| [runs/](../../configs/v3/runs/) | Configuration effective de chaque expérience, overrides compris |
| [selection.json](../../configs/v3/selection.json) | Décision sur validation/confirmation, avant test |
| [final_protocol.json](../../configs/v3/final_protocol.json) | 30 checkpoints et hashes figés, 50 nouvelles scènes par checkpoint |
| [demo_equivalence.json](../../configs/v3/demo_equivalence.json) | Identité du checkpoint compact et trace attendue |

Les paramètres spécifiques de modèle (loss, EMA) restent lisibles dans la source
et le rapport. La configuration du run/checkpoint décrit la recette exécutée :
`continued=true`, `history=4`, pince continue. Les valeurs par défaut de la CLI ne suffisent
pas à identifier le modèle retenu.

## Entraînement et validation d'une reproduction

Les 90 HDF5 locaux du [dataset simple](../../configs/v3/single_dataset.json)
contiennent 80 épisodes train et 10 validation enregistrée. Les versions de
manifest sont conservées par SHA256 sous `configs/v3/datasets/`. La continuation
conserve les préfixes et ajoute 30 observations/actions réellement exécutées.
Cet exemple utilise un nom neuf et ne remplace aucun run livré :

```powershell
.venv\Scripts\python.exe -m geopolicy.v3 train --name local_v3_retrain_s0 --model direct_bc --seed 0 --updates 2000 --limit 80 --continued --history 4
.venv\Scripts\python.exe -m geopolicy.v3 evaluate --name local_v3_retrain_validation_s0 --checkpoint artifacts/v3/runs/local_v3_retrain_s0/best.pt --first 110100 --episodes 20 --videos 1
```

Il s'agit de reproduire la recette fixée ; les scènes de validation sont connues.
Une future amélioration devra avoir sa propre identité et un nouveau test réservé.
Le jeu final V3 400000–400049 ne peut plus servir au tuning.

Les superviseurs `baseline`, `confirmation`, `ablations`, `interaction`,
`interaction-confirmation` et `single-study` décrivent les expériences exécutées.
Leur état local conserve les étapes terminées. Les pilotes `progressive`,
`two-object-control` et `routed-control` ont produit les résultats négatifs
documentés ; le passage à la tâche complète reste interdit par leur gate.

## Reprendre un calcul interrompu

Un entraînement sauvegarde tous les 500 updates : poids, AdamW, scheduler, EMA,
normalisation, RNG Python/NumPy/Torch/CUDA et RNG des batches. Reprendre avec
les mêmes arguments et `--resume` ; `latest.pt` est le point de reprise.
La démo compacte contient uniquement les paramètres d'inférence et ne peut
pas reprendre l'optimisation.

Chaque rollout terminé et sa trace sont sauvegardés atomiquement. La commande
suivante reprend les épisodes manquants du test gelé, ou ne fait rien si tout
est complet ; elle refuse les changements de source, runtime ou checkpoint :

```powershell
.venv\Scripts\python.exe -m geopolicy.v3 final-study
```

Ne relancez pas `freeze` : le protocole ne doit jamais être regénéré après accès
au test. Vérifiez l'absence d'un autre worker avant de lancer un superviseur ;
deux superviseurs indépendants ne se coordonnent pas entre eux.

Les limites sont actives avant import Torch et pendant calcul. Elles surveillent
commit Windows, VRAM, température et disque. Un démarrage refusé laisse les
épisodes et checkpoints disponibles. Reprenez après récupération des ressources,
sans désactiver les garde-fous ni modifier le protocole figé.

## Vérifier et reconstruire les livrables

Les commandes lisent les artefacts locaux et écrivent uniquement V3 :

```powershell
.venv\Scripts\python.exe scripts/v3/summarize.py
.venv\Scripts\python.exe -m geopolicy.v3 audit
.venv\Scripts\python.exe -m geopolicy.v3 preserve
.venv\Scripts\python.exe scripts/v3/resource_summary.py
.venv\Scripts\python.exe scripts/v3/make_figures.py
.venv\Scripts\python.exe scripts/v3/build_report.py
.venv\Scripts\python.exe scripts/v3/verify_delivery.py
```

`export_demo.py` reconstruit le checkpoint compact/bundle depuis la source seed0
figée ; exécutez ensuite les deux replays. L'export demande les checkpoints locaux
complets, tandis que le replay portable s'en passe. Les agrégats exigent les
30×50 rollouts complets ; l'audit exige leurs traces ; la vérification des données
exige les 180 HDF5 simples et deux objets.

L'exception d'empreinte des capteurs originale est conservée dans les bruts.
`diagnose_sensor_anomaly.py` rejoue les deux séquences de 15 scènes sans changer
les politiques, et écrit le contrôle séparé nécessaire à l'agrégation de cette
livraison. Les scores complets gardent les 50 scènes ; seul le contraste apparié
affecté utilise 49 scènes par seed. Cette analyse de sensibilité a été ajoutée
après le contrôle final ; elle ne modifie ni le modèle sélectionné ni le test.

Les dix tests V3 ont réussi avant le gel des sources :

```powershell
.venv\Scripts\python.exe -m pytest -q tests/v3
```

[Rapport](report.md), [résultats bruts](../../results/v3/test/),
[audit](../../results/v3/trace_audit.json), [livraison vérifiée](../../results/v3/delivery_verification.json).
Les poids/datasets/traces volumineux sont locaux et ignorés par Git ; V3
n'a pas été publiée en ligne.
