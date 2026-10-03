# GeoPolicy-Bench V3 — travail local en cours

**Question :** peut-on obtenir une imitation fiable en boucle fermée, puis
expliquer les changements utiles avant de reprendre la comparaison des caméras ?

Une baseline BC atteint 55/60 placements physiques sur une confirmation distincte,
à une tâche **un cube / un bac / consigne fixe**. Une interaction ciblée entre
30 actions réellement enregistrées après libération et un historique de quatre états robot
atteint 60/60 sur tuning, y compris au critère strict V2. Sa confirmation est
encore partielle : 10 scènes sur 60. Aucun score de test final V3 n'est disponible.

Le calcul s'est arrêté proprement sur le garde-fou mémoire Windows. Les résultats,
les checkpoints et la reprise sont conservés ; les tâches à plusieurs objets
et la nouvelle matrice caméra/prior ne sont pas encore exécutées.

- [Code V3](../../src/geopolicy/v3/), [configurations](../../configs/v3/).
- [Rapport provisoire et limites](report.md).
- [Commandes et reprise](reproduction.md), [journal détaillé](PROGRESS.md).
- [Résultats bruts](../../results/v3/validation/),
  [ablations](../../results/v3/ablation_summary.json),
  [audit des traces](../../results/v3/trace_audit.json).

Les checkpoints locaux sont sous `artifacts/v3/runs/`. La démonstration finale
avec checkpoint compact et vidéo reste à produire après la confirmation.
V1/V2 restent consultables dans le README public et leurs dossiers historiques.
Cette V3 n'a pas été publiée.
