# V4 — validation et décision avant le test réservé

La confirmation comprend **720 rollouts** : quatre groupes × trois seeds ×
six conditions × dix scènes 210100–210109. Elle est disjointe du pilote et
du futur test 500000–500019. Tous les modèles ont terminé le même budget de
2 000 updates sur 79 démonstrations réussies, retenues sur 80 collectes V3.

| Groupe | Condition | Physique /30 | Physique seeds0,1,2 /10 | Strict V2 /30 |
|---|---|---:|---|---:|
| fixed_clean | nominal | 27/30 | [9, 9, 9] | 27/30 |
| fixed_clean | occlusion60 | 0/30 | [0, 0, 0] | 0/30 |
| fixed_clean | absent | 0/30 | [0, 0, 0] | 0/30 |
| fixed_clean | depth010 | 28/30 | [9, 10, 9] | 28/30 |
| fixed_clean | depth025 | 29/30 | [10, 9, 10] | 29/30 |
| fixed_clean | missing70 | 18/30 | [5, 5, 8] | 18/30 |
| fixed_aug | nominal | 29/30 | [10, 10, 9] | 29/30 |
| fixed_aug | occlusion60 | 0/30 | [0, 0, 0] | 0/30 |
| fixed_aug | absent | 0/30 | [0, 0, 0] | 0/30 |
| fixed_aug | depth010 | 28/30 | [10, 9, 9] | 28/30 |
| fixed_aug | depth025 | 28/30 | [10, 9, 9] | 28/30 |
| fixed_aug | missing70 | 25/30 | [10, 7, 8] | 25/30 |
| fusion_clean | nominal | 30/30 | [10, 10, 10] | 30/30 |
| fusion_clean | occlusion60 | 28/30 | [10, 10, 8] | 28/30 |
| fusion_clean | absent | 23/30 | [8, 8, 7] | 23/30 |
| fusion_clean | depth010 | 30/30 | [10, 10, 10] | 30/30 |
| fusion_clean | depth025 | 30/30 | [10, 10, 10] | 30/30 |
| fusion_clean | missing70 | 29/30 | [10, 10, 9] | 29/30 |
| fusion_aug | nominal | 30/30 | [10, 10, 10] | 30/30 |
| fusion_aug | occlusion60 | 27/30 | [9, 9, 9] | 27/30 |
| fusion_aug | absent | 25/30 | [7, 8, 10] | 25/30 |
| fusion_aug | depth010 | 30/30 | [10, 10, 10] | 30/30 |
| fusion_aug | depth025 | 30/30 | [10, 10, 10] | 30/30 |
| fusion_aug | missing70 | 30/30 | [10, 10, 10] | 30/30 |

## Comparaisons appariées descriptives

| Première − seconde | Nominal physique, pp [IC95] | Moyenne cinq altérations physique, pp [IC95] | Moyenne cinq altérations strict, pp [IC95] |
|---|---:|---:|---:|
| fixed_aug − fixed_clean | +6.67 [-13.33, +30.00] | +4.00 [-8.00, +14.67] | +4.00 [-8.00, +14.67] |
| fusion_aug − fusion_clean | +0.00 [+0.00, +0.00] | +1.33 [-6.00, +10.67] | +1.33 [-6.00, +10.67] |
| fusion_clean − fixed_clean | +10.00 [+0.00, +30.00] | +43.33 [+32.67, +52.00] | +43.33 [+32.67, +52.00] |
| fusion_aug − fixed_aug | +3.33 [+0.00, +20.00] | +40.67 [+30.00, +54.00] | +40.67 [+30.00, +54.00] |

Bootstrap apparié croisé : trois seeds, dix scènes, 10 000 tirages, RNG81 ;
intervalles descriptifs à95 %, sans correction de multiplicité. Les cinq réglages
altérés sont moyennés à poids égaux dans chaque paire seed/scène avant les tirages.
Les répétitions ne créent pas 150 scènes indépendantes. Au plafond/plancher, un
intervalle nul ne révèle pas les événements non observés. Ce protocole n'établit
pas une non-infériorité nominale et un contraste ambigu ne prouve pas l'équivalence.

## Pilote conservé séparément :420 rollouts

| Phase | Fichiers | Rollouts exécutés |
|---|---:|---:|
| V3 diagnostic | 12 | 120 |
| pilot1000 | 12 | 60 |
| pilot2000 | 24 | 240 |

Le diagnostic V3 seed0 compte120 épisodes ; le pilote à1000updates compte60
épisodes sur cinq scènes et un sous-ensemble de conditions ; le pilote complet
à2000updates compte240 épisodes. Ces observations connues ont motivé la comparaison
de la recette A unique. Elles restent exclues des720 rollouts de confirmation.
Au pilote complet, fixe clean/aug est identique et fusion augmentée perd quatre
succès sur les cinquante rollouts altérés seed0. Aucun gain d'augmentation n'y a
été démontré ; les résultats à1000updates et les checkpoints pilotes sont conservés.

## Décision

**Procéder au test réservé avec les douze checkpoints et la recette A inchangée.**
Cette décision repose sur la validité des données/entrées, du clipping, de la
complétude et de l'appariement. La matrice négative ou ambiguë demeure une
comparaison contrôlée utile demandée par l'utilisateur ; aucun seuil de succès
ou gain de fusion/augmentation n'est imposé pour publier les résultats locaux.
Aucun gain d'augmentation n'est revendiqué à partir de cette validation limitée.
Aucun nouveau réglage, budget ou choix n'est effectué à partir du futur test.

Les observations initiales propres/altérées des deux modalités sont sauvegardées
et recomputées ; les sept commandes finies restent bornées à[−1,1]. Les deux vues
ont exactement les mêmes entrées initiales avant sélection de vue. Les perturbations
agissent après recadrage/voxelisation/échantillonnage de512points par caméra ; elles
ne modélisent pas une panne RGB-D physique complète. Aucun sim-to-real réel évalué.

[Décision et preuves JSON](../../results/v4/validation_decision.json) ·
[Bruts de validation](../../results/v4/validation/) · [Plan](../../configs/v4/plan.json).
