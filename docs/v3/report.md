# Rapport V3 provisoire — 3 octobre 2026

La manipulation apprise fonctionne régulièrement sur la tâche simplifiée.
Le projet V3 n'est pas encore terminé : confirmation de la recette candidate,
progression de complexité, étude des vues, test réservé et démo finale restent
à exécuter. Le worker est arrêté par le garde-fou mémoire Windows ; aucune
modification des pilotes ou des paramètres système n'est nécessaire au code.

## Références historiques à critères comparables

Le test V2 nominal est désormais connu et sert de référence historique.
Sur les mêmes 150 épisodes et le critère strict V2 : fusion V1 réévaluée 16/150,
fusion V2 10/150 ; ACT V1 réévalué 9/150, ACT V2 0/150.
Voir le [rapport V2](../v2_report.md) pour les protocoles et limites.

Les scores V3 ci-dessous portent sur **un seul cube et un seul bac**. Ils ne
permettent pas de conclure que V3 améliore la tâche complète V1/V2. La comparaison
V1/V2/V3 sur de nouvelles scènes identiques n'est pas encore exécutée.

## Contrôles exécutés

Le modèle reçoit RGB-D/points XYZRGB, état robot et labels connus de consigne.
Il ne reçoit ni pose d'objet vérité terrain, ni phase teacher ; aucune commande
du modèle n'est remplacée par un script. Les démonstrations sont produites par
une référence scriptée avec provenance oracle déclarée. Les poses ne servent
qu'à cette référence et aux métriques.

- Replay de 132 pas : états exacts, images et points exacts à quatre instants.
  Roundtrip des actions : erreur maximale 5,96e-8.
- Contrôle +X : déplacement +55,7 mm ; pince −1 ouvre à 79,0 mm, +1 ferme à
  1,93 mm dans le backend réel.
- Reprise d'optimisation CUDA : prochain update identique après restauration
  modèle/AdamW/scheduler/RNG pour BC et diffusion, deux renderers actifs.
- Surapprentissage diagnostique de quatre démos : les deux modèles placent
  4/4 objets sur leurs scènes train. Ce résultat ne prouve pas la généralisation.
- Audit indépendant de 438 rollouts et 74 442 pas : contenance reconstruite par
  fonction de support du cube, dwells et scores concordants. Il inclut les dix
  scènes de confirmation partielles, sans les transformer en score complet.

## Recettes et résultats de validation

Toutes les ablations BC utilisent les mêmes 80 démonstrations, les mêmes actions,
2 000 updates, batch 32, chunks de huit actions, exécution de deux actions.
La normalisation est calculée sur les préfixes originaux et conservée lors de
l'ajout des suffixes. L'historique concerne **l'état robot**, pas une suite d'images.
Les trois seeds d'entraînement sont 0, 1 et 2 ; chaque seed rencontre les mêmes
20 scènes de tuning 110100–110119, avec capteurs initiaux vérifiés identiques.

| Recette BC | Physique, par seed | Total physique | Total strict V2 |
|---|---|---:|---:|
| Préfixes originaux, état courant | 20, 17, 17 | 54/60 | 43/60 |
| Continuation seule (+30 actions réelles) | 19, 20, 11 | 50/60 | 50/60 |
| Historique de quatre états seul | 0, 0, 0 | 0/60 | 0/60 |
| Sortie pince binaire seule | 14, 19, 9 | 42/60 | 38/60 |
| Continuation + quatre états | 20, 20, 20 | 60/60 | 60/60 |

La baseline originale est confirmée sur 110200–110219 : physique 20, 18, 17,
soit 55/60 ; strict 17, 17, 5, soit 39/60. La recette combinée n'a encore exécuté
que les dix premières scènes de seed 0 (10/10 aux deux critères). Elle n'est
pas sélectionnée tant que la confirmation complète n'a pas satisfait la règle
[enregistrée](../../configs/v3/interaction_confirmation_plan.json).

La recette diffusion V1 réentraînée sur ces mêmes 80 préfixes et le même budget,
seed 0 seulement, obtient 2/20 physique et 1/20 strict. Son L1 enregistré est
plus faible que celui de BC ; cela ne correspond pas à un meilleur comportement.
Ce contrôle à une seed ne suffit pas à attribuer l'écart à l'architecture seule.

## Causes établies et hypothèses

L'historique rejeté est cohérent entre replay enregistré et inférence : 102 pas,
erreur d'entrée maximale nulle. Dans ses 60 rollouts, les 8 209 échantillons après
libération comprennent 7 278 contacts de pince et 5 578 vitesses linéaires
supérieures à 0,02 m/s ; aucun épisode n'a une seconde continue d'objet calme,
même en ignorant les contacts. Ses échecs ne se réduisent donc pas à une pince
qui se referme dans le vide. La cause unique du comportement reste indéterminée.

La quatrième cellule du plan données × historique améliore les scores de tuning
alors que chacun de ses facteurs isolés ne les améliore pas. Cela établit une
interaction dans les expériences exécutées, sans identifier à elle seule le
mécanisme. La couverture du retrait après libération est une hypothèse compatible
avec les traces. Ce suffixe scripté V3 n'est pas le suffixe PPO V2 : aucun gain
de PPO ou de la continuation V2 n'est revendiqué.

Le bootstrap apparié seed/scène (10 000 tirages) est descriptif : seulement trois
seeds, validation réutilisée, aucune correction de multiplicité. La différence
combinée–originale sur tuning est +10 points physique (intervalle descriptif
0 à +25), +28,33 points strict (+8,33 à +51,67). Ce ne sont pas des conclusions
de test final. L'historique augmente aussi la dimension d'entrée ; les suffixes
modifient la distribution des exemples malgré un budget d'updates constant.

## Deux critères de stabilité explicites

Les deux scores exigent contenance complète du cube tourné, hauteur valide,
levée antérieure >0,89 m, vitesses linéaire ≤0,02 m/s et angulaire ≤0,25 rad/s,
absence de contact pendant une seconde (21 observations à 20 Hz).

Le score physique V3 exige d'abord une libération observée avec pince ouverte
≥45 mm ; une fermeture ultérieure sans contact ne réinitialise pas la stabilité
de l'objet. Le score strict V2 conserve son exigence de pince ouverte pendant
tout le dwell. Les échecs de posture seule sont enregistrés séparément.
Les anciens résultats V1/V2 ne sont pas recalculés avec une définition assouplie.

## Suite et reproductibilité

La reprise sauvegardée est [documentée](reproduction.md). Le nouveau test réservé
reste verrouillé ; aucune de ses scènes n'a servi au tuning. La démo finale, la
vérification dans l'environnement de reproduction et une bullet CV vérifiée
restent à livrer. Aucun CV ou fait du profil maître n'a été modifié ; aucun
nouveau résultat n'a été publié.
