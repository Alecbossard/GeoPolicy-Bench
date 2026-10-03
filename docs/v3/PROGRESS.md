# GeoPolicy-Bench V3 — reprise du 2 octobre 2026

Statut : EN COURS. Branche locale `codex/geopolicy-v3`. Un agent principal,
un seul travail GPU lourd à la fois. Aucune publication ni modification du CV.

Le prompt V3, le README, le rapport V2, le journal historique et l'audit indépendant
ont été lus intégralement. V1 reste la référence de performance ; V2 a renforcé
le diagnostic sans démontrer de gain nominal. Les données continuées ne sont pas
présumées meilleures.

Tous les nouveaux fichiers utilisent les espaces `src/geopolicy/v3`, `configs/v3`,
`results/v3`, `artifacts/v3` et `docs/v3`. L'inventaire de préservation inclut les
fichiers suivis et les données, poids et résultats historiques locaux.

Première étape : un cube physique, une destination, une consigne fixe ; contrôle
scripté identifié comme oracle de diagnostic, puis surapprentissage de quatre
trajectoires. Comparaison de la recette diffusion V1 et d'une imitation directe
avec les mêmes entrées, données, actions et budgets. Aucun score de surapprentissage
ne sera présenté comme généralisation.

Le plan réserve 400000–400049 sans accès avant gel des choix. Les scènes de réglage
et de confirmation sont distinctes. Le seuil opérationnel avant une grande matrice
est 12/20 placements physiques stables sur chacun de ces jeux pour trois seeds.
Un échec de ce seuil impose un diagnostic ciblé ou une livraison négative honnête.

La métrique V2 reste calculée à l'identique comme score secondaire. La métrique
physique V3 exige une libération observée puis une seconde de contenance, vitesse
faible et absence de contact ; une fermeture ultérieure de pince vide est
enregistrée séparément. Cette modification ne sera pas utilisée pour gonfler
rétroactivement les scores V1/V2.

## Premiers résultats exécutés

- Inventaire initial : 6 210 fichiers historiques vérifiés, aucune divergence.
- Six démonstrations scriptées (quatre train, deux validation) : toutes réalisent
  le placement stable ; provenance oracle déclarée, aucun score étudiant.
- Replay de 132 pas : états exacts ; RGB et points exacts aux quatre instants
  vérifiés ; roundtrip actions maximal 5,96e-8. Commande +X déplace +55,7 mm ;
  pince −1 ouvre à 79,0 mm, +1 ferme à 1,93 mm.
- Gates CUDA, batch 32, deux renderers actifs : prochain update identique après
  restauration du modèle, AdamW et RNG pour BC et diffusion. Plus de 6,4 Go de
  VRAM libres aux gates, GPU ≤45 °C.
- Surapprentissage de quatre préfixes, 2 000 updates : BC et diffusion réalisent
  4/4 placements physiques sur les scènes train. Score strict V2 : BC 0/4,
  diffusion 4/4. Ce sont des diagnostics, pas une preuve de généralisation.
- BC quatre démos sur validation distincte : physique 11/20, strict V2 0/20 ;
  six échecs de prise et trois de transport. Les onze différences de métrique
  sont des échecs de posture de pince avec objet stable dans les traces.
- Diffusion quatre démos : validation physique et stricte 6/20. Aucune supériorité
  générale n'est conclue de ce contrôle à une seed.
- Audit indépendant des 48 rollouts / 10 415 pas initiaux : contenance reconstruite
  à partir de pose/rotation/bac par une autre formulation géométrique ; dwells,
  libération et deux scores concordent.

La collecte initiale a rencontré deux conversions de booléens NumPy non
sérialisables ; le nouveau code a été corrigé et la première sortie incomplète
conservée sous `artifacts/v3/repaired_attempts`. Aucun fichier historique affecté.

## Reprise du 3 octobre 2026

L'inventaire de 6 210 fichiers V1/V2 a été vérifié à nouveau : aucune divergence.
Baseline et ablations terminées ; aucun ancien worker restant lors de la reprise.
Les manifests de données sont archivés par hash pour préserver la reprise des
premiers runs après une collecte supplémentaire.

Sur 80 démos / 2 000 updates / batch 32, BC original obtient, par seed,
20/20, 17/20, 17/20 placements physiques sur tuning, puis 20/20, 18/20,
17/20 sur confirmation distincte. Le seuil de compétence à une tâche simple
est franchi. Diffusion seed 0 : 2/20 physique, 1/20 strict ; pas de grande
matrice lancée sur cette recette non fonctionnelle.

Les trois ablations exécutées avec trois seeds donnent sur tuning : original
54/60 physique (43/60 strict), continuation seule 50/60 (50/60 strict),
pince binaire 42/60 (38/60 strict), historique d'état seul 0/60 (0/60 strict).
La continuation conserve la normalisation des préfixes originaux.

Replay avec historique : 102 pas, maximum d'erreur d'entrée nul. Les échecs de
l'historique sont principalement après libération. L'analyse de 8 209 échantillons
après libération trouve 7 278 contacts et 5 578 vitesses linéaires excessives ;
aucun des 60 épisodes ne montre une seconde continue d'objet suffisamment calme.
Le mécanisme unique n'est pas établi ; ce sont des mouvements/contact mesurés,
pas seulement une fermeture de pince vide.

Un contrôle ciblé ajoute la continuation à l'historique : quatrième cellule
du plan factoriel données ×historique. Il teste la couverture du retrait après
libération à budget constant, sans supposer un gain des suffixes V2 PPO.
Terminé : `python -m geopolicy.v3 interaction`, sorties sous
`artifacts/v3/pipeline/interaction`. La nouvelle cellule obtient 20/20 physique
et strict V2 sur chacune des trois seeds de tuning (60/60). Les intervalles
descriptifs et les capteurs initiaux appariés sont dans
`results/v3/ablation_summary.json`. Tous les résultats négatifs restent conservés.

La réintroduction de deux cubes / un bac est préparée dans un espace distinct,
avec scènes/configs propres. Aucun nouveau test réservé n'a été ouvert.

## Confirmation et arrêt mémoire du 3 octobre

Le protocole de confirmation est enregistré avant exécution dans
`configs/v3/interaction_confirmation_plan.json` : 110200–110219, trois seeds,
sans entraînement supplémentaire. La recette n'est promue que si chaque seed
obtient au moins 12/20 aux deux critères, que son total physique est au moins
celui du contrôle original et que son total strict est supérieur. Les capteurs
initiaux doivent être identiques entre recettes.

La seed 0 a exécuté 10 scènes : 10/10 physique et strict. **Évaluation partielle,
aucune sélection finale.** Le garde-fou a arrêté proprement le worker après trois
mesures sous 1 Go de mémoire engagée libre (605, 392, 438 Mio). Le processus
occupait environ 3,52 Gio privés ; GPU 43 °C, plus de 6,6 Go de VRAM libres et
23 Go de disque libres. Il s'agit de la mémoire Windows, pas d'un problème GPU.
Traces et identités conservées sous
`artifacts/v3/evaluations/bc_continued_history480_s0_confirmation`.

Après l'arrêt, environ 4,1 Gio de mémoire engagée libre : marge insuffisante
pour relancer le même worker avec le garde-fou de 1 Gio. Une intervention de
l'utilisateur (fermeture d'onglets/applications inutiles, cible environ 6 Gio
libres avant lancement) est demandée. Aucun processus utilisateur fermé,
aucun paramètre Windows ou pilote modifié, aucun seuil abaissé.

Pendant cette attente : audit indépendant de **438 rollouts / 74 442 pas**,
géométrie et dwells concordants ; analyse des contacts après libération sauvegardée
sans nouveau rendu ; trois tests des gates de validation et un test du garde-fou
de démarrage réussis. L'inventaire de 6 210 fichiers historiques est à nouveau
intact (vérification 10:51:58 UTC). Le pipeline
progressif consomme la sélection validée, et le passage à la tâche complète
requiert la confirmation à trois seeds de la tâche deux objets / un bac.

Le nouveau preflight léger lit la mémoire avant d'importer PyTorch CUDA. À
10:52:48 UTC, il confirme 4,12 Gio de commit libre, GPU 41 °C / 7 126 Mio libres,
et refuse le démarrage sans charger le worker. Le seuil initial de 6 Gio est
documenté dans `configs/v3/runtime_limits.json` ; les seuils scientifiques et
le garde-fou pendant calcul restent inchangés. Aucun nouveau job lourd actif.

Reprise exacte après récupération de mémoire :
`python -m geopolicy.v3 interaction-confirmation`. Les dix scènes terminées
seront conservées et seules les scènes restantes seront exécutées.

## Mémoire récupérée et confirmation achevée

À 10:57:34 UTC, 8,07 Gio de commit Windows libres, GPU 39 °C : le preflight
autorise la reprise. Les dix scènes sauvegardées sont conservées. Confirmation
achevée à 10:59:42 UTC : **20/20 physique et strict V2 pour chaque seed**, soit
60/60, contre 55/60 physique et 39/60 strict pour le contrôle original sur
les mêmes scènes. Capteurs initiaux identiques. La règle préenregistrée est
franchie ; `configs/v3/selection.json` retient continuation + quatre états robot.
Ces résultats restent de la validation, pas un test final.

La collecte et le pilote à deux cubes / un bac démarrent dans leur espace propre,
avec cette recette et les mêmes 2 000 updates. Un résultat insuffisant du pilote
arrêtera la progression, avant la matrice caméra/prior.

## Diagnostic à deux objets et portée de l'étude

80 démos train et 10 validation scriptées : toutes stables aux deux critères.
Pilote continuation/historique4 : **0/20**, 13 échecs d'approche, 7 de prise,
aucun mauvais objet levé, 11 épisodes avec collision. Replay de 124+135 pas
sur les deux consignes, 30 comparaisons de vues : états, historique normalisé,
points, masques et tokens exacts ; deux cubes et un seul bac vérifiés.

Contrôle isolant historique4→état courant : **0/20**, dont 2 échecs de sélection.
Changement initial de consigne sur 20 scènes : commande de translation change
en moyenne de 0,047 (h4) / 0,076 (h1), contre 1,6 pour la référence diagnostique ;
direction Y concordante dans 20/40 commandes pour chaque recette.
Pas d'action oracle dans les rollouts étudiants.

Pilote de routage des moments XYZRGB caméra vers le décodeur : **1/20**.
Même encodeur (calcul vérifié exactement prior on/off), données, normalisation,
état courant, loss, pince et 2 000 updates ; aucun GT objet ou phase teacher.
Résultat négatif conservé. Dernier contrôle borné : BC sur quatre trajectoires
à deux objets, 2 000 updates, **4/4 placements stricts sur train**. Le pipeline
peut apprendre ces trajectoires ; aucune généralisation n'est démontrée.

La progression à deux destinations est arrêtée ; pas de matrice sur la tâche
complète. Déviation explicite : vues fixe/poignet/fusion et prior on/off seront
comparés sur la tâche simple confirmée, avec mêmes 80+10 démos, actions,
normalisations, trois seeds et 2 000 updates. Aucune conclusion sur le grounding
multicible ne découlera de cette matrice. La recette diffusion V1 sera également
répétée sur les deux seeds manquantes pour compléter la comparaison principale.
Le plan est enregistré dans `configs/v3/single_study_plan.json`.
