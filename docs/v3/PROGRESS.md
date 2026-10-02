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

En cours : superviseur séquentiel `python -m geopolicy.v3 baseline`, puis 80
démonstrations et dix démos de validation pour une comparaison BC/diffusion avec
2 000 updates identiques. Journaux/checkpoints sous `artifacts/v3/pipeline/baseline`
et `artifacts/v3/runs`. Les manifests de données sont archivés par hash pour
préserver la reprise des premiers runs après une collecte supplémentaire.
