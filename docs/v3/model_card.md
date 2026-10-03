# Fiche du checkpoint local V3

Recette : DirectBC, fusion fixe+poignet, prior chroma40, historique de quatre
états robot, continuation enregistrée, pince continue. Entraînement :80démos,
2 000 updates sur RTX 4060 Laptop 8 Go. Sélection de recette sur tuning et
confirmation disjointe, avant le test. Poids déployés : EMA.

Checkpoint source : [best.pt](../../artifacts/v3/runs/bc_continued_history480_s0/best.pt).
Checkpoint d'inférence compact : [checkpoint.pt](../../artifacts/v3/demo/checkpoint.pt),
1484661 octets. Il ne contient pas les états d'optimiseur/reprise.
SHA256 compact : `145d23e7c164907fe9a5c7be04b1c5cb28a0b8709ac1234b394fec8965c942b8`.
SHA256 source : `93ddead2631290b9afd73faa777864f7824d9d671fb9b8fd0a25b2952cc03808`.
[Preuve de conversion](../../configs/v3/demo_equivalence.json).

Entrées autorisées :512 points XYZRGB/masques, 92 valeurs d'état robot normalisé,
quatre labels connus, normalisation des actions. Les labels ne sont pas du texte
libre. Aucune pose objet GT, segmentation GT ou phase teacher à l'inférence.
Sorties :8 actions de 7 dimensions ; deux actions exécutées, puis nouvelles
observations. Dé-normalisation et clipping ; −1ouvre/+1ferme la pince.

Usage validé : **un cube rouge/un bac bleu, placements aléatoires simulés**.
Résultat agrégé de cette recette, trois checkpoints seeds 0/1/2 : physique
147/150 et strict V2 147/150 sur 50 nouvelles scènes.
Résultat du checkpoint seed 0 : 49/50 physique et
49/50 strict. La vidéo montre uniquement scène400000,
première scène préspécifiée ; elle n'est pas une estimation de performance.

Reproduction vérifiée sur ce PC et deux environnements aux versions identiques,
avec égalité complète de la trace actions/physique de cette démo. L'égalité
bit à bit sur d'autres plateformes, GPU ou bibliothèques n'est pas garantie.
La démo fonctionne avec le bundle local sans données d'entraînement ni checkpoints
historiques. Le checkpoint complet demeure nécessaire pour reprendre l'optimisation.

Limites : tâche multicible non maîtrisée ; profondeur idéale ; objets/couleurs
connus ; pas de variation physique majeure, langage libre ou transfert robot.
Le succès exige1 seconde de stabilité après release, pas une stabilité indéfinie.
Les scripts/teachers GT sont des références de collecte/diagnostic, séparés des
politiques étudiantes. Voir les [causes et limites](report.md).
