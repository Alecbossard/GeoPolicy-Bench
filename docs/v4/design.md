# V4 : question et protocole de robustesse

La tâche reprend exactement V3 : un cube rouge, un bac bleu, une consigne fixe,
positions initiales variables et robot Panda simulé. La question est l'effet
de la vue (fixe ou fusion fixe/poignet) et des augmentations sur le placement
stable lorsque les entrées RGB-D se dégradent. La couleur et la consigne restent
connues ; il ne s'agit pas de généralisation multicible ou de transfert réel.

## Données et comparaison contrôlée

Les 80 collectes V3 comprennent 79 démonstrations réussies effectivement utilisées
et une collecte exclue, scène 10032 (échec de prise). Les 79 trajectoires continuées
totalisent 9 986 observations/actions. Les dix démonstrations enregistrées de
validation restent séparées. Les fichiers, actions, normalisations et ordre des
trajectoires sont identiques entre les quatre variantes. Aucun fichier V3 n'est
réenregistré. Les mentions historiques «80» dans les noms et plafonds de CLI restent
des identités de collecte, avec une correction documentaire explicite.

Quatre variantes : fixe sans/avec augmentations et fusion sans/avec augmentations.
Même DirectBC V3, historique causal de quatre états robot, prior chromatique manuel,
512 points au total, horizon de huit actions dont deux exécutées, pince continue,
2 000 updates, batch 32, trois seeds. Les commandes robot sont apprises intégralement.
Les labels fixes, capteurs et état robot constituent les seules entrées étudiantes.
Poses vérité terrain et phases de référence servent seulement à collecter ou noter.

Un RNG choisit les mêmes exemples/actions à chaque update pour les variantes d'une
même seed. Un RNG distinct choisit et applique les augmentations. Les checkpoints
sauvegardent ces deux RNG, les poids, AdamW, scheduler et EMA. La sélection du meilleur
EMA au sein d'un run reprend le L1 déployé sur les dix démos de validation propres.
Ce choix de checkpoint ne prouve pas la réussite : les décisions de recette et
les conclusions utilisent des rollouts en boucle fermée.

## Perturbations et portée

Les perturbations agissent **après** le recadrage de workspace, voxélisation et
échantillonnage V3 par caméra, **avant** la fusion et son budget final de 512 points.
Elles décrivent une dégradation de la représentation en entrée de la policy.
Elles ne reproduisent pas la formation complète d'une image profondeur physique.

| Famille | Intensités test | Opération |
|---|---|---|
| Occultation fixe | 25 %, 60 % de l'aire image | Carré centré indépendant des objets, projection calibrée des points retenus ; RGB et profondeur de ces points absents |
| Caméra fixe absente | 100 % | XYZRGB finis nuls et masque entièrement faux ; vue poignet inchangée |
| Bruit profondeur | σ = 3, 10, 25 mm | Gaussien axial sur chaque rayon caméra, pour les deux vues ; RGB inchangé |
| Points manquants | 30 %, 70 %, 90 % | Suppression IID de points retenus dans les deux vues |

Le bruit axial conserve le rayon et modifie Z dans le repère caméra avant son
équivalent en repère base. Il n'est pas un jitter XYZ isotrope. Les points invalides
sont masqués et remis à zéro ; aucune NaN n'est fournie à l'encodeur. L'occultation
n'utilise ni segmentation ni pose d'objet. Son taux de points retirés peut différer
de sa fraction d'aire image. Le taux manquant avant fusion diffère aussi du nombre
final de points : l'échantillonnage peut remplir le budget avec des points restants.
Les nombres effectifs sont conservés à chaque prédiction dans les fichiers coverage.

Un générateur déterministe dépend de scène et pas, jamais du modèle ou de sa seed
d'entraînement. Les tirages de taille fixe pour chaque caméra rendent les intensités
comparables et les masques de suppression emboîtés. Au début d'une scène, les deux
caméras propres et dégradées sont conservées avant sélection fixe/fusion : RGB,
profondeur métrique, calibration, XYZRGB, masques et état robot. Après divergence
des actions, la vue poignet suit sa propre trajectoire : les observations ultérieures
ne peuvent être identiques entre policies. L'appariement porte sur le même reset et
la même règle de perturbation, pas sur une trajectoire imposée.

## Pilote, sélection et test réservé

Le pilote des checkpoints V3 utilise 210000–210009 et six conditions. Une première
comparaison des quatre variantes à 1 000 updates sur cinq de ces scènes sert à
contrôler coût et comportement ; le scheduler reste prévu pour 2 000 updates.
La continuation à 2 000 updates respecte le budget V3. Le choix des augmentations
et l'évaluation de la conservation nominale utilisent la validation, puis une
confirmation distincte 210100–210109. Le test 500000–500019 reste verrouillé jusqu'au
gel des douze checkpoints, sources, versions et perturbations.

La recette initiale choisit un exemple propre dans 55 % des cas, caméra fixe absente
dans 15 %, occultation dans 10 %, bruit profondeur dans 10 %, points manquants dans
10 %. Une seule famille par exemple. Les intensités sont uniformes dans les plages
du plan. Cette distribution n'est pas une fréquence estimée des pannes réelles.
La caméra fixe seule ne dispose d'aucune observation visuelle pendant un dropout ;
les positions aléatoires ne sont pas récupérables par un état robot initial identique.
Ce manque d'information est une limite du dispositif, sans garantie qu'une
augmentation le résoudra.

Le pilote peut rejeter une recette pour baisse nominale ou absence d'amélioration.
Les résultats négatifs seront conservés. Aucun budget supplémentaire ne sera
justifié par une baisse de loss seule. La grande comparaison compte 12 modèles
et 2 400 rollouts test (20 scènes × 3 seeds × 4 variantes × 10 conditions).
Le plafond global est 32 000 updates et 12 heures depuis le début des jobs V4,
avec un seul processus lourd à la fois. Les critères PC V3 sont conservés :
commit libre ≥ 5,5 Gio avant démarrage lourd, VRAM libre ≥ 1 Gio, disque ≥ 10 Gio,
GPU < 80 °C ; arrêt contrôlé selon les relevés en cours de calcul.

## Mesures et incertitude

Le placement physique primaire conserve l'intégralité du cube dans le bac, une
libération constatée, absence de contact et vitesses limitées pendant une seconde.
Le score strict V2 secondaire exige aussi la pince ouverte pendant cette durée.
Le succès reste mémorisé dès cette fenêtre : aucune stabilité infinie n'est revendiquée.
Les traces contiennent actions, positions, rotation, vitesses, contacts et dwell.
Un audit indépendant reconstruit confinement et durée, puis recalcule chaque
perturbation à partir des observations initiales sauvegardées.

Les comparaisons appariées croisent rééchantillonnage de trois seeds et vingt scènes,
10 000 tirages et intervalles descriptifs à 95 %. Les neuf conditions altérées ont
un poids égal dans une moyenne secondaire. Elles sont des réglages fixes, pas neuf
nouvelles scènes indépendantes. Les intervalles ne sont pas corrigés pour multiplicité.
Les étapes d'échec sont des indices de diagnostic, sans attribution automatique à
une cause de perception ou de commande. Les gains de fusion et d'augmentation restent
des hypothèses jusqu'aux mesures ; les scènes test historiques V1/V2/V3 ne servent
pas à sélectionner V4.
