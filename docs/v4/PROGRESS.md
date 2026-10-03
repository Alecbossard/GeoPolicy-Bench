# V4 — journal de reprise

Statut : V4 livrée et vérifiée localement le 3 octobre 2026.

Historique V1/V2/V3 :10 217fichiers photographiés avant correction documentaire, sans changement des artefacts scientifiques. V3 :79démos retenues/80collectées,10validation ; scène10032exclue.

Contrats exécutés : données/points/actions nominaux identiques à V3, calibration recorded/live exacte, bruit axial vérifié par calcul indépendant, masques emboîtés, entrée caméra vide finie, reprise CUDA d’un update avec RNG augmentation exactement identique.

Budget principal :12modèles×2000updates,10conditions×20scènes×3seeds×4groupes=2400rollouts ; plafond32000updates/12heures. GPU unique.

Reprise finale : `.venv\Scripts\python.exe scripts/v4/run_jobs.py configs/v4/final_jobs.json`. Logs et progression : `artifacts/v4/jobs`. Test 500000–500019 ouvert uniquement après gel de `configs/v4/final_protocol.json`.

## Pilote exécuté

V3seed0/10scènes : nominal fixe9/10,fusion10/10 ; occultation60% fixe0/10,fusion10/10 ; absence fixe0/10,fusion7/10 ; bruit10/25mm fixe9/10,fusion10/10 ; pointsmanquants70% fixe6/10,fusion10/10. Audit120rollouts/17519pas conforme, observationsinitiales identiques.

À1000updates/5scènes, fixe augmentée nominal2/5contre5/5clean ; fusionaug5/5clean5/5, absence3/5lesdeux. Ce résultat négatif précoce est conservé. Continuation au budgetV3prévu2000, sansaugmentation du budget ni conclusion fondée surloss. Les dossierspilotes sont conservés ; lespoids/rngà1000sont copiésdansmain_runs pour reprise indépendante.

## Décision après pilote complet à2000updates

Fixe clean/aug strict et physique identiques : nominal9/10,occultation0/10,absence0/10,profondeur10/25mm9/10,missing70%6/10. Fusionclean nominal10/10,occ10/10,absence7/10,noise10/10,missing10/10 ; fusionaug nominal10/10,occ9/10,absence7/10,noise9/10,missing9/10. Aucun gain d’augmentation démontré ; recette conservée pour comparaison négative contrôlée sur trois seeds, comme demandé. Pas de recherche de recette supplémentaire ni de budget au-delà2000par modèle.

## Confirmation avant gel

720 rollouts complets, séparés des 420 pilotes. Nominal : fixe clean 27/30, fixe augmentée 29/30 ; fusion clean et augmentée 30/30. Moyenne des cinq altérations : augmentation du fixe +4,00 pp [−8,00 ; +14,67], de la fusion +1,33 pp [−6,00 ; +10,67]. Aucun gain d’augmentation établi. Décision et preuves dans [validation.md](validation.md).

Les sources du modèle, des données et des perturbations sont inchangées après validation. Avant le gel, seuls les gardes de version Python/paquets et la reprise d’écriture atomique Windows ont été renforcés, avec tests de verrouillage réels. Incident transitoire d’accès fichier pendant une sauvegarde Git : neuf rollouts conservés et reprise du dixième, sans nouvelle expérience ni perte de résultat. Pas de sauvegarde Git pendant les évaluations suivantes.

## Test réservé et vérification finale

Les 120 cellules et 2 400 rollouts sont complets. Comptages physiques nominaux /60 : fixe clean 56, fixe augmenté 54, fusion clean 60, fusion augmentée 58. Sur les neuf conditions altérées /540 : 272, 271, 483 et 470 respectivement. Aucune recette ni aucun checkpoint n'est ajusté après ces résultats.

24 000 updates d'entraînement ont été exécutés, plus cinq updates distincts pour le contrat de reprise. Les contrôles sans augmentation seed 0 reproduisent exactement les 120 traces pilotes des checkpoints V3.

L'audit indépendant a vérifié 3 540 rollouts et 541 220 pas, recomputé les perturbations et contrôlé les critères de géométrie et stabilité. Une composante RGB brute varie de 133 à 134 dans un NPZ ; les représentations initiales propres/perturbées des deux caméras restent exactement identiques avant sélection des vues. Fixe et fusion sélectionnent ensuite des entrées différentes, et leurs trajectoires ultérieures peuvent diverger. L'exception est conservée, sa cause n'est pas démontrée et aucun résultat n'est retiré ou remplacé. Le QC par champ est enregistré séparément.

La démo préspécifiée (fusion augmentée, seed 0, caméra fixe absente, première scène 500000) réussit et reproduit exactement ses 110 pas, observations initiales, actions, physique et critères d'évaluation. Un second environnement épinglé, avec copie isolée de la source, reproduit aussi la vidéo au même SHA. Un essai avait été arrêté avant Torch par le seuil de mémoire pendant l'audit ; il a réussi une fois ce processus terminé, sans abaisser le garde.

Aucun CV modifié, aucune publication en ligne. Les résultats négatifs sont conservés ; aucun transfert vers un robot réel n'a été évalué.

Vérification finale : `results/v4/delivery_verification.json`, statut `verified=true`. Conservation des 10 217 fichiers contrôlée, validation antérieure au gel puis au test, deux replays exacts et 174 liens Markdown valides. Les 1 184 mesures de ressources donnent un maximum GPU de 45 °C, un minimum de mémoire engagée disponible de 2,14 Gio et 21,32 Gio de disque libre. Durée jusqu'à cette vérification : 4,86 heures, sous le plafond de douze heures.
