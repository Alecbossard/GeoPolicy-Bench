# V4 — journal de reprise

Statut : pilote de validation en cours.

Historique V1/V2/V3 :10 217fichiers photographiés avant correction documentaire, sans changement des artefacts scientifiques. V3 :79démos retenues/80collectées,10validation ; scène10032exclue.

Contrats exécutés : données/points/actions nominaux identiques à V3, calibration recorded/live exacte, bruit axial vérifié par calcul indépendant, masques emboîtés, entrée caméra vide finie, reprise CUDA d’un update avec RNG augmentation exactement identique.

Budget principal :12modèles×2000updates,10conditions×20scènes×3seeds×4groupes=2400rollouts ; plafond32000updates/12heures. GPU unique.

Reprise pilote : .venv\Scripts\python.exe scripts/v4/run_jobs.py configs/v4/pilot_diagnostic_jobs.json. Logs/progression :artifacts/v4/jobs. Test500000–500019verrouillé. Aucune revendication avant les résultats.

## Pilote exécuté

V3seed0/10scènes : nominal fixe9/10,fusion10/10 ; occultation60% fixe0/10,fusion10/10 ; absence fixe0/10,fusion7/10 ; bruit10/25mm fixe9/10,fusion10/10 ; pointsmanquants70% fixe6/10,fusion10/10. Audit120rollouts/17519pas conforme, observationsinitiales identiques.

À1000updates/5scènes, fixe augmentée nominal2/5contre5/5clean ; fusionaug5/5clean5/5, absence3/5lesdeux. Ce résultat négatif précoce est conservé. Continuation au budgetV3prévu2000, sansaugmentation du budget ni conclusion fondée surloss. Les dossierspilotes sont conservés ; lespoids/rngà1000sont copiésdansmain_runs pour reprise indépendante.
