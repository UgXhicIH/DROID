# Soft_DL_Pipe
Contains scripts for the complete pipeline

Plan de développement : fonction ajouté ou à ajouter à la pipeline existante, selon des degré de priorités.
- Priorité absolue🔴 - Priorité moyenne🟠 - Priorité minoritaire🟡 - Terminé🟢

Contraintes/élements essentielles : 
- Voué à évoluer sur une VM avec NVidia V100s, AMD Epyc 7452, 64Go RAM, OS Ubuntu
- Développement sur un VM avec Intel Xeon(R) Gold 6148 CPU, 64Go RAM, OS Windows (peut etre ammener à faire les calculs)
- Fonctionnement le plus en local possible
- Capacité de marcher sur CPU/RAM ou CPU/GPU/RAM selon la disponibilité hardware

1.Prédiction
- run_full_prediction(file, activities, with_physchem=True) 🟢
- predict_batch_peptides(sequences: list, activities)  : Prédiction sur une liste de séquences fournies en chat🟡
- predict_with_uncertainty_filter(file, activities, max_uncertainty=0.05) = Prédiction + filtre auto sur les MC bas🟠

2. Exploration
- get_top_n(run_id, activity, n=10) : Top N peptides pour une activité donnée🟠
- filter_peptides(run_id, criteria) : Filtre composable : proba > X, uncertainty < Y, length ∈ [A,B], charge > Z, etc.🟠
- get_peptide_details(run_id, sequence) : Toutes les infos sur un peptide donné (toutes prédictions, physchem, cluster…)🟡
- compare_peptides(seq1, seq2, run_id) : Comparaison côte à côte de deux peptides🟠
- summarize_run(run_id) : Stats globales : nb hits par activité, distribution des proba, distribution des uncertainties🟠
- cross_activity_analysis(run_id) : )Identifie les peptides "multi-actifs" (positifs sur plusieurs activités) ou "spécifiques"🟡

4. Analyse approfondie
- run_clustering(run_id, cutoff=0.7)  : Clustering PepFuNN + UMAP + Treemap, isolé🟢
- run_xai_top_n(run_id, activity, n=10)  : AI sur le top N (équivalent de l'actuel)🟡
- run_signalp(run_id, activities, threshold=0.95) : SignalP isolé🟡
- run_esmfold(run_id, activities, threshold, top_n) : ESMFold isolé🟡
- run_smiles_generation(run_id) : Génération SMILES + PTMs isolée🟡

6. Utilitaires
- compute_physchem_single(sequence) : Physchem complet pour une séquence unique🟠
- list_available_activities() : Renvoie la liste des activités prédictibles (évite hardcoding LLM-side)🟡
- list_active_modules() : Renvoie la liste des modules dispos (ESMFold, SignalP, XAI…) avec leur état🟡
- validate_sequence(sequence) : Vérifie qu'une séquence est valide (AA standards, longueur, PTMs reconnus)🟡
- clean_sequence(sequence) : Nettoie une séquence (retire PTMs entre parenthèses, normalise, caractères non cannonique/spéciaux)🟡
- get_file_info(file_path) : Inspecte un Excel uploadé : feuilles, nombre de peptides par feuille, colonnes dispo🟢

8. Gestion de runs
- list_recent_runs() : Liste les analyses passées avec leur run_id, date, fichier source🟠
- load_run(run_id)  : Recharge un run précédent pour pouvoir le requêter🟢
- delete_run(run_id) : Nettoie un run du disque🟡
- export_results(run_id, format) : Export filtrable et reformatable🟡

10. Interprétation
- deep_explain(question, context_type, context_data)  — types: , xai, mechanism, quality, general🟢 ⚠️Utilisation de Qwen3:32b_Q5

12. Contrôle qualité
- quality_control(run_id, activity, aspect) — coverage, calibration, uncertainty, all🟢

14. HADDOCK
- run_haddock_pipeline(run_id, activity, target_pdb, top_n=10) 🟠 ⚠️Necessite l'ESM Fold

16. Contrôle d'exécution
- cancel_running_task(task_id) 🟡
- get_task_status(task_id)🟡
