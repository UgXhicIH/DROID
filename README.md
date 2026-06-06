# 🧬 DROID — Pipeline agentique de prédiction d'activités biologiques des peptides

> **DROID** · *Deep learning for Residue Orchestration, Intelligence and Discovery*  
> Pipeline 100 % local · ESM-2 650M + CNN + Monte Carlo Dropout · Interface Streamlit + Ollama

---

## Présentation

PROTEOGEN est un pipeline de peptidomique computationnelle conçu pour prédire les activités biologiques de peptides à partir de leurs séquences, en s'appuyant sur une architecture de Deep Learning combinant :

- **ESM-2 (650M paramètres)** de Meta AI pour les embeddings token-level (1280D)
- **CNN 1D (Keras)** pour la classification binaire par activité
- **Monte Carlo Dropout (40 passes)** pour l'estimation bayésienne de l'incertitude
- **Platt Scaling post-hoc** pour la calibration des probabilités

L'interface est pilotée par un **agent LLM local (Ollama)** interprétant les requêtes en langage naturel, permettant à des chercheurs de lancer et configurer des analyses sans écrire de code.

---

## Architecture du projet

```
PROTEOGEN/
│
├── Dev_app_token.py               # Interface Streamlit — agent DROID (LangChain + Ollama)
├── Dev_proteogen_pipeline.py      # Pipeline principal + modules optionnels
├── proteogen_pipeline_token.py    # Version alternative du pipeline (token-level)
├── Clustering_PepFuNN_1280D_UMAP.py  # Module de clustering (PepFuNN/Butina + UMAP 3D)
├── Dev_deepseek_interpreter.py    # Interpréteur LLM post-clustering (Qwen3-32B)
├── restyle_droid.py               # Thème visuel DROID pour l'interface Streamlit
│
├── Model_TOKEN_CNN_PROTEOGEN_1280D/   # Modèles CNN Keras (28 activités)
│   └── *.keras
│
├── Graph_UniDL_Prediction_1280D/  # Graphiques HTML Plotly (XAI, radar VHSE, reliability)
├── Output_Cluster_1280D/          # Résultats clustering (UMAP, treemaps, master XLSX)
├── Logs_1280D/                    # Logs d'exécution horodatés
├── Output_Accession_Interest_XLSX_File_1280D/
├── Output_Accession_Interest_FASTA_File_1280D/
└── Output_SignalP_1280D/          # Résultats SignalP 6.0
```

---

## Modules disponibles

| Module | Description | Activation |
|--------|-------------|------------|
| **Prédictions DL** | ESM-2 + CNN + MC Dropout (28 modèles) | Toujours actif |
| **Physico-chimie** | MW, pI, VHSE, Z-Scales, Cruciani, MS-WHIM | Toujours actif |
| **SMILES** | Génération via RDKit + gestion PTMs |
| **XAI** | Alanine Scanning informatique (top 10 peptides) |
| **Clustering** | PepFuNN/Butina + UMAP 3D + Treemap (par feuille) |
| **SignalP 6.0** | Extraction FASTA + prédiction peptides signal |
| **ESMFold** | Prédiction structure 3D (`.pdb` + pLDDT) |
| **Interpréteur DeepSeek** | Analyse LLM approfondie des clusters | Via interface Streamlit |

---

## Activités biologiques prédites (28 modèles)

| Catégorie | Activités |
|-----------|-----------|
| **Signalisation** | Chimiotaxie, Cytokine, Régulation_Hormonale, Neuropeptide |
| **Transport** | Pénétration_cellulaire, Drug_Delivery |
| **Toxicité** | Activité_Hémolytique, Cytotoxicité, Neurotoxicité, Toxicité, Allergène |
| **Anti-infectieux** | Anti_Virale, Anti_Bacterien, Anti_Fongique, Anti_Parasitique, Anti_MRSA, Anti_Biofilm, Quorum_Sensing |
| **Cardiovasculaire / Métabolique** | Anti_Hyper_Tension, Dipeptidyl_peptidase_IV |
| **Inflammatoire / Neurologique** | Anti_Inflammatoire, Anti_Age, Anti_Amnésique |
| **Oncologie** | Anti_Cancer, Anti_Tumeur |
| **Autres** | Anti_Oxydant, Opioïde, Umami |

---

## Prérequis

### Environnement Python

```bash
python >= 3.10
```

### Dépendances principales

```bash
# Deep Learning
pip install tensorflow keras torch esm

# Bioinformatique
pip install biopython rdkit peptides peptidy

# Interface & Agent
pip install streamlit langchain langchain-ollama langgraph

# Clustering & Visualisation
pip install umap-learn plotly seaborn scikit-learn scikit-learn-intelex

# Utilitaires
pip install pandas numpy openpyxl
```

### Ollama (LLM local)

```bash
# Installation
curl -fsSL https://ollama.ai/install.sh | sh
https://ollama.com/install.ps1

# Modèle recommandé
ollama pull qwen2.5:14b

# Modèle interpréteur clusters (optionnel, nécessite GPU 32Go VRAM)
# ollama pull proteogen-qwen3-32b:q5
```

### SignalP 6.0 (optionnel)

Nécessite une installation séparée de [SignalP 6.0](https://services.healthtech.dtu.dk/services/SignalP-6.0/) accessible en ligne de commande via `signalp6`.

---

## Utilisation

### 1. Lancer l'interface Streamlit

```bash
streamlit run Dev_app_token.py
```

Par défaut, l'interface se connecte à Ollama sur `http://127.0.0.1:11434`.

### 2. Préparer le fichier d'entrée

Le fichier Excel d'entrée (`.xlsx`) doit contenir des feuilles nommées `*_Peptides`, chacune avec une colonne `Peptide` listant les séquences en acides aminés standard (une lettre par résidu).

**Exemple de structure :**

```
MonEtude_Peptides  (feuille)
├── Peptide         : ACDEFGHIKL
├── Accession       : pep_001      (optionnel)
└── ...autres colonnes ignorées
```

Les modifications post-traductionnelles (PTMs) au format `(-0.98)`, `(-17.02)`, `(+0.98)` sont supportées pour la génération SMILES.

### 3. Interagir avec l'agent DROID

L'agent comprend le langage naturel en français et anglais :

```
"Analyse ce fichier et prédit les activités Anti-bactérien et Quorum Sensing avec clustering."

"Fais les prédictions pour tous les microbes avec XAI."

"Recharge le run PROTEOGEN_Results_MonEtude_2026-05-21_15-51-54."

"Vérifie mon fichier Excel et dis-moi ce qu'il contient."

"Quelle est la qualité de calibration du modèle Anti_Cancer ?"
```

## Sorties générées

### Fichier Excel de résultats (`PROTEOGEN_Results_*.xlsx`)

Pour chaque feuille d'entrée, le fichier de résultat contient :

- `Peptide_<Activité>` — probabilité calibrée (Platt Scaling post MC Dropout)
- `Incertitude_<Activité>` — écart-type MC Dropout (σ)
- Paramètres physico-chimiques : `Molecular_Weight`, `Hydrophobicity`, `Isoelectric_Point`, `Charge_at_pH`, `Aliphatic_Index`, `Aromatic_Index`, `H_Donor`, `Sharing_Coefficient`
- Descripteurs VHSE (8), Z-Scales (5), Cruciani (3), MS-WHIM (3)
- Feuille `Matériel_et_Méthodes` : traçabilité complète des paramètres d'exécution

### Dashboard HTML interactif

Dashboard autonome (aucune dépendance serveur) avec :
- Tableau filtrable (DataTables) par feuille, avec tri par colonne et recherche plein texte
- Barres de probabilité colorées (rouge → vert) + badge de confiance MC Dropout
- Filtre multi-actifs (intersection ≥ 0.90, jusqu'à 5 activités simultanées)

### Graphiques HTML Plotly

| Fichier | Description |
|---------|-------------|
| `Radar_VHSE_<feuille>.html` | Profil VHSE moyen de la feuille |
| `PepFuNN_UMAP-<feuille>-<id>.html` | Espace chimique 3D (UMAP densMAP) |
| `Treemap_<feuille>-<id>.html` | Treemap clusters × activité dominante |
| `Motif_global_XAI_<activité>.html` | Motif actif récurrent par activité |
| `Reliability_<activité>.html` | Reliability diagram (calibration Brier/ECE) |

---

## Méthode scientifique

### Pipeline de prédiction

```
Séquence peptidique
      ↓
[Nettoyage PTMs & normalisation]
      ↓
[ESM-2 650M — embeddings token-level (1, L≤50, 1280D)]
      ↓
[CNN 1D Keras — 40 passes Monte Carlo Dropout]
      ↓
[Platt Scaling — calibration post-hoc]
      ↓
Probabilité calibrée + Incertitude (σ)
```

### Calibration des modèles

Chaque modèle est évalué par deux métriques de calibration calculées sur un jeu de validation indépendant :

- **Brier Score** : erreur quadratique moyenne (`< 0.05` excellent, `< 0.10` bon, `< 0.20` modéré)
- **ECE** (*Expected Calibration Error*) : écart confiance/précision calibrée (`< 0.03` très bien calibré)

Le Platt Scaling (`sigmoid(a·logit(p) + b)`) est appliqué aux probabilités MC Dropout brutes, avec les paramètres `a` et `b` estimés par régression logistique sur le jeu de validation.

### Clustering chimique (PepFuNN/Butina)

1. Génération des fingerprints Morgan (rayon 4, 1024 bits) via RDKit
2. Calcul de la matrice de distance Tanimoto
3. Clustering Butina avec seuil configurable (défaut `cutoff=0.7`)
4. Réduction dimensionnelle UMAP 3D (densMAP, `n_neighbors=20`, `min_dist=0.1`)
5. Treemap de composition par cluster et activité dominante

---

## Interpréteur DeepSeek — Analyse de clusters

Après clustering, l'interpréteur LLM (`Dev_deepseek_interpreter.py`) permet d'analyser un cluster sélectionné via un modèle Qwen3-32B (Q5) chargé dynamiquement sur Ollama :

- Composition du cluster (taille, profils, longueurs)
- Activités biologiques sur-représentées
- Peptides clés (haute probabilité, multi-actifs)
- Hypothèses mécanistiques et pistes expérimentales

Le modèle est déchargé et rechargé automatiquement pour optimiser l'utilisation VRAM.

---

## Notifications

PROTEOGEN envoie des notifications de fin de run via [ntfy.sh](https://ntfy.sh). Le topic par défaut est configurable :

```bash
export PROTEOGEN_NTFY_TOPIC="mon_topic_personnalise"
export PROTEOGEN_NTFY_URL="https://ntfy.sh"  # ou instance auto-hébergée
```

---

## Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `PROTEOGEN_NTFY_TOPIC` | `proteogen_UgXhicIH` | Topic ntfy pour les notifications |
| `PROTEOGEN_NTFY_URL` | `https://ntfy.sh` | URL du serveur ntfy |
| `OLLAMA_KEEP_ALIVE` | `5m` | Durée de conservation des modèles Ollama en mémoire |
| `TF_ENABLE_ONEDNN_OPTS` | `1` | Optimisations oneDNN pour TensorFlow (CPU) |
| `KMP_DUPLICATE_LIB_OK` | `True` | Évite les conflits MKL/OpenMP |

---

## Limites connues

- **Longueur maximale** des séquences : 50 acides aminés (padding zéro au-delà, troncature si dépassement)
- **ESMFold** nécessite un GPU avec ≥ 32 Go de VRAM
- **SignalP 6.0** doit être installé séparément et accessible via `signalp6` en ligne de commande
- Le modèle interpréteur `proteogen-qwen3-32b:q5` nécessite également ≥ 32 Go de VRAM
- La génération SMILES avec gestion des PTMs peut être lente pour de grandes listes de peptides

---

## Auteurs & Crédits

- **Pipeline PROTEOGEN & interface DROID** — Morgan Letoux
- **Module clustering PepFuNN** — Rodrigo Ochoa (Novo Nordisk), adapté et étendu par Morgan Letoux  
  *(UMAP densMAP, export HTML, clustering par feuille, support multi-activités)*

### Dépendances scientifiques clés

| Outil | Référence |
|-------|-----------|
| ESM-2 | Lin et al., *Science*, 2023 |
| PepFuNN / Butina | Ochoa et al. ; Butina, *JCIM*, 1999 |
| UMAP | McInnes et al., 2018 |
| SignalP 6.0 | Teufel et al., *Nature Biotechnology*, 2022 |
| RDKit | [rdkit.org](https://www.rdkit.org) |

---

## Licence

Ce projet est à usage interne de recherche.
