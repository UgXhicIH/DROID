# Copyright (c) 2026 Morgan Letoux. All rights reserved.
# This file is part of PROTEOGEN/DROID.
# Unauthorized use, reproduction or distribution is strictly prohibited.
# See LICENSE for details.
"""
PROTEOGEN — Interface web agentique
Streamlit + LangChain + Ollama (100 % local)
"""
import os
os.environ['OLLAMA_KEEP_ALIVE']= '5m'
import re
import json
import tempfile
import time
import logging
import threading
from typing import Optional
import streamlit as st #type: ignore
import streamlit.components.v1 as components #type: ignore
from streamlit.runtime.scriptrunner import add_script_run_ctx #type: ignore
from langchain_core.tools import tool #type: ignore
from langchain_ollama import ChatOllama #type: ignore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage #type: ignore
from langgraph.prebuilt import create_react_agent #type: ignore

# Import du pipeline (heavy imports exécutés une seule fois par le cache Streamlit)
import Dev_proteogen_pipeline as pipeline
import Dev_deepseek_interpreter as deepseek_interp
import restyle_droid

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="PROTEOGEN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

restyle_droid.apply()
restyle_droid.sidebar_brand()

# Override scope Dev_ : texte encre foncée (lisible) sur l'alerte success de la zone principale.
# Cause racine : restyle_droid.py:325 fixe le fond vert #A1C9A1 sans couleur de texte (prod, non modifié).
st.markdown(
    """
    <style>
    [data-testid="stMain"] [data-testid="stAlertContentSuccess"],
    [data-testid="stMain"] [data-testid="stAlertContentSuccess"] * { color:#1a1a1a !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BARRE LATÉRALE
# ============================================================
with st.sidebar:
    st.divider()

    # ── Upload fichier ────────────────────────────────────────
    st.subheader("📂 Fichier d'entrée")
    uploaded_file = st.file_uploader(
        "Uploader un fichier Excel (.xlsx)",
        type=["xlsx"],
        help=(
            "Le XLSX doit contenir des feuilles nomméees '*_Peptides' "
            "avec une colonne 'Peptide'."
        ),
    )

    if uploaded_file:
        # Sauvegarde dans un dossier temporaire persistant dans la session
        prev_name = st.session_state.get("uploaded_file_name")
        if prev_name != uploaded_file.name:
            tmp_dir   = tempfile.mkdtemp(prefix="proteogen_")
            file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["uploaded_file_path"] = file_path
            st.session_state["uploaded_file_name"] = uploaded_file.name
        st.success(f"Prêt : **{uploaded_file.name}**")
    else:
        st.session_state.pop("uploaded_file_path", None)
        st.session_state.pop("uploaded_file_name", None)
        st.info("Aucun fichier chargé")

    st.divider()

    # ── Configuration LLM ─────────────────────────────────────
    st.subheader("⚙️ Modèle Ollama")
    ollama_model = st.selectbox(
        "Modèle",
        ["Qwen2.5:14b",
         "Qwen2.5:7b",
         "llama3.1:8b",
         "gpt-oss:20b"
         ],
        index=0,
        help=("Sélection du LLM Ollama :\n\n"
        "Qwen2.5:14b: recommandé, équilibre performances/ressources\n\n"
        "Qwen2.57:b | llama3.1:8b: adapté aus machines limitées en hardware\n\n"
        "gpt-oss:20b: modèle le plus puissant, nécessite GPU 32Go VRAM"
        )
    )
    ollama_base_url = st.text_input("URL Ollama", value="http://127.0.0.1:11434")

    st.divider()

    # ── Activités disponibles ─────────────────────────────────
    st.subheader("🔬 Activités disponibles")
    with st.expander("Voir la liste complète", expanded=False):
        for act in pipeline.AVAILABLE_ACTIVITIES:
            st.markdown(f"- `{act}`")

    st.divider()

    # ── Modules activables ────────────────────────────────
    st.subheader("🧠 Modules disponibles")
    with st.expander("Voir la liste complète", expanded=False):
        for act in pipeline.ALL_MODULES:
            st.markdown(f"- `{act}`")

    st.divider()
    # ── Bouton réinitialisation conversation ──────────────────
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state["messages"]      = []
        st.session_state["chat_history"]  = []
        st.session_state["pipeline_result"] = None
        st.rerun()

# ── Log capture & suivi de progression ───────────────────
class StreamlitUIHandler(logging.Handler):
    def __init__(self, log_list, thread_state=None):
        super().__init__()
        self.log_list = log_list
        self.thread_state = thread_state
        self._activity_seen = 0

    def emit(self, record):
        msg = self.format(record)
        self.log_list.append(msg)
        if self.thread_state and "Analyse en cours :" in msg:
            self._activity_seen += 1
            total = self.thread_state.get("total_activities", 1) or 1
            self.thread_state["progress"] = min(0.50 + 0.40 * self._activity_seen / total, 0.90)
            self.thread_state["progress_label"] = (
                f"Pipeline : {self._activity_seen}/{total} activité(s) traitée(s)..."
            )


class PipelineLogFilter(logging.Filter):
    _BLOCKED = frozenset({"httpx", "httpcore", "urllib3", "requests",
                          "h11", "asyncio", "filelock", "PIL", "matplotlib"})
    def filter(self, record):
        return record.name.split(".")[0] not in self._BLOCKED
# ============================================================
# OUTIL LANGCHAIN
# ============================================================
THREAD_STATE = {
    "file_path": None,
    "pipeline_result": None
}
@tool
def lancer_pipeline_proteogen(
    target_activities: list,
    run_smiles: bool = False,
    run_clustering: bool = False,
    run_signalp: Optional[list] = None,
    run_xai: bool = False,
    run_esmfold: bool = False,
    esmfold_activities: Optional[list] = None,
    esmfold_threshold: float = 0.95,
    esmfold_top_n: int = 30,
) -> str:
    """
    Lance le pipeline PROTEOGEN de prédiction d'activités biologiques des peptides.
    Utiliser cet outil dès que l'utilisateur demande une prédiction, une analyse ou le lancement du pipeline.

    Args:
        target_activities: Liste des noms exacts des activités à prédire. [] = toutes.
            Noms valides (EXACTS, 28 activités) :
            Chimiotaxie, Cytokine, Pénétration_cellulaire, Drug_Delivery, Régulation_Hormonale,
            Activité_Hémolytique, Cytotoxicité, Neurotoxicité, Toxicité,f Allergène,
            Quorum_Sensing, Anti_Virale, Anti_Bacterien, Anti_Fongique, Anti_Parasitique,
            Anti_MRSA, Anti_Biofilm, Anti_Hyper_Tension, Dipeptidyl_peptidase_IV,
            Anti_Inflammatoire, Anti_Age, Anti_Amnésique, Neuropeptide, Anti_Cancer,
            Anti_Tumeur, Anti_Oxydant, Opioïde, Umami.

        run_smiles: True si l'utilisateur souhaite générer les colonnes SMILES (RDKit).
        run_clustering: True si l'utilisateur mentionne clustering, UMAP, t-SNE.
        run_signalp: Liste d'activités pour l'extraction FASTA + SignalP 6.0. [] ou None = ignoré.
        run_xai: True pour l'analyse XAI (Alanine Scanning top 10). False par défaut.
        run_esmfold: True pour prédiction structure 3D ESM-Fold. False par défaut (nécessite GPU).
        esmfold_activities: Sous-liste d'activités pour ESM-Fold. None = même liste que target_activities.
        esmfold_threshold: Seuil probabilité pour ESM-Fold (défaut 0.95).
        esmfold_top_n: Nombre max de peptides par activité à folder (défaut 30).
    """
    input_file = THREAD_STATE["file_path"]
    if not input_file:
        return (
            "Erreur : aucun fichier Excel n'est chargé. "
            "Demandez à l'utilisateur d'uploader un fichier .xlsx via la barre latérale."
        )
    if not os.path.exists(input_file):
        return "Erreur : le fichier temporaire a expiré. Veuillez uploader à nouveau le fichier Excel."

    activities     = [a for a in (target_activities or []) if a in pipeline.AVAILABLE_ACTIVITIES]
    signalp_list   = [a for a in (run_signalp or [])       if a in pipeline.AVAILABLE_ACTIVITIES]
    esmfold_list   = [a for a in (esmfold_activities or []) if a in pipeline.AVAILABLE_ACTIVITIES]
    invalid_acts    = [a for a in (target_activities or [])  if a not in pipeline.AVAILABLE_ACTIVITIES]
    invalid_signalp = [a for a in (run_signalp or [])        if a not in pipeline.AVAILABLE_ACTIVITIES]
    invalid_esmfold = [a for a in (esmfold_activities or []) if a not in pipeline.AVAILABLE_ACTIVITIES]
    if invalid_acts or invalid_signalp or invalid_esmfold:
        ignored = invalid_acts + invalid_signalp + invalid_esmfold
        THREAD_STATE.setdefault("warnings", []).append(f"Activités inconnues ignorées : {ignored}")

    try:
        result = pipeline.run_proteogen_pipeline(
            input_file         = input_file,
            target_activities  = activities if activities else None,
            run_smiles         = run_smiles,
            run_clustering     = run_clustering,
            run_signalp        = signalp_list if signalp_list else None,
            run_xai            = run_xai,
            run_esmfold        = run_esmfold,
            esmfold_activities = esmfold_list if esmfold_list else (activities if activities else None),
            esmfold_threshold  = esmfold_threshold,
            esmfold_top_n      = esmfold_top_n,
        )
        THREAD_STATE["pipeline_result"] = result
        return result["summary"]
    except Exception as e:
        return f"Erreur lors de l'exécution du pipeline PROTEOGEN : {str(e)}"


@tool
def lancer_clustering(run_id: str = "", cutoff: float = 0.7) -> str:
    """
    Lance le clustering PepFuNN + UMAP + Treemap PAR FEUILLE sur un run PROTEOGEN existant.
    Chaque feuille '*_Peptides' est clusterisée indépendamment et obtient son propre UMAP + Treemap.
    À utiliser quand l'utilisateur veut faire du clustering hors-pipeline ou tester un autre cutoff
    sans relancer toute la pipeline.

    Args:
        run_id : nom du fichier XLSX résultat (avec ou sans .xlsx, avec ou sans préfixe
                 'PROTEOGEN_Results_'). Si vide, le dernier run de la session est utilisé.
        cutoff : seuil de similarité Tanimoto pour Butina (défaut 0.7, plage ]0,1[).
                 Plus haut = clusters plus serrés, plus bas = clusters plus larges.
    """
    if not run_id:
        last_result = THREAD_STATE.get("pipeline_result")
        if last_result and last_result.get("output_excel"):
            run_id = os.path.splitext(os.path.basename(last_result["output_excel"]))[0]
        else:
            return (
                "Erreur : aucun run_id fourni et aucun run précédent en session. "
                "Demandez à l'utilisateur le nom du fichier XLSX résultat ou de lancer la pipeline d'abord."
            )
    if not (0.0 < cutoff < 1.0):
        return f"Erreur : cutoff={cutoff} hors plage ]0,1[. Valeur typique : 0.7."

    try:
        res = pipeline.run_clustering(run_id=run_id, cutoff=cutoff)
    except Exception as e:
        return f"Erreur lors du clustering : {str(e)}"

    if res.get("error"):
        return f"Erreur clustering : {res['error']}"

    lines = [
        f"Clustering par feuille terminé sur '{res['run_id']}' (cutoff={res['cutoff']})",
        f"Séquences uniques (toutes feuilles) : {res['n_sequences']}",
        f"Clusters totaux : {res['n_clusters']}",
    ]
    clusters_per_sheet = res.get("clusters_per_sheet") or {}
    if clusters_per_sheet:
        lines.append("Détail par feuille :")
        for sheet, n_clu in clusters_per_sheet.items():
            lines.append(f"   - {sheet} : {n_clu} clusters")
    umap_sheets = res.get("umap_sheets_html") or []
    if umap_sheets:
        lines.append(f"   UMAP par feuille     : {len(umap_sheets)} fichier(s)")
    if res["treemap_sheets_html"]:
        lines.append(f"   Treemaps par feuille : {len(res['treemap_sheets_html'])} fichier(s)")
    if res["master_xlsx"]:
        lines.append(f"   Master XLSX : {res['master_xlsx']}")
    if res["warnings"]:
        lines.append("\nAvertissements :")
        for w in res["warnings"]:
            lines.append(f"  ! {w}")
    return "\n".join(lines)


@tool
def inspecter_fichier_xlsx() -> str:
    """
    Inspecte le fichier Excel chargé AVANT tout lancement de pipeline.
    À utiliser dès que l'utilisateur veut : vérifier son fichier, voir les feuilles,
    compter les peptides, savoir si le fichier est compatible, ou diagnostiquer un problème.
    Aucun argument : opère sur le fichier uploadé via la barre latérale.

    Retourne un résumé textuel :
      - nom, taille, format, nombre de feuilles
      - pour chaque feuille '*_Peptides' : nb peptides valides, nb vides, présence colonne 'Peptide'
      - autres feuilles (ignorées par la pipeline)
      - si le fichier est déjà traité (colonnes 'Peptide_*') + activités détectées
      - warnings (feuilles incompatibles, peptides manquants, etc.)
    """
    input_file = THREAD_STATE.get("file_path")
    if not input_file:
        return (
            "Erreur : aucun fichier Excel chargé. "
            "Demandez à l'utilisateur d'uploader un .xlsx via la barre latérale."
        )
    info = pipeline.get_file_info(input_file)
    if info.get("error"):
        return f"Erreur d'inspection : {info['error']}"

    lines = [
        f"📄 Fichier : {info['file_name']} ({info['file_size_mb']} Mo, .{info['format']})",
        f"📑 {info['n_sheets']} feuille(s) : {', '.join(info['sheet_names'])}",
        f"Total peptides détectés : {info['total_peptides']}",
    ]
    if info["peptide_sheets"]:
        lines.append("\nFeuilles compatibles pipeline ('*_Peptides') :")
        for ps in info["peptide_sheets"]:
            flag = "✓" if ps["has_peptide_col"] else "✗ (colonne 'Peptide' absente)"
            lines.append(
                f"  • {ps['name']} — {ps['n_peptides']} peptides "
                f"(vides : {ps['n_empty']}) — {flag}"
            )
    if info["other_sheets"]:
        lines.append("\nAutres feuilles (ignorées par la pipeline) :")
        for os_ in info["other_sheets"]:
            lines.append(f"  • {os_['name']} — {os_['n_rows']} lignes")
    if info["already_processed"]:
        acts = ", ".join(info["detected_activities"]) or "?"
        lines.append(f"\n⚠️ Fichier déjà traité — activités présentes : {acts}")
    if info["warnings"]:
        lines.append("\nAvertissements :")
        for w in info["warnings"]:
            lines.append(f"  ! {w}")
    return "\n".join(lines)


@tool
def charger_run(run_id: str) -> str:
    """
    Recharge un run PROTEOGEN précédent depuis son fichier XLSX résultat.
    Reconstruit l'état session (Excel téléchargeable, graphiques HTML, log) pour pouvoir
    réafficher les résultats, requêter le run, ou enchaîner un clustering post-pipeline
    sans relancer toute la prédiction.
    À utiliser quand l'utilisateur dit : "recharge le run X", "reprends l'analyse précédente",
    "ouvre les résultats de Y", "réaffiche le run du JJ-MM", "load run".
    Args:
        run_id : nom du fichier XLSX résultat (avec ou sans .xlsx, avec ou sans préfixe
                 'PROTEOGEN_Results_'). Doit exister dans le CWD.
    """
    if not run_id:
        return (
            "Erreur : run_id requis. Fournissez le nom du XLSX résultat "
            "(ex. 'PROTEOGEN_Results_xxx_2026-05-21_15-51-54')."
        )
    try:
        res = pipeline.load_run(run_id=run_id)
    except Exception as e:
        return f"Erreur lors du chargement du run : {str(e)}"
    if res.get("error"):
        return f"Erreur de chargement : {res['error']}"
    THREAD_STATE["pipeline_result"] = res
    acts = res["detected_activities"]
    lines = [
        f"Run rechargé : {os.path.basename(res['run_path'])}",
        f"   Peptides   : {res['n_peptides']}",
        f"   Activités prédites : {len(acts)}" + (f" ({', '.join(acts)})" if acts else ""),
        f"   Feuilles '_Peptides' : {len(res['peptide_sheets'])}",
        f"   Graphiques HTML : {len(res['html_files'])}",
    ]
    if res["log_file"]:
        lines.append(f"   Log : {os.path.basename(res['log_file'])}")
    if res["warnings"]:
        lines.append("\nAvertissements :")
        for w in res["warnings"]:
            lines.append(f"  ! {w}")
    return "\n".join(lines)


@tool
def controle_qualite_calibration(target_activities: list) -> str:
    """
    Évalue la qualité de calibration (Brier score + ECE) des modèles CNN PROTEOGEN.
    À appeler quand l'utilisateur demande la fiabilité, qualité, calibration, Brier,
    ECE, ou reliability d'un ou plusieurs modèles d'activité biologique.
    Indépendant du pipeline : peut être appelé même sans fichier Excel uploadé.
    Génère un reliability diagram HTML par activité si bins de calibration disponibles.

    Args:
        target_activities: Liste des noms exacts d'activités à évaluer. [] = toutes.
            Mêmes noms exacts que pour lancer_pipeline_proteogen.

    Returns:
        Tableau Markdown avec Brier/ECE/grade par activité + chemins reliability diagrams HTML.
    """
    requested = target_activities or []
    activities = [a for a in requested if a in pipeline.AVAILABLE_ACTIVITIES]
    invalid    = [a for a in requested if a not in pipeline.AVAILABLE_ACTIVITIES]
    try:
        results = pipeline.quality_control_all(activities if activities else None)
    except Exception as e:
        return f"Erreur contrôle qualité calibration : {e}"

    lines = [
        "| Activité | Brier | Note Brier | ECE | Note ECE | n_val | Reliability HTML |",
        "|---|---|---|---|---|---|---|",
    ]
    diagrams = []
    for r in results:
        act = r.get("activity", "?")
        cal = r.get("calibration", {}) or {}
        if cal.get("status") == "non_calculé":
            lines.append(f"| {act} | — | — | — | — | — | non calculé |")
            continue
        brier = cal.get("brier")
        ece   = cal.get("ece")
        n_val = cal.get("n_val")
        diag  = cal.get("reliability_diagram", "")
        if diag:
            diagrams.append(diag)
        brier_s = f"{brier:.4f}" if isinstance(brier, (int, float)) else "—"
        ece_s   = f"{ece:.4f}"   if isinstance(ece,   (int, float)) else "—"
        lines.append(
            f"| {act} | {brier_s} | {cal.get('brier_grade','?')} | "
            f"{ece_s} | {cal.get('ece_grade','?')} | {n_val} | "
            f"{'`'+os.path.basename(diag)+'`' if diag else '—'} |"
        )

    md = "**Contrôle qualité calibration (Brier / ECE)**\n\n" + "\n".join(lines)
    if diagrams:
        md += (
            f"\n\n{len(diagrams)} reliability diagram(s) HTML générés dans "
            f"`Graph_UniDL_Prediction_1280D/`."
        )
    if invalid:
        md += f"\n\n⚠️ Activités inconnues ignorées : {invalid}"
    md += (
        "\n\n*Brier* : erreur quadratique moyenne (0=parfait, <0.05=excellent, "
        "<0.10=bon, <0.20=modéré). *ECE* : Expected Calibration Error "
        "(<0.03=très bien calibré, <0.07=bien calibré, <0.15=modéré)."
    )
    return md


# ============================================================
# PROMPT SYSTÈME (défini au niveau module, partagé)
# ============================================================
SYSTEM_PROMPT = """
Tu es DROID (Deep learning for Residue Orchestration, Intelligence and Discovery), un assistant scientifique expert en bioinformatique et en analyse de peptides bioactifs.
Tu aides des chercheurs à analyser leurs données de peptidomique en utilisant un pipeline de Deep Learning basé sur ESM-2 (650M) et des CNN avec Monte Carlo Dropout et d'autres modules

Ton rôle :
1. Comprendre les demandes en langage naturel (français ou anglais)
2. Identifier les activités biologiques souhaitées et les mapper vers les noms exacts
3. Si l'utilisateur veut vérifier/inspecter/voir le contenu de son fichier AVANT de lancer la pipeline,
   ou en cas de doute sur la compatibilité, appeler 'inspecter_fichier_xlsx' (aucun argument).
4. Si l'utilisateur veut (re)lancer le clustering sur un run déjà existant ou tester un autre cutoff
   sans refaire toute la pipeline, appeler 'lancer_clustering' (run_id optionnel, cutoff optionnel).
5. Si l'utilisateur veut recharger / rouvrir / réafficher un run précédent (fichier XLSX résultat),
   appeler 'charger_run' avec le run_id.
6. Si l'utilisateur demande la fiabilité, qualité, calibration, Brier, ECE ou reliability d'un ou plusieurs modèles,
   appeler 'controle_qualite_calibration' avec la liste des activités ciblées (ou [] pour toutes).
7. Appeler l'outil 'lancer_pipeline_proteogen' avec les bons paramètres
8. Expliquer les résultats de manière claire, précise et scientifique

Déclencheurs de 'inspecter_fichier_xlsx' (sans argument) :
- "vérifie mon fichier", "inspecte le fichier", "que contient mon fichier", "combien de peptides",
  "quelles feuilles", "mon fichier est-il compatible", "diagnostique", "check file"
- En cas d'erreur de la pipeline pour comprendre la cause côté fichier

Déclencheurs de 'lancer_clustering' (run_id et cutoff optionnels) :
- "refaire le clustering", "clustering avec cutoff X", "essaie un autre cutoff", "recluster",
  "UMAP du run X", "treemap du dernier run", "clustering post-pipeline"
- Si l'utilisateur ne précise pas le run_id, omettre l'argument : le dernier run de session sera utilisé
- cutoff par défaut 0.7 ; valeurs typiques 0.5–0.9

Déclencheurs de 'charger_run' (run_id obligatoire) :
- "recharge le run X", "ouvre le run Y", "reprends l'analyse précédente", "réaffiche les résultats de Z",
  "load run X", "remets le run du JJ-MM-AAAA", "je veux revoir le run X"
- run_id = nom du XLSX résultat (avec ou sans extension, avec ou sans préfixe 'PROTEOGEN_Results_').
- Après chargement, l'utilisateur peut enchaîner clustering / inspection sur ce run sans relancer la pipeline.

Déclencheurs de 'controle_qualite_calibration' (target_activities optionnel) :
- "fiabilité du modèle X", "qualité Anti_Bacterien", "calibration de Cytokine", "Brier de Neuropeptide",
  "ECE pour Anti_Cancer", "reliability diagram", "est-ce que le modèle est bien calibré ?"
- target_activities=[] si l'utilisateur demande pour TOUS les modèles, sinon liste des activités citées.
- Indépendant du pipeline : pas besoin de fichier Excel uploadé.
- NE PAS appeler automatiquement après lancer_pipeline_proteogen ; uniquement sur demande explicite de l'opérateur.

Correspondances linguistiques et MACROS importantes :
- MACRO MICROBES : Si l'utilisateur demande "les microbes", "microbien", "bactos" ou "pipeline microbe" → target_activities=["Anti_Bacterien", "Quorum_Sensing", "Anti_Fongique", "Anti_Parasitique",

- "Chimiotaxie", "chimiotactisme", "chemotaxis", "ChmxTaq" → Chimiotaxie
- "Cytokine", "cytokines", "cytokin" → Cytokine
- "pénétration cellulaire", "CPP" → Pénétration_cellulaire
- "drug delivery", "livraison de médicament", "transport de médicament" → Drug_Delivery
- "Hormones", "hormone", "endocrine" → Régulation_Hormonale

- "Hémolyse", "hémolytique", "hemolysis", "hemolytic", "Activité Hémolytique" → Activité_Hémolytique
- "Cytotoxicité", "cytotoxic", "cytotox" → Cytotoxicité
- "Neurotox", "neurotoxicité", "Neurotoxique" → Neurotoxicité
- "Allergene", "allergénicité", "allergenic", "allergy", "allergie" → Allergène
- "toxicité", "toxic", "tox" → Toxicité

- "quorum sensing", "QS" → Quorum_Sensing
- "AV", "activité antivirale", "antiviral" → Anti_Virale
- "anti-bactérien", "antibactérien", "antimicrobien", "AB" → Anti_Bacterien
- "anti-fongique", "antifongique", "AF" → Anti_Fongique
- "Anti-parasitique", "antiparasitique", "AP" → Anti_Parasitique
- "AMRSA", "anti-résistance aux antibiotiques", "anti resistance", "Staph résitant" → Anti_MRSA
- "Anti-biofilm", "antibiofilm", "ABF" → Anti_Biofilm

- "anti-hypertension", "antihypertension", "ACE", "AHT" → Anti_Hyper_Tension
- "DPPIV, "DDPIV", "DPP-IV", "DPP4", "dipeptidyl peptidase IV" → Dipeptidyl_peptidase_IV
- "AIP", "anti-inflammatoire", "anti inflammatoire" → Anti_Inflammatoire
- "anti-âge", "anti age", "anti-age", "antiaging", "anti-aging", "AA" → Anti_Age
- "anti-amnésique", "anti amnésique", "anti-amnesique", "antiamnesique", "anti-amnesic", "AAm" → Anti_Amnésique
- "neuropeptide", "neuropep" → Neuropeptide
- "anti-cancer", "anticancer" → Anti_Cancer
- "anti-tumeur", "antitumor", "antitumoral" → Anti_Tumeur
- "anti-oxydant", "antioxydant", "AO" → Anti_Oxydant
- "opioïde", "opioide", "opioid", "morphinique" → Opioïde
- "umami", "goût umami", "savoureux" → Umami

- "clustering", "UMAP", "t-SNE", "clusters", "visualisation chimique" → run_clustering=True
- "pas de clustering", "pas UMAP", "sans clusters" → run_clustering=False
- "SignalP", "peptide signal", "signal peptide" → run_signalp=[liste]
- "sans XAI", "pas de XAI", "ignorer XAI", "rapide", "vite" → run_xai=False
- "Avec XAI", "lancer XAI", "XAI activé" "Faire XAI" → run_xai=True
- "Avec ESM-Fold", "lancer ESM-Fold", "ESM-Fold activé" "Faire ESM-Fold", "ESMFOLD", "ESM FOLD",
  "structure 3D", "fold 3D", "prédiction structure" → run_esmfold=True
- "Pas d'ESM-Fold", "ignorer ESM-Fold", "sans ESM-Fold", "ESM-Fold désactivé" → run_esmfold=False
- ESM-Fold optionnel : esmfold_activities (sous-liste), esmfold_threshold (proba seuil, défaut 0.95),
  esmfold_top_n (top N peptides par activité, défaut 30). Ne passer que si l'utilisateur les mentionne.

Règles pour run_signalp :
- run_signalp EST UNE LISTE, même pour une seule activité : ["Neuropeptide"]
- "SignalP sur neuropeptides et anti-cancer" → run_signalp=["Neuropeptide", "Anti_Cancer"]

Activités disponibles (28 noms EXACTS — utiliser à l'identique) :
Chimiotaxie, Cytokine, Pénétration_cellulaire, Drug_Delivery, Régulation_Hormonale,
Activité_Hémolytique, Cytotoxicité, Neurotoxicité, Toxicité, Allergène,
Quorum_Sensing, Anti_Virale, Anti_Bacterien, Anti_Fongique, Anti_Parasitique, Anti_MRSA, Anti_Biofilm,
Anti_Hyper_Tension, Dipeptidyl_peptidase_IV, Anti_Inflammatoire, Anti_Age, Anti_Amnésique,
Neuropeptide, Anti_Cancer, Anti_Tumeur, Anti_Oxydant, Opioïde, Umami

Informations techniques :
- run_xai=False par défaut ; activer seulement si explicitement demandé
- run_smiles=False par défaut ; activer seulement si l'utilisateur demande les colonnes SMILES (peut être lent)
- run_clustering=False par défaut ; activer seulement si l'utilisateur mentionne le clustering, UMAP
- run_signalp=None par défaut ; activer seulement si l'utilisateur mentionne SignalP suivi d'activités spécifiques
- run_esmfold=False par défaut ; activer seulement si l'utilisateur mentionne ESM-Fold ou prédiction de structure 3D
- Les visualisations HTML sont dans Graph_UniDL_Prediction_1280D et Output_Cluster_1280D
"""


# ============================================================
# LLM AVEC OUTILS (mis en cache)
# ============================================================
@st.cache_resource
def get_llm_with_tools(model_name: str, base_url: str):
    llm = ChatOllama(model=model_name, base_url=base_url, temperature=0, request_timeout=360, model_kwargs={"keep_alive":0})
    return llm.bind_tools([
        lancer_pipeline_proteogen,
        inspecter_fichier_xlsx,
        lancer_clustering,
        charger_run,
        controle_qualite_calibration,
    ])
# ============================================================
# ZONE D'AFFICHAGE DES RÉSULTATS PIPELINE
# ============================================================
def display_pipeline_results(result: dict):
    """Affiche le fichier Excel téléchargeable et les graphiques HTML."""
    st.success("✅ Pipeline terminé !")
    st.markdown(f"**Résumé :** {result['summary']}")
    dashboard_path = result.get("dashboard_html")
    has_dashboard = bool(dashboard_path) and os.path.exists(dashboard_path)
    cols = st.columns([1, 1, 1]) if has_dashboard else st.columns([1, 1])

    with cols[0]:
        if os.path.exists(result["output_excel"]):
            with open(result["output_excel"], "rb") as f:
                st.download_button(
                    label="⬇️ Télécharger les résultats Excel",
                    data=f,
                    file_name=os.path.basename(result["output_excel"]),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        else:
            st.warning(f"Fichier Excel introuvable : {result['output_excel']}")

    with cols[1]:
        if os.path.exists(result["log_file"]):
            with open(result["log_file"], "r", encoding="utf-8") as f:
                log_content = f.read()
            st.download_button(
                label="📋 Télécharger le log",
                data=log_content,
                file_name=os.path.basename(result["log_file"]),
                mime="text/plain",
                use_container_width=True,
            )

    if has_dashboard:
        with cols[2]:
            with open(dashboard_path, "rb") as f:
                st.download_button(
                    label="🧬 Télécharger le dashboard HTML",
                    data=f,
                    file_name=os.path.basename(dashboard_path),
                    mime="text/html",
                    use_container_width=True,
                    help="Ouvrir dans un navigateur pour visualisation plein écran (tableau filtrable par activité bio).",
                )

    # Visualisations interactives
    if result["html_files"]:
        st.markdown(f"### 📊 Visualisations ({len(result['html_files'])} graphique(s))")
        tab_labels = [os.path.basename(p) for p in result["html_files"]]
        tabs = st.tabs(tab_labels)
        for tab, html_path in zip(tabs, result["html_files"]):
            with tab:
                try:
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    components.html(html_content, height=520, scrolling=True)
                except Exception as e:
                    st.error(f"Impossible d'afficher {os.path.basename(html_path)} : {e}")
    else:
        st.info("Aucun graphique HTML généré (normal si XAI et clustering désactivés).")


# ============================================================
# INTERFACE PRINCIPALE — CHAT
# ============================================================
restyle_droid.main_header(
    "PROTEOGEN",
    "Agent d'analyse peptidique · ESM-2 + CNN + Monte Carlo Dropout · 100 % local via Ollama",
)

# Initialisation de l'état de session
if "messages"       not in st.session_state:
    st.session_state["messages"]      = []
if "chat_history"   not in st.session_state:
    st.session_state["chat_history"]  = []
if "pipeline_result" not in st.session_state:
    st.session_state["pipeline_result"] = None

# ── Recovery après tab suspendu : restaurer depuis THREAD_STATE ─────────────
# THREAD_STATE est module-level (persistant aux reruns), session_state lié à la
# WebSocket. Si tab navigateur suspendu trop longtemps, session_state est resetté
# mais THREAD_STATE garde résultat ou statut du thread. Restauration ici évite
# perte sortie analyse / écran d'accueil alors que pipeline tournait.
def _recover_orphan_thread_state():
    fresh = (not st.session_state["messages"]
             and st.session_state["pipeline_result"] is None)
    has_result = THREAD_STATE.get("pipeline_result") is not None
    has_answer = bool(THREAD_STATE.get("agent_answer"))
    is_running = not THREAD_STATE.get("is_done", True)
    if not (fresh and (has_result or has_answer or is_running)):
        return
    if has_result:
        st.session_state["pipeline_result"] = THREAD_STATE["pipeline_result"]
    if has_answer:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "🔄 *Session restaurée après inactivité.*\n\n"
                       + str(THREAD_STATE["agent_answer"]),
        })
    elif is_running:
        label = THREAD_STATE.get("progress_label", "...")
        prog  = int(round(float(THREAD_STATE.get("progress", 0.0)) * 100))
        st.session_state["messages"].append({
            "role": "assistant",
            "content": (f"⏳ *Pipeline en cours côté serveur* : `{label}` "
                        f"({prog}%). Rafraîchissez dans quelques secondes "
                        f"pour récupérer la sortie."),
        })
    elif has_result:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "🔄 *Session restaurée — résultat précédent réaffiché.*",
        })

_recover_orphan_thread_state()
# ── Keepalive : empêche navigateur de throttler/suspendre le tab ────────────
# Wake Lock API + AudioContext silencieux + ping HTTP périodique. Évite que
# Chrome/Firefox déchargent le tab après inactivité prolongée et coupent la
# WebSocket Streamlit (qui resetterait session_state).
components.html("""
<script>
(async () => {
  if ('wakeLock' in navigator) {
    let lock = null;
    const acquire = async () => {
      try { lock = await navigator.wakeLock.request('screen'); }
      catch(e) { console.warn('WakeLock refusé:', e); }
    };
    await acquire();
    document.addEventListener('visibilitychange', async () => {
      if (document.visibilityState === 'visible' && (!lock || lock.released)) {
        await acquire();
      }
    });
  }
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    gain.gain.value = 0;
    osc.connect(gain).connect(ctx.destination);
    osc.start();
  } catch(e) { console.warn('AudioContext indisponible:', e); }
  setInterval(() => {
    fetch(window.location.pathname + '?_keepalive=' + Date.now(),
          { method: 'HEAD', cache: 'no-store' }).catch(() => {});
  }, 25000);
})();
</script>
""", height=0)

# Message d'accueil au premier lancement
if not st.session_state["messages"]:
    welcome = (
        "Bonjour ! Je suis **DROID**, votre assistant d'analyse peptidomique.\n\n"
        "Pour commencer :\n"
        "1. **Uploadez** votre fichier Excel (`.xlsx`) dans la barre latérale\n"
        "2. **Décrivez** l'analyse ou le tool souhaitée en langage naturel\n\n"
        "Exemple : *\"À partir de ce fichier, fais les prédictions pour Anti-bactérien "
        "et Quorum Sensing, et lance le clustering.\"*\n\n"
        "Exemple: *\"Analyse mon fichier et indique moi ce qu'il contient\"*"
    )
    st.session_state["messages"].append({"role": "assistant", "content": welcome})

# ── Affichage de l'historique du chat ──────────────────────────────────────
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Résultats pipeline (persistants jusqu'à la prochaine analyse) ───────────
if st.session_state["pipeline_result"]:
    with st.container(border=True):
        display_pipeline_results(st.session_state["pipeline_result"])

# ── Interpréteur DeepSeek — clusters post-pipeline ─────────────────────────
if st.session_state["pipeline_result"] and st.session_state["pipeline_result"].get("output_excel"):
    with st.container(border=True):
        st.markdown("### 🧠 Interpréteur DeepSeek-R1 — Clusters")
        st.caption(
            "Analyse approfondie d'un cluster."
        )
        _run_basename = os.path.basename(st.session_state["pipeline_result"]["output_excel"])
        _base_id_ds = os.path.splitext(_run_basename)[0].replace("PROTEOGEN_Results_", "", 1)
        _clusters_info = deepseek_interp.list_clusters(_base_id_ds)
        if _clusters_info.get("error"):
            st.info(f"⚠️ {_clusters_info['error']}")
        else:
            _sheets_dispo = list(_clusters_info["by_sheet"].keys())
            if not _sheets_dispo:
                st.info("Aucun cluster détecté dans le master XLSX.")
            else:
                _col_s, _col_c = st.columns([1, 1])
                with _col_s:
                    _sheet_sel = st.selectbox("Feuille", _sheets_dispo, key="deepseek_sheet")
                with _col_c:
                    _clusters_for_sheet = _clusters_info["by_sheet"].get(_sheet_sel, [])
                    _cluster_sel = st.selectbox("Cluster", _clusters_for_sheet, key="deepseek_cluster")
                _question_user = st.text_area(
                    "Question (optionnel)",
                    value="",
                    placeholder="Laisser vide pour analyse standard (composition, activités, mécanismes, pistes).",
                    key="deepseek_question",
                    height=80,
                )
                if st.button("🧠 Interpréter avec DeepSeek-R1", use_container_width=True, key="deepseek_run_btn"):
                    st.markdown("---")
                    st.markdown(f"**DeepSeek interprète** — feuille `{_sheet_sel}`, cluster `{_cluster_sel}`")
                    _out_container = st.empty()
                    _full_text = []
                    with st.spinner("DeepSeek en cours (unload QWEN, load DeepSeek-R1, stream)..."):
                        for _chunk in deepseek_interp.explain_cluster_stream(
                            run_id=_run_basename,
                            sheet=_sheet_sel,
                            cluster_id=_cluster_sel,
                            user_question=(_question_user.strip() or None),
                            base_url=ollama_base_url,
                        ):
                            _full_text.append(_chunk)
                            _out_container.markdown("".join(_full_text))
                    pipeline.send_ntfy_notification(
                        message=f"DeepSeek terminé — cluster {_cluster_sel} de {_sheet_sel}",
                        title="PROTEOGEN — DeepSeek",
                        priority="default",
                        tags="brain",
                    )

# ── Zone de saisie ──────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Décrivez votre demande ou besoin en outils"
):
    # Afficher le message utilisateur
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Pas de blocage frontal : certains outils (controle_qualite_calibration, charger_run)
    # n'exigent pas de fichier uploadé. Chaque @tool valide ses pré-requis en interne
    # et renvoie un message d'erreur clair si nécessaire.
    if True:
        #Lancement de l'agent
        with st.chat_message("assistant"):
            st_log_container = st.empty() # Conteneur vide pour notre "terminal" en direct

            #Préparation du statut pour le Thread
            THREAD_STATE["file_path"] = st.session_state.get("uploaded_file_path")
            THREAD_STATE["pipeline_result"] = None
            THREAD_STATE["agent_answer"] = None
            THREAD_STATE["error"] = None
            THREAD_STATE["is_done"] = False
            THREAD_STATE["progress"] = 0.0
            THREAD_STATE["progress_label"] = "Initialisation..."
            THREAD_STATE["total_activities"] = 0

            #Initialisation de la mémoire des logs
            if "live_logs" not in st.session_state:
                st.session_state["live_logs"] = []
            st.session_state["live_logs"].clear()

            #Logger — only attach to our own logger, not the root (évite le bruit HTTP)
            ui_handler = StreamlitUIHandler(st.session_state["live_logs"], THREAD_STATE)
            ui_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%H:%M:%S"))
            ui_handler.addFilter(PipelineLogFilter())
            worker_logger = logging.getLogger("proteogen.worker")
            worker_logger.addHandler(ui_handler)
            worker_logger.setLevel(logging.INFO)
            worker_logger.propagate = False

            import ast # (À mettre au début du fichier avec tes autres imports)

            KNOWN_TOOLS = {
                "lancer_pipeline_proteogen",
                "inspecter_fichier_xlsx",
                "lancer_clustering",
                "charger_run",
                "controle_qualite_calibration",
            }

            def _extract_tool_call_from_any_json(text: str):
                """
                Extrait un tool call émis en JSON brut dans le content du LLM.
                Retourne (name, args) si trouvé ET name ∈ KNOWN_TOOLS, sinon (None, None).
                Couvre Qwen2.5/Llama3 qui émettent parfois le tool call en text au lieu
                du champ structuré tool_calls.
                """
                try:
                    start_idx = text.find('{')
                    end_idx = text.rfind('}')
                    if start_idx == -1 or end_idx == -1:
                        return None, None
                    json_str = text[start_idx:end_idx+1]
                    json_str = json_str.replace("False", "false").replace("True", "true").replace("None", "null")
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        try:
                            data = ast.literal_eval(json_str)
                        except (ValueError, SyntaxError):
                            return None, None
                    name = data.get("name") or data.get("tool") or data.get("function", "")
                    if isinstance(name, dict):
                        name = name.get("name", "")
                    if name not in KNOWN_TOOLS:
                        return None, None
                    args = data.get("parameters", data.get("arguments", data.get("args", {})))
                    if isinstance(args, str):
                        args = args.replace("False", "false").replace("True", "true").replace("None", "null")
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = ast.literal_eval(args)
                    return name, (args if isinstance(args, dict) else {})
                except Exception as e:
                    logging.getLogger("proteogen.worker").warning(f"[WORKER] Erreur parsing JSON brut: {e}")
                    return None, None

            # Fonction de travail (thread séparé) — boucle tool-calling manuelle
            def worker_agent(prompt_in, history_in):
                log = logging.getLogger("proteogen.worker")
                pipeline.clear_stop()
                try:
                    THREAD_STATE["progress"] = 0.05
                    THREAD_STATE["progress_label"] = "Démarrage de l'agent..."
                    log.info("[WORKER] Démarrage du thread agent")
                    llm_tools = get_llm_with_tools(ollama_model, ollama_base_url)

                    messages = (
                        [SystemMessage(content=SYSTEM_PROMPT)]
                        + list(history_in)
                        + [HumanMessage(content=prompt_in)]
                    )

                    # Étape 1 — appel LLM avec outils disponibles
                    THREAD_STATE["progress"] = 0.10
                    THREAD_STATE["progress_label"] = "Analyse LLM de la requête..."
                    log.info("[WORKER] Étape 1 — appel LLM (avec outils)...")
                    ai_response = llm_tools.invoke(messages)
                    THREAD_STATE["progress"] = 0.40
                    THREAD_STATE["progress_label"] = "Interprétation de la réponse LLM..."
                    log.info(f"[WORKER] tool_calls structurés : {ai_response.tool_calls}")
                    raw_content = str(ai_response.content) if ai_response.content else ""
                    log.info(f"[WORKER] content brut : {raw_content}")

                    tool_calls = list(ai_response.tool_calls or [])

                    if not tool_calls and raw_content:
                        name, args = _extract_tool_call_from_any_json(raw_content)
                        if name and args is not None:
                            if name == "lancer_pipeline_proteogen" and "target_activities" not in args:
                                args["target_activities"] = []
                            if name == "controle_qualite_calibration" and "target_activities" not in args:
                                args["target_activities"] = []
                            log.info(f"[WORKER] Fallback JSON → tool={name}, args={args}")
                            tool_calls = [{"name": name, "args": args, "id": "fallback_0"}]
                        else:
                            # Détection nom de tool en texte brut (dernier recours)
                            for tname in KNOWN_TOOLS:
                                if tname in raw_content:
                                    log.info(f"[WORKER] Nom '{tname}' détecté sans args lisibles → défauts")
                                    if tname == "lancer_pipeline_proteogen":
                                        default_args = {"target_activities": [], "run_clustering": False, "run_signalp": [], "run_xai": False}
                                    elif tname == "controle_qualite_calibration":
                                        default_args = {"target_activities": []}
                                    elif tname == "lancer_clustering":
                                        default_args = {"run_id": "", "cutoff": 0.7}
                                    elif tname == "charger_run":
                                        default_args = {"run_id": ""}
                                    else:
                                        default_args = {}
                                    tool_calls = [{"name": tname, "args": default_args, "id": "fallback_0"}]
                                    break
                            else:
                                log.info("[WORKER] Aucun tool call détecté → réponse directe")

                    if tool_calls:
                        THREAD_STATE["progress"] = 0.50
                        THREAD_STATE["progress_label"] = "Exécution du pipeline PROTEOGEN..."
                        log.info(f"[WORKER] Étape 2 — exécution de {len(tool_calls)} outil(s)...")
                        tool_results = []
                        for tc in tool_calls:
                            if tc["name"] == "lancer_pipeline_proteogen":
                                acts = tc["args"].get("target_activities") or []
                                THREAD_STATE["total_activities"] = len(acts) if acts else len(pipeline.AVAILABLE_ACTIVITIES)
                                tool_result = lancer_pipeline_proteogen.invoke(tc["args"])
                                tool_results.append(str(tool_result))
                                log.info(f"[WORKER] Résultat : {str(tool_result)[:300]}")
                            elif tc["name"] == "inspecter_fichier_xlsx":
                                THREAD_STATE["progress_label"] = "Inspection du fichier Excel..."
                                tool_result = inspecter_fichier_xlsx.invoke(tc.get("args", {}))
                                tool_results.append(str(tool_result))
                                log.info(f"[WORKER] Inspection : {str(tool_result)[:300]}")
                            elif tc["name"] == "lancer_clustering":
                                THREAD_STATE["progress_label"] = "Clustering PepFuNN en cours..."
                                tool_result = lancer_clustering.invoke(tc.get("args", {}))
                                tool_results.append(str(tool_result))
                                log.info(f"[WORKER] Clustering : {str(tool_result)[:300]}")
                            elif tc["name"] == "charger_run":
                                THREAD_STATE["progress_label"] = "Chargement du run précédent..."
                                tool_result = charger_run.invoke(tc.get("args", {}))
                                tool_results.append(str(tool_result))
                                log.info(f"[WORKER] Chargement run : {str(tool_result)[:300]}")
                            elif tc["name"] == "controle_qualite_calibration":
                                THREAD_STATE["progress_label"] = "Contrôle qualité calibration (Brier/ECE)..."
                                tool_result = controle_qualite_calibration.invoke(tc.get("args", {}))
                                tool_results.append(str(tool_result))
                                log.info(f"[WORKER] Calibration : {str(tool_result)[:300]}")
                            else:
                                log.warning(f"[WORKER] Tool inconnu reçu du LLM : {tc['name']}")
                                tool_results.append(f"⚠️ Outil non géré côté worker : `{tc['name']}`")

                        THREAD_STATE["progress"] = 0.92
                        THREAD_STATE["progress_label"] = "Génération de la réponse finale..."
                        log.info("[WORKER] Étape 3 — réponse finale prête.")
                        summary = "\n\n".join(tool_results) if tool_results else "*(aucun résultat retourné par les outils)*"
                        called_names = [tc["name"] for tc in tool_calls]
                        header = (
                            "✅ **Pipeline PROTEOGEN terminé.**"
                            if "lancer_pipeline_proteogen" in called_names
                            else f"✅ **Outil(s) exécuté(s)** : {', '.join(called_names)}"
                        )
                        THREAD_STATE["agent_answer"] = f"{header}\n\n{summary}"
                    else:
                        THREAD_STATE["agent_answer"] = raw_content

                except InterruptedError as e:
                    log.warning(f"[WORKER] Pipeline interrompu : {e}")
                    THREAD_STATE["agent_answer"] = "⚠️ Pipeline interrompu."
                    pipeline.send_ntfy_notification(
                        message=f"Pipeline interrompu par l'opérateur : {e}",
                        title="PROTEOGEN — interrompu",
                        priority="default", tags="warning",
                    )
                except Exception as e:
                    log.error(f"[WORKER] Exception : {e}")
                    THREAD_STATE["error"] = str(e)
                    pipeline.send_ntfy_notification(
                        message=f"worker_agent crash {type(e).__name__} : {e}",
                        title="PROTEOGEN — ERREUR agent",
                        priority="urgent", tags="rotating_light,x",
                    )
                finally:
                    THREAD_STATE["progress"] = 1.0
                    THREAD_STATE["progress_label"] = "Terminé !"
                    log.info("[WORKER] Thread terminé.")
                    THREAD_STATE["is_done"] = True

            # Lancement du Thread indépendant
            t = threading.Thread(
                target=worker_agent,
                args=(prompt, st.session_state["chat_history"]),
                daemon=True,  # killed automatically when the main process exits (Ctrl+C)
            )
            add_script_run_ctx(t) # Vital : autorise le thread à lire l'état de Streamlit
            t.start()

            # Boucle d'écoute pour l'interface graphique
            t_start = time.time()
            _root_handler_added = False
            while not THREAD_STATE["is_done"]:
                # Attacher au root logger dès que basicConfig l'a configuré (pipeline)
                if not _root_handler_added and logging.root.handlers:
                    logging.root.addHandler(ui_handler)
                    _root_handler_added = True

                elapsed = int(time.time() - t_start)
                with st_log_container.container():
                    st.markdown(f" **DROID analyse votre requête...** `{elapsed}s`")
                    st.progress(
                        THREAD_STATE.get("progress", 0.0),
                        text=THREAD_STATE.get("progress_label", ""),
                    )
                    recent_logs = st.session_state["live_logs"][-20:]
                    log_text = "\n".join(recent_logs) if recent_logs else "En attente du modèle Ollama (CPU — peut prendre 30–60 s)..."
                    st.code(log_text, language="text")
                time.sleep(1)

            # Nettoyage (désactiver les loggers)
            worker_logger.removeHandler(ui_handler)
            if _root_handler_added:
                logging.root.removeHandler(ui_handler)

            # Affichage de la réponse finale
            st_log_container.empty()
            
            if THREAD_STATE["error"]:
                answer = (
                    f"**Erreur de communication avec Ollama ou le Pipeline** : `{THREAD_STATE['error']}`\n\n"
                    f"Vérifiez :\n"
                    f"- qu'Ollama est bien lancé (`ollama serve`)\n"
                    f"- que le modèle `{ollama_model}` est installé\n"
                    f"- que l'URL `{ollama_base_url}` est correcte"
                )
                st.error(answer)
            else:
                answer = THREAD_STATE["agent_answer"]
                st.markdown(answer)

            # Sauvegarde des résultats générés par le pipeline
            if THREAD_STATE["pipeline_result"]:
                st.session_state["pipeline_result"] = THREAD_STATE["pipeline_result"]

        # Mise à jour de l'historique de conversation
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.session_state["chat_history"].append(HumanMessage(content=prompt))
        st.session_state["chat_history"].append(AIMessage(content=answer))

        # Re-rendu final pour faire apparaître les graphiques et le bouton de téléchargement Excel
        if st.session_state["pipeline_result"]:
            st.rerun()
