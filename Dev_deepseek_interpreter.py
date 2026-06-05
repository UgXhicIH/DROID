"""
PROTEOGEN — Module interpréteur DeepSeek
Explique résultats post-pipeline (clusters principalement).
Switch Ollama : unload qwen2.5:14b -> load proteogen-qwen3-32b:q5 -> stream -> unload.
Modèle = Qwen3-32B Q5 reTemplaté (cf Modelfile_qwen3_proteogen) ; raisonnement coupé (/no_think).
Appelé exclusivement depuis Streamlit (Dev_app_token.py), bypass agent QWEN.
"""
import os
import re
import json
import glob
import logging
import urllib.request
import urllib.error
from typing import Iterator, Optional

import pandas as pd  # type: ignore

DIR_CLUSTER = "Output_Cluster_1280D"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
QWEN_MODEL = "qwen2.5:14b"
DEEPSEEK_MODEL = "proteogen-qwen3-32b:q5"  # ChatML + stop corrigés (cf Modelfile_qwen3_proteogen)
HTTP_TIMEOUT = 600
UNLOAD_TIMEOUT = 30


def _ollama_post(path: str, payload: dict, base_url: str, timeout: int, stream: bool = False):
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _unload_ollama_model(model: str, base_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Unload modèle Ollama via keep_alive=0 (libère VRAM)."""
    try:
        with _ollama_post(
            "/api/generate",
            {"model": model, "prompt": "", "keep_alive": 0},
            base_url,
            UNLOAD_TIMEOUT,
        ) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        logging.warning(f"Unload {model} échoué : {e}")
        return False


def _resolve_master_xlsx(base_id: str) -> Optional[str]:
    """Scanne DIR_CLUSTER pour pattern Self_Ids_Sequence_PepFuNN_{base_id}.xlsx, prend le plus récent.
    Master XLSX nommé sans suffixe date (graphs_id = input_file basename), alors que run_id
    issu de output_excel a un suffixe _YYYY-MM-DD_HH-MM-SS. Retry sans suffixe si miss.
    """
    if not base_id:
        return None
    candidates = [base_id]
    stripped = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", "", base_id)
    if stripped != base_id:
        candidates.append(stripped)
    for cand in candidates:
        pattern = os.path.join(DIR_CLUSTER, f"Self_Ids_Sequence_PepFuNN_{cand}.xlsx")
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    return None


def list_clusters(base_id: str) -> dict:
    """Liste sheets + clusters disponibles depuis master_xlsx. Pour peupler dropdowns UI."""
    master = _resolve_master_xlsx(base_id)
    if not master:
        return {"error": f"master_xlsx introuvable pour '{base_id}'. Lancer clustering d'abord.", "master_xlsx": None, "by_sheet": {}}
    try:
        df = pd.read_excel(master)
    except Exception as e:
        return {"error": f"Lecture master_xlsx échouée : {e}", "master_xlsx": master, "by_sheet": {}}
    if "Sheet" not in df.columns or "Cluster_Sheet" not in df.columns:
        return {"error": "Colonnes 'Sheet'/'Cluster_Sheet' absentes.", "master_xlsx": master, "by_sheet": {}}
    by_sheet = {}
    for sheet in sorted(df["Sheet"].dropna().unique().tolist()):
        clusters = sorted(df[df["Sheet"] == sheet]["Cluster_Sheet"].dropna().unique().tolist())
        by_sheet[sheet] = clusters
    return {"error": None, "master_xlsx": master, "by_sheet": by_sheet}


def _build_cluster_context(master_xlsx: str, sheet: str, cluster_id: str, max_peptides: int = 25) -> dict:
    """Extrait contexte cluster (composition, activités moyennes, top peptides)."""
    try:
        df = pd.read_excel(master_xlsx)
    except Exception as e:
        return {"error": f"Lecture XLSX échouée : {e}"}
    if "Sheet" not in df.columns or "Cluster_Sheet" not in df.columns:
        return {"error": "Colonnes 'Sheet'/'Cluster_Sheet' absentes."}
    df_filt = df[(df["Sheet"] == sheet) & (df["Cluster_Sheet"].astype(str) == str(cluster_id))]
    if df_filt.empty:
        return {"error": f"Cluster '{cluster_id}' absent de feuille '{sheet}'."}
    pred_cols = [c for c in df_filt.columns if str(c).startswith("Peptide_") and not str(c).endswith("_raw")]
    activity_means = {}
    if pred_cols:
        means = df_filt[pred_cols].mean(numeric_only=True).sort_values(ascending=False).round(4)
        activity_means = {k.replace("Peptide_", ""): float(v) for k, v in means.items()}

    sort_col = "Proba_Max" if "Proba_Max" in df_filt.columns else None
    top_peps = df_filt.nlargest(max_peptides, sort_col) if sort_col else df_filt.head(max_peptides)
    peptides_list = []
    for _, row in top_peps.iterrows():
        peptides_list.append({
            "peptide": str(row.get("Peptide", "")),
            "seq_clean": str(row.get("Seq_Clean", "")),
            "activité_dominante": str(row.get("Activité_Dominante", "")),
            "proba_max": float(row["Proba_Max"]) if "Proba_Max" in row and pd.notna(row["Proba_Max"]) else None,
        })
    seq_lengths = df_filt["Seq_Clean"].dropna().astype(str).map(len).tolist() if "Seq_Clean" in df_filt.columns else []
    length_stats = {}
    if seq_lengths:
        length_stats = {
            "min": int(min(seq_lengths)),
            "max": int(max(seq_lengths)),
            "mean": round(sum(seq_lengths) / len(seq_lengths), 2),
        }
    return {
        "sheet": sheet,
        "cluster_id": str(cluster_id),
        "n_peptides": int(len(df_filt)),
        "longueurs": length_stats,
        "activités_moyennes_top": dict(list(activity_means.items())[:10]),
        "peptides_top": peptides_list,
    }


_FORMAT_RULES = (
    "\nFormat de réponse IMPÉRATIF (rendu mobile DROID, espace réduit) :\n"
    "- Prose markdown structurée en sections avec titres `##`.\n"
    "- Toute donnée structurée (listes de peptides, activités, métriques, comparaisons) : "
    "TABLEAU markdown compact (`| col | col |`), jamais en liste à puces verbeuse.\n"
    "- INTERDIT : sortie JSON, bloc ```json```, accolades de structure de données.\n"
    "- Réponds en français, concis.\n"
)


def _build_prompt(ctx: dict, user_question: Optional[str]) -> str:
    intro = (
        "Tu es un expert en peptidomique et bioinformatique. "
        "Tu interprètes un cluster de peptides bioactifs issu du pipeline PROTEOGEN "
        "(ESM-2 650M + CNN + Monte Carlo Dropout + clustering PepFuNN/Butina).\n\n"
        f"Données du cluster :\n```json\n{json.dumps(ctx, ensure_ascii=False, indent=2)}\n```\n\n"
    )
    if user_question:
        return intro + f"Question opérateur :\n{user_question}\n" + _FORMAT_RULES
    return intro + (
        "Analyse demandée :\n"
        "1. Composition du cluster (taille, profils dominants, longueurs).\n"
        "2. Activités biologiques sur-représentées et cohérence inter-activités.\n"
        "3. Peptides clés (proba haute, multi-actifs, motifs apparents).\n"
        "4. Hypothèses mécanistiques + pistes expérimentales.\n"
    ) + _FORMAT_RULES


def _strip_think(chunks: Iterator[str]) -> Iterator[str]:
    """Retire les blocs <think>...</think> du flux Qwen3 (gère les balises coupées entre chunks)."""
    open_tag, close_tag = "<think>", "</think>"
    buf = ""
    in_think = False
    for chunk in chunks:
        buf += chunk
        progressed = True
        while progressed:
            progressed = False
            if in_think:
                idx = buf.find(close_tag)
                if idx != -1:
                    buf = buf[idx + len(close_tag):]
                    in_think = False
                    progressed = True
                elif len(buf) >= len(close_tag):
                    buf = buf[-(len(close_tag) - 1):]  # ne garde qu'une fin de balise partielle
            else:
                idx = buf.find(open_tag)
                if idx != -1:
                    if idx > 0:
                        yield buf[:idx]
                    buf = buf[idx + len(open_tag):]
                    in_think = True
                    progressed = True
                elif len(buf) >= len(open_tag):
                    emit = buf[:-(len(open_tag) - 1)]
                    if emit:
                        yield emit
                    buf = buf[-(len(open_tag) - 1):]
    if buf and not in_think:
        yield buf


def explain_cluster_stream(
    run_id: str,
    sheet: str,
    cluster_id: str,
    user_question: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    reload_qwen_after: bool = True,
) -> Iterator[str]:
    """
    Génère stream de tokens DeepSeek interprétant un cluster.
    Yield tokens texte au fur et à mesure. En cas d'erreur, yield message puis stop.
    """
    base_id = run_id.replace("PROTEOGEN_Results_", "").replace(".xlsx", "")
    master = _resolve_master_xlsx(base_id)
    if not master:
        yield f"**Erreur** : master_xlsx introuvable pour run `{run_id}`. Lancer clustering d'abord."
        return
    ctx = _build_cluster_context(master, sheet, cluster_id)
    if "error" in ctx:
        yield f"**Erreur contexte** : {ctx['error']}"
        return
    prompt = _build_prompt(ctx, user_question)
    _unload_ollama_model(QWEN_MODEL, base_url)
    payload = {
        "model": DEEPSEEK_MODEL,
        # /no_think : désactive le raisonnement Qwen3 (sortie directe, plus rapide sur mobile)
        "messages": [{"role": "user", "content": prompt + "\n\n/no_think"}],
        "stream": True,
        "keep_alive": "5m",
        "options": {
            "num_ctx": 8192,
            "num_predict": 4096,  # borne dure anti-boucle : coupe toute génération sans fin
            "stop": ["<|im_start|>", "<|im_end|>"],
        },
    }

    def _raw_chunks() -> Iterator[str]:
        try:
            with _ollama_post("/api/chat", payload, base_url, HTTP_TIMEOUT) as resp:
                for raw_line in resp:
                    if not raw_line:
                        continue
                    try:
                        obj = json.loads(raw_line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    chunk = obj.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if obj.get("done"):
                        break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            yield f"\n\n**Erreur DeepSeek** : {e}"

    try:
        yield from _strip_think(_raw_chunks())
    finally:
        _unload_ollama_model(DEEPSEEK_MODEL, base_url)
        if reload_qwen_after:
            try:
                _ollama_post(
                    "/api/generate",
                    {"model": QWEN_MODEL, "prompt": "", "keep_alive": "5m"},
                    base_url,
                    UNLOAD_TIMEOUT,
                ).close()
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
                pass
