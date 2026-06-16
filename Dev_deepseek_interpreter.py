"""
PROTEOGEN — Module interpréteur DeepSeek
Explique résultats post-pipeline. Trois niveaux d'analyse :
  - explain_cluster_stream     : un cluster (master_xlsx PepFuNN).
  - explain_run_global_stream  : synthèse multi-activités d'un run (XLSX résultats, hors clustering).
  - compare_clusters_stream    : comparaison de ≥2 clusters (même feuille ou feuilles différentes).
Switch Ollama : unload qwen2.5:14b -> load proteogen-qwen3-32b:q5 -> stream -> unload.
Modèle = Qwen3-32B Q5 reTemplaté (cf Modelfile_qwen3_proteogen) ; raisonnement coupé (/no_think).
Appelé exclusivement depuis Streamlit (Dev_app_token.py), bypass agent QWEN.
Dépendances légères (pandas only) — aucun import ESM/Keras (pas de chargement TF).
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
DEEPSEEK_MODEL = "UgXhicIH/proteogen-qwen3-32b:q5"
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
        activity_means = {str(k).replace("Peptide_", ""): float(v) for k, v in means.items()}

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


def _stream_deepseek(
    prompt: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    reload_qwen_after: bool = True,
) -> Iterator[str]:
    """Cœur streaming partagé : swap qwen->deepseek, génère, strip <think>, restaure qwen.
    Factorise la machinerie Ollama commune aux trois niveaux d'analyse.
    Yield tokens texte au fil de l'eau ; en cas d'erreur HTTP, yield message puis stop.
    """
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
    yield from _stream_deepseek(prompt, base_url=base_url, reload_qwen_after=reload_qwen_after)


# ============================================================
# RÉSOLUTION XLSX RÉSULTATS (≠ master clusters)
# Convention pipeline _resolve_run_path : fichier dans le CWD.
# ============================================================
def _resolve_run_xlsx(run_id: str) -> Optional[str]:
    """Résout run_id -> chemin absolu du XLSX résultats dans le CWD.
    Essaie {run_id}.xlsx puis PROTEOGEN_Results_{run_id}.xlsx. None si miss.
    """
    if not run_id:
        return None
    candidates = [
        run_id if run_id.endswith(".xlsx") else f"{run_id}.xlsx",
        run_id if run_id.startswith("PROTEOGEN_Results_") else f"PROTEOGEN_Results_{run_id}.xlsx",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def list_run_activities(run_id: str) -> dict:
    """Liste activités prédites + feuilles peptides du XLSX résultats. Peuple dropdowns synthèse globale.
    Indépendant du clustering (lit le résultat brut, colonnes 'Peptide_*').
    """
    run_xlsx = _resolve_run_xlsx(run_id)
    if not run_xlsx:
        return {"error": f"XLSX résultats introuvable pour '{run_id}' (attendu dans le CWD).",
                "run_xlsx": None, "activities": [], "sheets": {}}
    try:
        xls = pd.ExcelFile(run_xlsx)
    except Exception as e:
        return {"error": f"Lecture résultats échouée : {e}", "run_xlsx": run_xlsx, "activities": [], "sheets": {}}
    activities = set()
    sheets = {}
    for s in xls.sheet_names:
        if "_Peptides" not in str(s):
            continue
        try:
            df = pd.read_excel(run_xlsx, sheet_name=s)
        except Exception:
            continue
        if "Peptide" not in df.columns:
            continue
        activities.update(
            str(c)[len("Peptide_"):] for c in df.columns
            if str(c).startswith("Peptide_") and not str(c).endswith("_raw")
        )
        ser = df["Peptide"].astype(str).str.strip()
        sheets[s] = int(len(ser) - ser.isin(["", "nan", "None"]).sum())
    return {"error": None, "run_xlsx": run_xlsx, "activities": sorted(activities), "sheets": sheets}


# ============================================================
# CAPACITÉ 1 — SYNTHÈSE RUN GLOBAL (multi-activités, hors clusters)
# ============================================================
def _build_run_global_context(
    run_xlsx: str,
    target_activities: Optional[list] = None,
    hit_threshold: float = 0.5,
    top_peptides: int = 15,
) -> dict:
    """Agrège le run au niveau activités, toutes feuilles '*_Peptides' confondues.
    Par activité : nb positifs (proba >= seuil), taux, proba moyenne/max, incertitude MC moyenne.
    + distribution de l'activité dominante (argmax par peptide), top peptides multi-actifs, longueurs.
    Ne dépend PAS du clustering : recompute l'activité dominante si 'Activité_Dominante' absente.
    NB : la colonne 'Peptide_{activité}' est lue telle quelle (sortie pipeline).
    """
    try:
        xls = pd.ExcelFile(run_xlsx)
    except Exception as e:
        return {"error": f"Lecture résultats échouée : {e}"}
    frames, sheet_rows = [], {}
    for s in xls.sheet_names:
        if "_Peptides" not in str(s):
            continue
        try:
            df = pd.read_excel(run_xlsx, sheet_name=s)
        except Exception:
            continue
        if "Peptide" not in df.columns:
            continue
        df = df.copy()
        df["__sheet__"] = s
        frames.append(df)
        sheet_rows[s] = int(len(df))
    if not frames:
        return {"error": "Aucune feuille '*_Peptides' exploitable (run non traité ?)."}
    full = pd.concat(frames, ignore_index=True, sort=False)

    pred_cols = [c for c in full.columns
                 if str(c).startswith("Peptide_") and not str(c).endswith("_raw")]
    if not pred_cols:
        return {"error": "Aucune colonne 'Peptide_*' détectée (run non traité ?)."}
    all_acts = [str(c)[len("Peptide_"):] for c in pred_cols]
    if target_activities:
        want = set(target_activities)
        kept = [(c, a) for c, a in zip(pred_cols, all_acts) if a in want]
        if not kept:
            return {"error": f"Aucune activité ciblée présente. Disponibles : {', '.join(all_acts)}"}
        pred_cols, acts = [c for c, _ in kept], [a for _, a in kept]
    else:
        acts = all_acts

    n_total = int(len(full))
    pred_mat = full[pred_cols].apply(pd.to_numeric, errors="coerce")

    per_activity = []
    for c, a in zip(pred_cols, acts):
        valid = pred_mat[c].dropna()
        if valid.empty:
            continue
        unc_mean = None
        unc_col = f"Incertitude_{a}"
        if unc_col in full.columns:
            uc = pd.to_numeric(full[unc_col], errors="coerce").dropna()
            unc_mean = round(float(uc.mean()), 4) if not uc.empty else None
        n_hit = int((valid >= hit_threshold).sum())
        per_activity.append({
            "activité": a,
            "n_positifs": n_hit,
            "taux_positifs": round(n_hit / n_total, 4) if n_total else 0.0,
            "proba_moyenne": round(float(valid.mean()), 4),
            "proba_max": round(float(valid.max()), 4),
            "incertitude_moyenne": unc_mean,
        })
    per_activity.sort(key=lambda d: d["taux_positifs"], reverse=True)

    # Activité dominante par peptide (argmax), robuste aux lignes toutes-NaN via fillna(-1)
    dom_idx = pred_mat.fillna(-1.0).idxmax(axis=1)
    dom_val = pred_mat.max(axis=1)
    dom_act = dom_idx.map(lambda c: str(c)[len("Peptide_"):] if isinstance(c, str) else None)
    mask = dom_val.notna() & (dom_val >= hit_threshold)
    dom_counts = dom_act[mask].value_counts()
    dominant_distribution = {str(k): int(v) for k, v in dom_counts.head(12).items()}

    # Top peptides multi-actifs : nb activités >= seuil puis proba dominante
    n_active = (pred_mat >= hit_threshold).sum(axis=1)
    seq_col = "Seq_Clean" if "Seq_Clean" in full.columns else "Peptide"
    rank = pd.DataFrame({
        "peptide": full["Peptide"].astype(str),
        "sheet": full["__sheet__"].astype(str),
        "n_activités_positives": n_active.astype(int),
        "activité_dominante": dom_act,
        "proba_dominante": dom_val.round(4),
    }).sort_values(["n_activités_positives", "proba_dominante"], ascending=False).head(top_peptides)
    peptides_top = [
        {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        for rec in rank.to_dict(orient="records")
    ]

    lengths = full[seq_col].dropna().astype(str).map(len)
    lengths = lengths[lengths > 0]
    length_stats = {}
    if not lengths.empty:
        length_stats = {"min": int(lengths.min()), "max": int(lengths.max()),
                        "mean": round(float(lengths.mean()), 2)}

    base_id = os.path.splitext(os.path.basename(run_xlsx))[0].replace("PROTEOGEN_Results_", "", 1)
    return {
        "type": "run_global",
        "run_xlsx": os.path.basename(run_xlsx),
        "n_peptides_total": n_total,
        "n_feuilles": len(sheet_rows),
        "peptides_par_feuille": sheet_rows,
        "seuil_positif": hit_threshold,
        "longueurs": length_stats,
        "n_activités_analysées": len(per_activity),
        "activités": per_activity[:20],
        "distribution_activité_dominante": dominant_distribution,
        "peptides_top_multiactifs": peptides_top,
        "clustering_disponible": bool(_resolve_master_xlsx(base_id)),
    }


def _build_global_prompt(ctx: dict, user_question: Optional[str]) -> str:
    intro = (
        "Tu es un expert en peptidomique et bioinformatique. "
        "Tu interprètes la SYNTHÈSE GLOBALE d'un run PROTEOGEN "
        "(ESM-2 650M + CNN + Monte Carlo Dropout, prédictions multi-activités, calibration Platt). "
        "Vue d'ensemble toutes activités confondues, indépendante du clustering. "
        "Le seuil de positivité ('seuil_positif') sépare positifs/négatifs ; "
        "'incertitude_moyenne' = écart-type MC Dropout (plus haut = moins fiable).\n\n"
        f"Données agrégées du run :\n```json\n{json.dumps(ctx, ensure_ascii=False, indent=2)}\n```\n\n"
    )
    if user_question:
        return intro + f"Question opérateur :\n{user_question}\n" + _FORMAT_RULES
    return intro + (
        "Analyse demandée :\n"
        "1. Profil global du run (volume, feuilles, longueurs).\n"
        "2. Activités sur-représentées (taux de positifs, proba moyenne) ; fiabilité (incertitude MC).\n"
        "3. Co-occurrences et peptides multi-actifs notables.\n"
        "4. Activités à faible signal ou forte incertitude (prudence d'interprétation).\n"
        "5. Recommandations de priorisation expérimentale.\n"
    ) + _FORMAT_RULES


def explain_run_global_stream(
    run_id: str,
    target_activities: Optional[list] = None,
    user_question: Optional[str] = None,
    hit_threshold: float = 0.5,
    base_url: str = DEFAULT_OLLAMA_URL,
    reload_qwen_after: bool = True,
) -> Iterator[str]:
    """
    Stream DeepSeek : synthèse globale multi-activités d'un run (XLSX résultats, hors clustering).
    target_activities : sous-liste d'activités à restreindre (None = toutes les colonnes prédites).
    hit_threshold     : seuil de proba comptant un peptide comme positif (défaut 0.5).
    """
    run_xlsx = _resolve_run_xlsx(run_id)
    if not run_xlsx:
        yield f"**Erreur** : XLSX résultats introuvable pour run `{run_id}` (attendu dans le CWD)."
        return
    ctx = _build_run_global_context(run_xlsx, target_activities, hit_threshold)
    if "error" in ctx:
        yield f"**Erreur contexte** : {ctx['error']}"
        return
    prompt = _build_global_prompt(ctx, user_question)
    yield from _stream_deepseek(prompt, base_url=base_url, reload_qwen_after=reload_qwen_after)


# ============================================================
# CAPACITÉ 2 — COMPARAISON INTER-CLUSTERS / INTER-FEUILLES
# ============================================================
def _normalize_targets(targets: list) -> list:
    """Normalise une liste hétérogène -> [(sheet, cluster_id), ...].
    Accepte {'sheet','cluster_id'}, (sheet, cluster_id) ou [sheet, cluster_id].
    """
    norm = []
    for t in targets or []:
        if isinstance(t, dict):
            s, c = t.get("sheet"), t.get("cluster_id")
        elif isinstance(t, (list, tuple)) and len(t) >= 2:
            s, c = t[0], t[1]
        else:
            continue
        if s is None or c is None:
            continue
        norm.append((str(s), str(c)))
    return norm


def _build_comparison_context(master_xlsx: str, targets: list, max_peptides: int = 12) -> dict:
    """Construit le contexte comparatif pour ≥2 clusters (même feuille ou feuilles différentes).
    Réutilise _build_cluster_context par cluster, puis croise activités et tailles.
    """
    norm = _normalize_targets(targets)
    if len(norm) < 2:
        return {"error": "Comparaison : fournir au moins 2 clusters (sheet, cluster_id)."}
    if len(norm) > 4:
        max_peptides = 8  # garde le contexte sous num_ctx quand beaucoup de clusters

    clusters, errors, activity_union = [], [], {}
    for s, c in norm:
        ctx = _build_cluster_context(master_xlsx, s, c, max_peptides=max_peptides)
        if "error" in ctx:
            errors.append(f"{s}/{c} : {ctx['error']}")
            continue
        clusters.append(ctx)
        label = f"{s}/{c}"
        for a, v in ctx.get("activités_moyennes_top", {}).items():
            activity_union.setdefault(a, {})[label] = round(float(v), 4)
    if len(clusters) < 2:
        return {"error": "Moins de 2 clusters valides. " + " ; ".join(errors)}

    labels = [f"{x['sheet']}/{x['cluster_id']}" for x in clusters]
    shared = sorted([a for a, d in activity_union.items() if len(d) == len(clusters)])
    return {
        "type": "comparaison_clusters",
        "n_clusters": len(clusters),
        "clusters_comparés": labels,
        "tailles": {f"{x['sheet']}/{x['cluster_id']}": x["n_peptides"] for x in clusters},
        "longueurs": {f"{x['sheet']}/{x['cluster_id']}": x.get("longueurs", {}) for x in clusters},
        "activités_partagées": shared,
        "activités_moyennes_par_cluster": activity_union,
        "clusters": clusters,
        "erreurs_partielles": errors or None,
    }


def _build_comparison_prompt(ctx: dict, user_question: Optional[str]) -> str:
    intro = (
        "Tu es un expert en peptidomique et bioinformatique. "
        "Tu COMPARES plusieurs clusters de peptides bioactifs issus du pipeline PROTEOGEN "
        "(ESM-2 650M + CNN + MC Dropout + clustering PepFuNN/Butina), potentiellement de feuilles différentes. "
        "'activités_moyennes_par_cluster' donne la proba moyenne par activité et par cluster ; "
        "'activités_partagées' = activités présentes dans tous les clusters.\n\n"
        f"Données comparatives :\n```json\n{json.dumps(ctx, ensure_ascii=False, indent=2)}\n```\n\n"
    )
    if user_question:
        return intro + f"Question opérateur :\n{user_question}\n" + _FORMAT_RULES
    return intro + (
        "Analyse comparative demandée :\n"
        "1. Tableau récapitulatif : taille, longueurs, activité dominante par cluster.\n"
        "2. Activités partagées vs spécifiques ; écarts de probabilités moyennes.\n"
        "3. Similarités/divergences de composition et de motifs apparents.\n"
        "4. Hypothèses sur ce qui distingue les clusters (chimie, fonction).\n"
        "5. Lequel prioriser selon l'objectif (puissance, sélectivité, nouveauté).\n"
    ) + _FORMAT_RULES


def compare_clusters_stream(
    run_id: str,
    targets: list,
    user_question: Optional[str] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    reload_qwen_after: bool = True,
) -> Iterator[str]:
    """
    Stream DeepSeek : comparaison de ≥2 clusters (même feuille ou feuilles différentes).
    targets : liste de {'sheet','cluster_id'} (ou tuples [sheet, cluster_id]).
              Catalogue disponible via list_clusters(base_id)['by_sheet'].
    """
    base_id = run_id.replace("PROTEOGEN_Results_", "").replace(".xlsx", "")
    master = _resolve_master_xlsx(base_id)
    if not master:
        yield f"**Erreur** : master_xlsx introuvable pour run `{run_id}`. Lancer clustering d'abord."
        return
    ctx = _build_comparison_context(master, targets)
    if "error" in ctx:
        yield f"**Erreur contexte** : {ctx['error']}"
        return
    prompt = _build_comparison_prompt(ctx, user_question)
    yield from _stream_deepseek(prompt, base_url=base_url, reload_qwen_after=reload_qwen_after)