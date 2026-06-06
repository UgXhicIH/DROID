# Copyright (c) 2026 Morgan Letoux. All rights reserved.
# This file is part of PROTEOGEN/DROID.
# Unauthorized use, reproduction or distribution is strictly prohibited.
# See LICENSE for details.
"""
restyle_droid.py
================
Theme visuel DROID pour l'interface Streamlit PROTEOGEN.

Injection CSS via st.markdown qui cible les `data-testid` Streamlit (stables)
en `!important` pour ecraser le theme par defaut et le <style> .main-title
existant. La palette et le logo sont repris du dashboard DROID.

Integration dans Dev_app_token.py :
    import restyle_droid
    restyle_droid.apply()
    restyle_droid.sidebar_brand()
    restyle_droid.main_header("PROTEOGEN", "Agent ...")

Tweaks figes (cf. maquette "Apercu Interface DROID.html") :
    - avatar assistant : logo DROID (override CSS ::before)
    - accent UI : rouge DROID #B0473F (boutons, onglets, focus, avatar)
    - warn semantique : ocre #C58A2E (alertes warning legitimes)
    - sidebar : papier encre #2b2722
"""

from __future__ import annotations

import streamlit as st  # type: ignore


PALETTE = {
    "ok":        "#4F8A4F",
    "ok_2":      "#A1C9A1",
    "no":        "#B0473F",
    "warn":      "#C58A2E",
    "accent":    "#B0473F",
    "info":      "#4A6FA5",
    "ink":       "#1a1a1a",
    "ink_soft":  "#555555",
    "ink_faint": "#8a8a8a",
    "ink_ghost": "#b7b3aa",
    "paper":     "#fdfcfa",
    "paper_2":   "#f4ede0",
    "paper_3":   "#ebe5d6",
    "canvas":    "#eceae4",
    "ink_panel": "#2b2722",
    "rule":      "#d8d4cc",
    "rule_soft": "#e8e3d8",
}


def droid_logo_svg(size: int = 60, on_dark: bool = False) -> str:
    ring = "#f4ede0" if on_dark else "#3a3530"
    inner_dots = "#2b2722" if on_dark else "#fdfcfa"
    plate = "#f4ede0" if on_dark else "#3a3530"
    return f"""<svg viewBox="0 0 200 200" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <g transform="translate(100,100) scale(2)">
    <path d="M 0 -44 L 38.1051 -22 L 38.1051 22 L 0 44 L -38.1051 22 L -38.1051 -22 Z"
      fill="none" stroke="{ring}" stroke-width="0.9" stroke-dasharray="2 3" opacity="0.5"/>
    <path d="M 0 -36 L 31.1769 -18 L 31.1769 18 L 0 36 L -31.1769 18 L -31.1769 -18 Z"
      fill="{plate}"/>
    <path d="M 0 -24 L 20.7846 -12 L 20.7846 12 L 0 24 L -20.7846 12 L -20.7846 -12 Z"
      fill="none" stroke="{inner_dots}" stroke-width="1.2"/>
    <circle cx="0" cy="0" r="8" fill="#C58A2E"/>
    <circle cx="0" cy="0" r="3" fill="{inner_dots}"/>
    <circle cx="0" cy="-36" r="3.4" fill="{inner_dots}"/>
    <circle cx="31.1769" cy="-18" r="3.4" fill="{inner_dots}"/>
    <circle cx="31.1769" cy="18" r="3.4" fill="{inner_dots}"/>
    <circle cx="0" cy="36" r="3.4" fill="{inner_dots}"/>
    <circle cx="-31.1769" cy="18" r="3.4" fill="{inner_dots}"/>
    <circle cx="-31.1769" cy="-18" r="3.4" fill="{inner_dots}"/>
    <circle cx="0" cy="-44" r="2.2" fill="#C58A2E"/>
    <circle cx="38.1051" cy="-22" r="2.2" fill="#C58A2E"/>
    <circle cx="38.1051" cy="22" r="2.2" fill="#C58A2E"/>
    <circle cx="0" cy="44" r="2.2" fill="#C58A2E"/>
    <circle cx="-38.1051" cy="22" r="2.2" fill="#C58A2E"/>
    <circle cx="-38.1051" cy="-22" r="2.2" fill="#C58A2E"/>
  </g>
</svg>"""


def _logo_data_uri(size: int = 30, on_dark: bool = False) -> str:
    import base64
    svg = droid_logo_svg(size=size, on_dark=on_dark)
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _css() -> str:
    p = PALETTE
    logo_uri = _logo_data_uri(size=30, on_dark=False)
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {{
  --droid-ok:#4F8A4F; --droid-no:#B0473F; --droid-warn:#C58A2E; --droid-info:#4A6FA5;
  --droid-accent:{p['accent']};
  --droid-ink:{p['ink']}; --droid-ink-soft:{p['ink_soft']}; --droid-ink-faint:{p['ink_faint']};
  --droid-paper:{p['paper']}; --droid-paper-2:{p['paper_2']}; --droid-paper-3:{p['paper_3']};
  --droid-canvas:{p['canvas']}; --droid-rule:{p['rule']}; --droid-rule-soft:{p['rule_soft']};
  --droid-sans:"Calibri","Carlito","Trebuchet MS",system-ui,sans-serif;
  --droid-mono:"JetBrains Mono","SFMono-Regular",Consolas,monospace;
}}

.stApp, [data-testid="stAppViewContainer"] {{
  background:{p['canvas']} !important;
  font-family:var(--droid-sans) !important;
  color:{p['ink']} !important;
}}
[data-testid="stHeader"] {{ background:transparent !important; }}
[data-testid="stMainBlockContainer"], .block-container {{
  padding-top:1.4rem !important;
  max-width:1180px !important;
}}
[data-testid="stMain"] {{ background:{p['canvas']} !important; }}
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div {{
  background:{p['canvas']} !important;
}}

[data-testid="stSidebar"] {{
  background:{p['ink_panel']} !important;
  border-right:1px solid #1c1814 !important;
}}
[data-testid="stSidebar"] * {{ color:{p['paper_2']} !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stSubheader {{
  font-family:var(--droid-mono) !important;
  letter-spacing:.04em !important;
  color:{p['paper']} !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] em {{
  color:{p['ink_ghost']} !important;
}}
[data-testid="stSidebar"] hr {{ border-color:#473f37 !important; }}
[data-testid="stSidebar"] code {{
  background:#1f1b17 !important; color:#e6b15e !important;
  border:1px solid #473f37 !important;
}}

h1, h2, h3, h4 {{
  font-family:var(--droid-sans) !important;
  color:{p['ink']} !important;
  letter-spacing:.01em !important;
}}

.main-title, .droid-main-title {{
  font-family:var(--droid-sans) !important;
  font-size:2.1rem !important;
  font-weight:700 !important;
  letter-spacing:.02em !important;
  color:{p['ink']} !important;
  background:none !important;
  -webkit-text-fill-color:{p['ink']} !important;
  -webkit-background-clip:initial !important;
  text-align:left !important;
  padding:0 !important;
}}
.sub-caption, .droid-sub {{
  text-align:left !important;
  color:{p['ink_faint']} !important;
  font-family:var(--droid-mono) !important;
  font-size:.74rem !important;
  letter-spacing:.04em !important;
  text-transform:uppercase !important;
}}
.droid-header {{
  display:flex; align-items:center; gap:16px;
  border-bottom:1px solid {p['rule']};
  padding-bottom:14px; margin-bottom:18px;
}}
.droid-header .droid-header-txt {{ display:flex; flex-direction:column; gap:2px; }}

.stButton > button, [data-testid="stBaseButton-secondary"] {{
  font-family:var(--droid-mono) !important;
  font-size:.78rem !important;
  letter-spacing:.06em !important;
  text-transform:uppercase !important;
  color:{p['ink']} !important;
  background:{p['paper']} !important;
  border:1px solid {p['rule']} !important;
  border-radius:2px !important;
  box-shadow:none !important;
  transition:background .15s, border-color .15s !important;
}}
.stButton > button:hover, [data-testid="stBaseButton-secondary"]:hover {{
  background:{p['paper_2']} !important;
  border-color:{p['accent']} !important;
  color:{p['ink']} !important;
}}
[data-testid="stDownloadButton"] > button {{
  font-family:var(--droid-mono) !important;
  font-size:.78rem !important;
  letter-spacing:.06em !important;
  text-transform:uppercase !important;
  background:{p['ink']} !important;
  color:{p['paper_2']} !important;
  border:1px solid {p['ink']} !important;
  border-radius:2px !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
  background:{p['accent']} !important;
  border-color:{p['accent']} !important;
  color:#fff !important;
}}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div {{
  font-family:var(--droid-mono) !important;
  font-size:.82rem !important;
  background:#fff !important;
  color:{p['ink']} !important;
  border:1px solid {p['rule']} !important;
  border-radius:2px !important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {{
  border-color:{p['accent']} !important;
  box-shadow:0 0 0 1px {p['accent']} !important;
}}
[data-testid="stWidgetLabel"] p {{
  font-family:var(--droid-mono) !important;
  font-size:.72rem !important;
  letter-spacing:.05em !important;
  text-transform:uppercase !important;
  color:{p['ink_soft']} !important;
}}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {{
  background:{p['paper_2']} !important;
  border:1px dashed {p['rule']} !important;
  border-radius:2px !important;
}}
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {{
  background:{p['ink']} !important; color:{p['paper_2']} !important;
  border-color:{p['ink']} !important;
}}

[data-baseweb="tab-list"] {{
  gap:2px !important;
  border-bottom:1px solid {p['rule']} !important;
}}
[data-baseweb="tab"] {{
  font-family:var(--droid-mono) !important;
  font-size:.74rem !important;
  letter-spacing:.06em !important;
  text-transform:uppercase !important;
  color:{p['ink_faint']} !important;
  background:transparent !important;
}}
[data-baseweb="tab"][aria-selected="true"] {{
  color:{p['ink']} !important;
}}
[data-baseweb="tab-highlight"] {{ background:{p['accent']} !important; }}

[data-testid="stChatMessage"] {{
  background:{p['paper']} !important;
  border:1px solid {p['rule_soft']} !important;
  border-radius:3px !important;
  box-shadow:0 1px 0 rgba(0,0,0,.03) !important;
  color:{p['ink']} !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] em,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {{
  color:{p['ink']} !important;
}}
[data-testid="stChatInput"] textarea {{
  color:{p['ink']} !important;
  background:{p['paper']} !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
  color:{p['ink']} !important;
}}
[data-testid="stChatMessageAvatarAssistant"] {{
  background:transparent !important;
  color:transparent !important;
  background-image:url("{logo_uri}") !important;
  background-repeat:no-repeat !important;
  background-position:center !important;
  background-size:30px 30px !important;
}}
[data-testid="stChatMessageAvatarAssistant"] svg,
[data-testid="stChatMessageAvatarAssistant"] span {{
  visibility:hidden !important;
}}
[data-testid="stChatMessageAvatarUser"] {{
  background:{p['ink']} !important; color:{p['paper_2']} !important;
}}
[data-testid="stChatInput"] {{
  background:{p['paper']} !important;
  border:1px solid {p['rule']} !important;
  border-radius:3px !important;
}}
[data-testid="stChatInput"] textarea {{ font-family:var(--droid-sans) !important; }}
[data-testid="stChatInput"]:focus-within {{ border-color:{p['accent']} !important; }}

[data-testid="stVerticalBlockBorderWrapper"] {{
  background:{p['paper']} !important;
  border:1px solid {p['rule']} !important;
  border-radius:3px !important;
}}

[data-testid="stExpander"] details {{
  background:transparent !important;
  border:1px solid {p['rule']} !important;
  border-radius:2px !important;
}}
[data-testid="stExpander"] summary {{
  font-family:var(--droid-mono) !important;
  font-size:.78rem !important;
  letter-spacing:.04em !important;
}}

[data-testid="stAlert"] {{
  border-radius:2px !important;
  font-family:var(--droid-sans) !important;
  border-left:3px solid {p['ink_faint']} !important;
}}
[data-testid="stAlertContentSuccess"] {{ background:#A1C9A1 !important; border-left-color:{p['ok']} !important; }}
[data-testid="stAlertContentInfo"]    {{ background:{p['paper_2']} !important; border-left-color:{p['no']} !important; }}
[data-testid="stAlertContentWarning"] {{ background:#f8efdd !important; border-left-color:{p['warn']} !important; }}
[data-testid="stAlertContentError"]   {{ background:#f6e7e4 !important; border-left-color:{p['no']} !important; }}

hr, [data-testid="stDivider"] hr {{ border-color:{p['rule']} !important; }}
[data-testid="stMarkdownContainer"] code {{
  font-family:var(--droid-mono) !important;
  font-size:.82em !important;
  background:{p['paper_2']} !important;
  color:#9a5d12 !important;
  border:1px solid {p['rule_soft']} !important;
  border-radius:2px !important;
  padding:1px 5px !important;
}}
[data-testid="stProgress"] [role="progressbar"] > div {{ background:{p['accent']} !important; }}
[data-testid="stSpinner"] i {{ border-top-color:{p['accent']} !important; }}
[data-testid="stMarkdownContainer"] table {{
  border-collapse:collapse !important; font-family:var(--droid-sans) !important;
}}
[data-testid="stMarkdownContainer"] th {{
  background:{p['paper_2']} !important; font-family:var(--droid-mono) !important;
  font-size:.72rem !important; letter-spacing:.04em !important; text-transform:uppercase !important;
  color:{p['ink_soft']} !important; border-bottom:1px solid {p['rule']} !important;
}}
[data-testid="stMarkdownContainer"] td {{ border-bottom:1px solid {p['rule_soft']} !important; }}

/* ---------- Sidebar overrides (uniformisation) ---------- */
/* Labels widget sidebar : clairs (ink_ghost) au lieu de gris foncé invisible */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
  color:{p['ink_ghost']} !important;
}}

/* Zone uploader sidebar : fond papier crème + texte encre */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
  background:{p['paper']} !important;
  border:1px dashed {p['rule']} !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] *,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {{
  color:{p['ink']} !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {{
  background:{p['paper']} !important;
  border:1px solid {p['rule']} !important;
  border-radius:2px !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] small {{
  color:{p['ink_soft']} !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {{
  background:{p['ink']} !important;
  color:{p['ink']} !important;
  border:1px solid {p['ink']} !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] * {{
  color:{p['paper_2']} !important;
}}

/* Selectbox sidebar (Modèle Ollama) : champ clair + texte encre */
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
  background:{p['paper']} !important;
  border:1px solid {p['rule']} !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="select"] > div * {{
  color:{p['ink']} !important;
}}
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] [data-baseweb="menu"] > div,
[data-baseweb="popover"] [data-baseweb="menu"] ul,
[data-baseweb="menu"] {{
  background:{p['paper']} !important;
  color:{p['ink']} !important;
  border:1px solid {p['rule']} !important;
}}
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] li {{
  background:{p['paper']} !important;
  color:{p['ink']} !important;
  font-family:var(--droid-mono) !important;
}}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] li:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="menu"] li[aria-selected="true"] {{
  background:{p['paper_2']} !important;
  color:{p['ink']} !important;
}}

/* Text input sidebar (URL Ollama) : champ clair + texte encre */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {{
  background:{p['paper']} !important;
  color:{p['ink']} !important;
  border:1px solid {p['rule']} !important;
}}

/* Boutons sidebar (Effacer la conversation) : variante encre cohérente */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
  background:#3a3530 !important;
  color:{p['paper_2']} !important;
  border:1px solid #5a5147 !important;
}}
[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] * {{
  color:{p['paper_2']} !important;
}}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{
  background:{p['ink_panel']} !important;
  border-color:{p['accent']} !important;
}}

/* Alertes sidebar (st.success / st.info "Prêt : ...") : uniformisées encre clair */
[data-testid="stSidebar"] [data-testid="stAlert"],
[data-testid="stSidebar"] [data-testid="stAlert"] > div,
[data-testid="stSidebar"] [data-testid="stAlert"] [data-testid^="stAlertContent"] {{
  background:#fdfcfa !important;
  border-radius:2px !important;
}}
[data-testid="stSidebar"] [data-testid="stAlert"] {{
  border-left:3px solid {p['ink_faint']} !important;
}}
[data-testid="stSidebar"] [data-testid="stAlertContentSuccess"],
[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {{
  border-left-color:{p['ok']} !important;
}}
[data-testid="stSidebar"] [data-testid="stAlertContentInfo"],
[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {{
  border-left-color:{p['no']} !important;
}}
[data-testid="stSidebar"] [data-testid="stAlertContentWarning"],
[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {{
  border-left-color:{p['warn']} !important;
}}
[data-testid="stSidebar"] [data-testid="stAlertContentError"],
[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{
  border-left-color:{p['no']} !important;
}}
[data-testid="stSidebar"] [data-testid="stAlert"] *,
[data-testid="stSidebar"] [data-testid="stAlert"] p,
[data-testid="stSidebar"] [data-testid="stAlert"] strong,
[data-testid="stSidebar"] [data-testid="stAlert"] svg {{
  color:{p['ink_soft']} !important;
  fill:{p['ink_soft']} !important;
}}

/* ---------- Footer chat (zone de saisie) ---------- */
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottomBlockContainer"] {{
  background:{p['canvas']} !important;
}}
[data-testid="stChatInput"] {{
  background:{p['paper']} !important;
  border:1px solid {p['rule']} !important;
  border-radius:3px !important;
}}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"] {{
  background:{p['paper']} !important;
  border:none !important;
}}
[data-testid="stChatInput"] button {{
  background:{p['ink']} !important;
  color:{p['paper_2']} !important;
  border:1px solid {p['ink']} !important;
}}
[data-testid="stChatInput"] button:hover {{
  background:{p['accent']} !important;
  border-color:{p['accent']} !important;
}}
</style>"""


def apply() -> None:
    st.markdown(_css(), unsafe_allow_html=True)


def sidebar_brand(
    title: str = "DROID",
    tagline: str = "Deep learning for Residue Orchestration, Intelligence and Discovery",
) -> None:
    with st.sidebar:
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:12px;padding:4px 0 2px;">
  {droid_logo_svg(size=46, on_dark=True)}
  <div style="display:flex;flex-direction:column;line-height:1.05;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:1.35rem;font-weight:700;
                 letter-spacing:.08em;color:#fdfcfa;">{title}</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:.58rem;letter-spacing:.06em;
                 color:#b7b3aa;text-transform:uppercase;max-width:210px;">{tagline}</span>
  </div>
</div>""",
            unsafe_allow_html=True,
        )


def main_header(title: str, subtitle: str = "") -> None:
    sub_html = f'<span class="droid-sub">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f"""<div class="droid-header">
  {droid_logo_svg(size=54, on_dark=False)}
  <div class="droid-header-txt">
    <span class="droid-main-title">{title}</span>
    {sub_html}
  </div>
</div>""",
        unsafe_allow_html=True,
    )
