# proteogen_pipeline.py pour IA agentique, avec token_level

import os
import time
import warnings
from warnings import simplefilter
from datetime import datetime
import logging
import re
import sys
import subprocess
import gc
import collections
import threading
import functools
import urllib.request
import urllib.error
import json
import html as _html

# Global stop event — set via request_stop() to interrupt a running pipeline
_stop_event = threading.Event()

def request_stop():
    _stop_event.set()

def clear_stop():
    _stop_event.clear()

# Variables d'environnement — doivent être définies avant tout import TF/Keras
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Evite les conflits entre librairies MKL/OpenMP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Masque les logs d'info TensorFlow (garde Warning/Error)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'   # Active les optimisations oneDNN de TensorFlow (gain CPU)
warnings.filterwarnings("ignore", message=".*tf.reset_default_graph.*")
#simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from sklearnex import patch_sklearn  # type: ignore
patch_sklearn()
# Dossiers de sortie (relatifs au CWD au moment de l'exécution, crée automatiquement si inexistant)
DIR_LOGS          = "Logs_1280D"
DIR_GRAPHS        = "Graph_UniDL_Prediction_1280D"
DIR_XLSX_Interest = "Output_Accession_Interest_XLSX_File_1280D"
DIR_FASTA_Interest= "Output_Accession_Interest_FASTA_File_1280D"
DIR_CLUSTER       = "Output_Cluster_1280D"
DIR_SIGNALP       = "Output_SignalP_1280D"
DIR_ESMFOLD       = "Output_ESMFold_1280D"
DIR_DASHBOARD     = "Output_Dashboard_1280D"
DIR_ESMFOLD       = "Output_ESMfold_1280D"
# Nombre de passes MC Dropout — variable globale accessible par generate_XAI_model
mc_passes = 40
# Imports lourds (exécutés une seule fois au chargement du module)
from keras.models import load_model   # type: ignore
import torch                          # type: ignore
import esm                            # type: ignore
import pandas as pd                   # type: ignore
import numpy as np                    # type: ignore
import peptides                       # type: ignore
import peptidy                        # type: ignore
import matplotlib                     # type: ignore
matplotlib.use('Agg')                 # Backend non-interactif (serveur sans écran)
import matplotlib.pyplot as plt       # type: ignore
import Clustering_PepFuNN_1280D_UMAP  # type: ignore
import plotly.express as px           # type: ignore
import plotly.graph_objects as go     # type: ignore
from keras import backend as K        # type: ignore
from rdkit import Chem                # type: ignore
from Bio import SeqIO                 # type: ignore
from scipy.special import logit as scipy_logit, expit as sigmoid  # type: ignore
# ============================================================
# LISTE COMPLÈTE DES MODÈLES DISPONIBLES
    # 'nom_colonne' : nom de la colonne de prédiction dans le fichier Excel de sortie
    # 'model'       : chemin vers le modèle CNN Keras sauvegardé (.keras)
    # Les lignes commentées (#) correspondent aux modèles désactivés (non entraînés ou exclus)
# ============================================================
ALL_MODELS = [
    {"nom_colonne": "Chimiotaxie",            "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/ChmxTaq_PROTEOGEN_token_model_1280D.keras",
     "brier": 0.18348931312425373, "ece": 0.070895, "platt_a": 4.285494, "platt_b": -11.009266, "n_val": 912, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0615,"accuracy": 0.0147,"count": 68},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1529,"accuracy": 0.0928,"count": 97},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2482,"accuracy": 0.1918,"count": 73},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3556,"accuracy": 0.3483,"count": 89},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4534,"accuracy": 0.5109,"count": 92},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5532,"accuracy": 0.7,"count": 130},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6499,"accuracy": 0.708,"count": 137},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7491,"accuracy": 0.6917,"count": 120},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8365,"accuracy": 0.6909,"count": 55},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9539,"accuracy": 0.8824,"count": 51}
    ]},
    {"nom_colonne": "Cytokine",               "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Cytokine_PROTEOGEN_token_model_1280D.keras",
     "brier": 0.06620109880006493, "ece": 0.045403, "platt_a": 3.619203, "platt_b": -8.189797, "n_val": 4876, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0105,"accuracy": 0.0356,"count": 1883},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1484,"accuracy": 0.0968,"count": 93},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2498,"accuracy": 0.1758,"count": 91},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3517,"accuracy": 0.0938,"count": 96},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4484,"accuracy": 0.1444,"count": 90},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5546,"accuracy": 0.234,"count": 94},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6499,"accuracy": 0.5447,"count": 123},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7576,"accuracy": 0.7401,"count": 227},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8587,"accuracy": 0.8786,"count": 692},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9477,"accuracy": 0.9812,"count": 1487}
    ]},
    {"nom_colonne": "Pénétration_cellulaire", "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/CPP_PROTEOGEN_token_model_1280D.keras",
     "brier": 0.11426204082920248, "ece": 0.028418, "platt_a": 2.234238, "platt_b": 1.086905, "n_val": 2460, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0451,"accuracy": 0.0336,"count": 655},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.146,"accuracy": 0.1845,"count": 206},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2479,"accuracy": 0.2656,"count": 128},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3488,"accuracy": 0.4505,"count": 111},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4511,"accuracy": 0.5043,"count": 115},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5471,"accuracy": 0.4831,"count": 118},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6542,"accuracy": 0.6134,"count": 119},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7545,"accuracy": 0.6686,"count": 17},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.853,"accuracy": 0.8672,"count": 271},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9599,"accuracy": 0.9662,"count": 562}
    ]},
    {"nom_colonne": "Drug_Delivery",          "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/DrugDelivery_PROTEOGEN_token_model_1280D.keras",
     "brier": 0.08275502125231138, "ece": 0.009157, "platt_a": 2.402311, "platt_b": 3.128962, "n_val": 2624, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0346,"accuracy": 0.0343,"count": 817},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1452,"accuracy": 0.1531,"count": 196},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2502,"accuracy": 0.2349,"count": 149},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3456,"accuracy": 0.3707,"count": 116},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.447,"accuracy": 0.4624,"count": 93},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5511,"accuracy": 0.5915,"count": 71},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6485,"accuracy": 0.6667,"count": 6},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7556,"accuracy": 0.6977,"count": 86},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8526,"accuracy": 0.8235,"count": 153},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9772,"accuracy": 0.9796,"count": 883}
    ]},
    {"nom_colonne": "Régulation_Hormonale",   "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Hormone_PROTEOGEN_token_model_1280D.keras",
     "brier": 0.07068438288688891, "ece": 0.010352, "platt_a": 4.819943, "platt_b": 3.316501, "n_val": 4876, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0171,"accuracy": 0.0154,"count": 1749},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1479,"accuracy": 0.1339,"count": 239},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2536,"accuracy": 0.1923,"count": 104},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3471,"accuracy": 0.3939,"count": 99},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.454,"accuracy": 0.4626,"count": 147},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5507,"accuracy": 0.5682,"count": 132},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6544,"accuracy": 0.6976,"count": 205},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7526,"accuracy": 0.741,"count": 278},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8538,"accuracy": 0.8747,"count": 391},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9761,"accuracy": 0.97,"count": 1532}
    ]},
    {"nom_colonne": "Potentialisateur",   "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Potentiator_PROTEOGEN_token_DoRA_CNN_model.keras",
     "brier": 0.13433599245121514, "ece": 0.042651, "platt_a": 11.285076, "platt_b": 4.764228, "n_val": 431, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.039 ,"accuracy": 0.0235,"count": 85},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1502,"accuracy": 0.1935,"count": 31},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2505,"accuracy": 0.25  ,"count": 32},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.352 ,"accuracy": 0.4412,"count": 34},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4527,"accuracy": 0.5294,"count": 17},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5579,"accuracy": 0.4857,"count": 35},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6575,"accuracy": 0.5357,"count": 28},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7499,"accuracy": 0.7273,"count": 33},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8557,"accuracy": 0.9151,"count": 59},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9503,"accuracy": 0.9351,"count": 77}
    ]},
    {"nom_colonne": "Stimulation_enzymatique",   "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Potentiator_PROTEOGEN_token_DoRA_CNN_model.keras",
     "brier": 0.13177160222070358, "ece": 0.071827, "platt_a": 2.874647, "platt_b": -3.116224, "n_val": 2402, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0271,"accuracy": 0.0734,"count": 463},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1481,"accuracy": 0.2419,"count": 186},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2468,"accuracy": 0.2688,"count": 186},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3506,"accuracy": 0.2395,"count": 167},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4546,"accuracy": 0.2612,"count": 134},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5516,"accuracy": 0.3602,"count": 161},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.655 ,"accuracy": 0.5793,"count": 145},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.752 ,"accuracy": 0.7549,"count": 204},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8572,"accuracy": 0.9329,"count": 343},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.933 ,"accuracy": 0.9734,"count": 413}
    ]},
    {"nom_colonne": "Activité_Hémolytique",   "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/AcHemo_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.13238538446300185, "ece": 0.029125, "platt_a": 2.627478, "platt_b": -0.233982, "n_val": 3210, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0314,"accuracy": 0.0481,"count": 810},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1434, "accuracy": 0.1558,"count": 154},
        {"bin_start": 0.2, "bin_end": 0.3,"confidence": 0.247,"accuracy": 0.2522,"count": 115},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3528,"accuracy": 0.2667,"count": 135},
        {"bin_start": 0.4, "bin_end": 0.5,"confidence": 0.451,"accuracy": 0.3789, "count": 161},
        {"bin_start": 0.5,"bin_end": 0.6, "confidence": 0.5516,"accuracy": 0.5427,"count": 199},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6526,"accuracy": 0.6429,"count": 308},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7523,"accuracy": 0.7082,"count": 425},
        {"bin_start": 0.8, "bin_end": 0.9, "confidence": 0.8547, "accuracy": 0.8837, "count": 602}, 
        {"bin_start": 0.9, "bin_end": 1.0, "confidence": 0.9229, "accuracy": 0.9668, "count": 301}
    ]},
    {"nom_colonne": "Cytotoxicité",           "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Cytotoxic_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.047571035554607705, "ece": 0.029079, "platt_a": 2.450568, "platt_b": -2.386859, "n_val":  3196, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0179,"accuracy": 0.0358,"count": 1257},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.144,"accuracy": 0.1548,"count": 155},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2461,"accuracy": 0.1348,"count": 89},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3489,"accuracy": 0.2419,"count": 62},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4527,"accuracy": 0.3404,"count": 47},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5527,"accuracy": 0.3714,"count": 70},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.656,"accuracy": 0.5079,"count": 63},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7479,"accuracy": 0.6981,"count": 53},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8596,"accuracy": 0.9079,"count": 76},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9844,"accuracy": 0.9985,"count": 1324}
    ]},
    {"nom_colonne": "Neurotoxicité",          "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Neurotoxine_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.05652207517597067, "ece": 0.012461, "platt_a": 4.459136, "platt_b": 1.579437, "n_val": 4876, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0109,"accuracy": 0.0147,"count": 1910},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1441,"accuracy": 0.1479,"count": 169},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2494,"accuracy": 0.1491,"count": 114},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.351,"accuracy": 0.3614,"count": 83},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.454,"accuracy": 0.4783,"count": 69},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5575,"accuracy": 0.56,"count": 75},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6525,"accuracy": 0.7241,"count": 116},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7554,"accuracy": 0.7246,"count": 167},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.857,"accuracy": 0.8225,"count": 400},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9686,"accuracy": 0.9752,"count": 1773}
    ]},
    {"nom_colonne": "Toxicité",               "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Toxicity_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.08043014751105627, "ece": 0.051283, "platt_a": 3.377733, "platt_b": 1.968625, "n_val": 3864, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.04,"accuracy": 0.0102,"count": 1179},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1457,"accuracy": 0.0892,"count": 370},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2445,"accuracy": 0.3137,"count": 204},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3457,"accuracy": 0.5208,"count": 144},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4525,"accuracy": 0.6429,"count": 112},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.552,"accuracy": 0.7387,"count": 111},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6481,"accuracy": 0.7745,"count": 102},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7497,"accuracy": 0.7863,"count": 131},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8551,"accuracy": 0.7882,"count": 203},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9797,"accuracy": 0.9572,"count": 1308}
    ]},
    {"nom_colonne": "Anti_Toxine",            "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Anti_Toxin_PROTEOGEN_token_DORA_CNN_model.keras",
     'brier': 0.13607491674807587, "ece": 0.028244, "platt_a": 4.761979, "platt_b": -5.257137, "n_val": 246,
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0415,"accuracy": 0.0208,"count": 48},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1519,"accuracy": 0.0714,"count": 14},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2579,"accuracy": 0.2778,"count": 18},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence":   0.36,"accuracy": 0.4615,"count": 13},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4527,"accuracy": 0.4583,"count": 24},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5497,"accuracy":    0.6,"count": 20},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6554,"accuracy": 0.6667,"count": 21},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7484,"accuracy": 0.7333,"count": 15},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8477,"accuracy": 0.8621,"count": 29},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9576,"accuracy": 0.9318,"count": 44}
    ]},
    {"nom_colonne": "Allergène",              "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Allergen_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.08057520755009741, "ece": 0.027906, "platt_a": 3.34204, "platt_b": 3.732625, "n_val": 3639, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0415,"accuracy": 0.0237,"count": 1098},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1437,"accuracy": 0.1192,"count": 386},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2472,"accuracy": 0.3187,"count": 182},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3488,"accuracy": 0.458,"count": 131},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4532,"accuracy": 0.5631,"count": 103},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.554,"accuracy": 0.6064,"count": 94},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6435,"accuracy": 0.6696,"count": 112},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7542,"accuracy": 0.7944,"count": 107},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8572,"accuracy": 0.7977,"count": 173},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9796,"accuracy": 0.9705,"count": 1253}
    ]},
    {"nom_colonne": "Quorum_Sensing",         "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/QS_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.07362461640330556, "ece": 0.042863, "platt_a": 7.655157, "platt_b": 11.02569, "n_val": 584, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0201,"accuracy": 0.025,"count": 200},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1456,"accuracy": 0.1071,"count": 28},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2326,"accuracy": 0.1667,"count": 18},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3466,"accuracy": 0.1429,"count": 21},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4426,"accuracy": 0.619,"count": 21},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5495,"accuracy": 0.7143,"count": 14},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6346,"accuracy": 0.4783,"count": 23},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7581,"accuracy": 0.8824,"count": 34},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8566,"accuracy": 0.8919,"count": 37},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9754,"accuracy": 0.9628,"count": 188}
    ]},
    {"nom_colonne": "Anti_Virale",             "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/AV_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.07578374974451571, "ece": 0.029697, "platt_a": 2.588824, "platt_b": -0.122615, "n_val": 4588, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0302,"accuracy": 0.044,"count": 1387},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1436,"accuracy": 0.1138,"count": 378},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2483,"accuracy": 0.1266,"count": 229},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3492,"accuracy": 0.2828,"count": 198},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4516,"accuracy": 0.442,"count": 138},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5467,"accuracy": 0.5,"count": 148},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6513,"accuracy": 0.8189,"count": 127},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7539,"accuracy": 0.8263,"count": 190},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.853,"accuracy": 0.9032,"count": 279},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9799,"accuracy": 0.9749,"count": 1514}
    ]},
    {"nom_colonne": "Anti_Bacterien",         "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/AB_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.06667045243637942, "ece": 0.040717, "platt_a": 4.398331, "platt_b": -2.226266, "n_val": 1600, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0312,"accuracy": 0.0613,"count": 571},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1446,"accuracy": 0.1038,"count": 106},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2439,"accuracy": 0.1667,"count": 54},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.348,"accuracy": 0.18,"count": 50},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4507,"accuracy": 0.3667,"count": 30},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5455,"accuracy": 0.3864,"count": 44},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6524,"accuracy": 0.5088,"count": 57},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7455,"accuracy": 0.8163,"count": 49},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8577,"accuracy": 0.9818,"count": 55},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9878,"accuracy": 1.0,"count": 584}
    ]},
    {"nom_colonne": "Anti_Fongique",          "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/AF_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.16680802236032288, "ece": 0.044228, "platt_a": 3.768273, "platt_b": 6.578118, "n_val": 2336, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0492,"accuracy": 0.0366,"count": 191},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1514,"accuracy": 0.081,"count": 210},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2537,"accuracy": 0.1833,"count": 251},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3532,"accuracy": 0.3834,"count": 253},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4499,"accuracy": 0.4933,"count": 300},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.549,"accuracy": 0.5535,"count": 271},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6472,"accuracy": 0.7298,"count": 248},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7471,"accuracy": 0.7778,"count": 198},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8479,"accuracy": 0.8679,"count": 159},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9678,"accuracy": 0.902,"count": 255}
    ]},
    {"nom_colonne": "Anti_Parasitique",       "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/AntiParasitic_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.04695467524792357, "ece": 0.018611, "platt_a": 1.927235, "platt_b": -0.298947, "n_val": 4858, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0113,"accuracy": 0.0227,"count": 1936},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1441,"accuracy": 0.0984,"count": 193},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2452,"accuracy": 0.1613,"count": 93},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3492,"accuracy": 0.2152,"count": 79},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4525,"accuracy": 0.3871,"count": 62},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5524,"accuracy": 0.5181,"count": 83},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6512,"accuracy": 0.5786,"count": 140},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.755,"accuracy": 0.75,"count": 188},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8503,"accuracy": 0.8773,"count": 269},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.988,"accuracy": 0.9967,"count": 1815}
    ]},
    {"nom_colonne": "Anti_MRSA",              "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/AMRSA_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.06272620538073921, "ece": 0.02398, "platt_a": 4.308039, "platt_b": 0.629264, "n_val": 794, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0287,"accuracy": 0.0259,"count": 270},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1452,"accuracy": 0.2295,"count": 61},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2483,"accuracy": 0.1875,"count": 32},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3439,"accuracy": 0.2,"count": 35},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4441,"accuracy": 0.4737,"count": 19},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5513,"accuracy": 0.6,"count": 20},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6476,"accuracy": 0.6,"count": 15},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7546,"accuracy": 0.7895,"count": 19},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8556,"accuracy": 0.9545,"count": 22},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9903,"accuracy": 0.9867,"count": 301}
    ]},
    {"nom_colonne": "Anti_Biofilm",           "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/AntiBiofilm_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.05029028506504305, "ece": 0.022235, "platt_a": 2.673008, "platt_b": 0.505144, "n_val": 1112, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0169,"accuracy": 0.0115,"count": 434},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1432,"accuracy": 0.2286,"count": 35},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2516,"accuracy": 0.3793,"count": 29},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3492,"accuracy": 0.3488,"count": 43},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4528,"accuracy": 0.3333,"count": 39},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5564,"accuracy": 0.381,"count": 21},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6494,"accuracy": 0.6923,"count": 26},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.765,"accuracy": 0.9091,"count": 11},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8473,"accuracy": 0.9615,"count": 26},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9925,"accuracy": 0.9888,"count": 448}
    ]},
    {"nom_colonne": "Anti_Hyper_Tension",          "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/ACE_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.1117162550585916, "ece": 0.021423, "platt_a": 1.522305, "platt_b": 2.32561, "n_val": 3724, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0324,"accuracy": 0.0333,"count": 992},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1469,"accuracy": 0.1985,"count": 262},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2477,"accuracy": 0.2642,"count": 193},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3492,"accuracy": 0.3407,"count": 182},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4466,"accuracy": 0.4104,"count": 173},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.552,"accuracy": 0.4759,"count": 187},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6484,"accuracy": 0.6289,"count": 194},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7536,"accuracy": 0.6906,"count": 223},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8586,"accuracy": 0.8866,"count": 476},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9465,"accuracy": 0.9572,"count": 842}
    ]},
    {"nom_colonne": "Dipeptidyl_peptidase_IV", "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/DPPIV_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.1412936890168366, "ece": 0.070829, "platt_a": 5.275816, "platt_b": 7.823236, "n_val": 1062, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0596,"accuracy": 0.0314,"count": 191},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1383,"accuracy": 0.1176,"count": 119},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2487,"accuracy": 0.378,"count": 82},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3503,"accuracy": 0.5652,"count": 69},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4489,"accuracy": 0.4507,"count": 71},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5471,"accuracy": 0.5732,"count": 82},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6525,"accuracy": 0.5333,"count": 90},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7485,"accuracy": 0.5254,"count": 59},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8573,"accuracy": 0.7656,"count": 64},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9536,"accuracy": 0.9957,"count": 235}
    ]},
    {"nom_colonne": "Anti_Inflammatoire",     "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/AIP_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.10069551923599904, "ece": 0.031231, "platt_a": 2.9387, "platt_b": 3.784075, "n_val": 6104, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0303,"accuracy": 0.0165,"count": 1877},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1445,"accuracy": 0.0852,"count": 364},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2442,"accuracy": 0.2885,"count": 208},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3493,"accuracy": 0.3424,"count": 184},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4489,"accuracy": 0.5642,"count": 218},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5513,"accuracy": 0.694,"count": 232},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6501,"accuracy": 0.6899,"count": 287},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7542,"accuracy": 0.7881,"count": 486},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8581,"accuracy": 0.8432,"count": 861},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9444,"accuracy": 0.92,"count": 1387}
    ]},
    {"nom_colonne": "Anti_Age",               "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/AntiAge_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.0697625635662377, "ece": 0.019876, "platt_a": 2.896325, "platt_b": 5.474082, "n_val": 1200, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0268,"accuracy": 0.0278,"count": 432},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1415,"accuracy": 0.1562,"count": 64},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2487,"accuracy": 0.2927,"count": 41},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3457,"accuracy": 0.1935,"count": 31},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4533,"accuracy": 0.4667,"count": 30},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.552,"accuracy": 0.6296,"count": 27},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6635,"accuracy": 0.5357,"count": 28},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7598,"accuracy": 0.6667,"count": 39},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8609,"accuracy": 0.9059,"count": 85},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9659,"accuracy": 0.9716,"count": 423}
    ]},
    {"nom_colonne": "Anti_Amnésique",         "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/AntiAmnesique_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.04046499855698586, "ece": 0.019109, "platt_a": 1.673711, "platt_b": 2.282191, "n_val": 4876, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0164,"accuracy": 0.0049,"count": 2043},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1458,"accuracy": 0.16,"count": 175},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2428,"accuracy": 0.3111,"count": 90},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.344,"accuracy": 0.3699,"count": 73},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4516,"accuracy": 0.551,"count": 49},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5469,"accuracy": 0.5957,"count": 47},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6553,"accuracy": 0.7867,"count": 75},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7568,"accuracy": 0.8901,"count": 91},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8599,"accuracy": 0.8973,"count": 185},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.98,"accuracy": 0.9688,"count": 2048}
    ]},
    {"nom_colonne": "Neuropeptide",           "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Neuro_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.08346591308851685, "ece": 0.013393, "platt_a": 5.391221, "platt_b": 2.632106, "n_val": 4850, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0266,"accuracy": 0.0281,"count": 1460},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1451,"accuracy": 0.1561,"count": 346},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2466,"accuracy": 0.1698,"count": 265},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3465,"accuracy": 0.3134,"count": 201},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4488,"accuracy": 0.4811,"count": 185},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5514,"accuracy": 0.6108,"count": 185},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6492,"accuracy": 0.6739,"count": 184},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.752,"accuracy": 0.7546,"count": 216},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8531,"accuracy": 0.8685,"count": 289},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9793,"accuracy": 0.9756,"count": 1519}
    ]},
    {"nom_colonne": "Anti_Cancer",            "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/AntiCancer_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.10470496065855767, "ece": 0.048684, "platt_a": 1.386577, "platt_b": 0.283643, "n_val": 4824, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0474,"accuracy": 0.0599,"count": 1352},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1422,"accuracy": 0.215,"count": 414},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2503,"accuracy": 0.2946,"count": 224},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3502,"accuracy": 0.3317,"count": 202},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4516,"accuracy": 0.3144,"count": 194},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.549,"accuracy": 0.3904,"count": 228},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6504,"accuracy": 0.4871,"count": 310},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.751,"accuracy": 0.75,"count": 296},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8508,"accuracy": 0.9633,"count": 327},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9768,"accuracy": 0.9953,"count": 1277}
    ]},
    {"nom_colonne": "Anti_Tumeur",            "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/TumorHomTar_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.09681113177202805, "ece": 0.02802, "platt_a": 7.910248, "platt_b": 8.38331, "n_val": 1398, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0307,"accuracy": 0.0226,"count": 399},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.142,"accuracy": 0.1146,"count": 96},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2517,"accuracy": 0.2152,"count": 79},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3547,"accuracy": 0.377,"count": 61},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4481,"accuracy": 0.5821,"count": 67},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5469,"accuracy": 0.6379,"count": 58},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6502,"accuracy": 0.6393,"count": 61},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.757,"accuracy": 0.7176,"count": 85},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8519,"accuracy": 0.8911,"count": 101},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9714,"accuracy": 0.954,"count": 391}
    ]},
    {"nom_colonne": "Anti_Oxydant",           "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Anti_Oxydative_PROTEOGEN_token_model_1280D.keras"
     ,"brier": 0.1167181421442858, "ece": 0.010942, "platt_a": 9.292753, "platt_b": 7.03234, "n_val": 2174, 
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0542,"accuracy": 0.0567,"count": 441},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1426,"accuracy": 0.1489,"count": 309},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2457,"accuracy": 0.2012,"count": 164},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3455,"accuracy": 0.3667,"count": 120},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4567,"accuracy": 0.4524,"count": 84},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5538,"accuracy": 0.5402,"count": 87},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6496,"accuracy": 0.7308,"count": 78},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7582,"accuracy": 0.7463,"count": 134},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8576,"accuracy": 0.8522,"count": 230},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9511,"accuracy": 0.9507,"count": 527}
    ]},
    {"nom_colonne": "Opioïde",           "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Opioid_PROTEOGEN_token_model_LoRa_1280D.keras"
     ,"brier": 0.04997639684660782, "ece": 0.024685, "platt_a": 5.647782, "platt_b": 2.761459, "n_val": None,
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0162,"accuracy": 0.0189,"count": 106},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1412,"accuracy": 0.1111,"count": 9},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2332,"accuracy": 0.0,"count": 8},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3504,"accuracy": 0.6,"count": 5},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.454,"accuracy": 0.5,"count": 10},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5605,"accuracy": 0.5,"count": 4},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6643,"accuracy": 0.5556,"count": 9},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7592,"accuracy": 0.8333,"count": 6},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8464,"accuracy": 1.0,"count": 6},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9816,"accuracy": 0.9817,"count": 109}
    ]},
    {"nom_colonne": "Anti_Diabétique",          "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Anti_Diabetic_PROTEOGEN_token_DoRA_CNN_model",
     'brier': 0.17144489045298766, "ece": 0.035104, "platt_a": 2.15849, "platt_b": 3.138361, "n_val": 2858,
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0468,"accuracy": 0.0292,"count": 343},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1526,"accuracy": 0.0977,"count": 215},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2505,"accuracy": 0.2289,"count": 284},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3502,"accuracy": 0.3793,"count": 274},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4471,"accuracy": 0.5481,"count": 208},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5508,"accuracy": 0.5963,"count": 218},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6544,"accuracy": 0.6415,"count": 265},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7556,"accuracy": 0.7495,"count": 487},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.84  ,"accuracy": 0.8033,"count": 539},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.912 ,"accuracy": 1.0   ,"count": 25}
    ]},
    {"nom_colonne": "Inhibition_Coagulation",          "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Coag_Inhibitor_PROTEOGEN_token_DoRA_CNN_model.keras",
     'brier': 0.10571146423768557, "ece": 0.031282, "platt_a": 4.9475, "platt_b": 2.179472, "n_val": 4256,
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.0146,"accuracy": 0.0337,"count": 1187},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1463,"accuracy": 0.1146,"count": 157},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2463,"accuracy": 0.2202,"count": 168},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.35  ,"accuracy": 0.2649,"count": 151},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4529,"accuracy": 0.3466,"count": 176},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5506,"accuracy": 0.4571,"count": 175},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6527,"accuracy": 0.6141,"count": 241},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.7549,"accuracy": 0.8026,"count": 461},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8552,"accuracy": 0.8784 ,"count": 732},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.937 ,"accuracy": 0.9431,"count": 808}
    ]},
    {"nom_colonne": "Umami",          "model": "Model_TOKEN_CNN_PROTEOGEN_1280D/Umami_PROTEOGEN_token_model_1280D.keras",
     'brier': 0.10534684918818943, "ece": 0.022235, "platt_a": 5.854702, "platt_b": 0.578756, "n_val": 598,
     "ece_bins": [
        {"bin_start": 0.0,"bin_end": 0.1,"confidence": 0.033,"accuracy": 0.0303,"count": 165},
        {"bin_start": 0.1,"bin_end": 0.2,"confidence": 0.1417,"accuracy": 0.1296,"count": 54},
        {"bin_start": 0.2,"bin_end": 0.3,"confidence": 0.2556,"accuracy": 0.1111,"count": 18},
        {"bin_start": 0.3,"bin_end": 0.4,"confidence": 0.3455,"accuracy": 0.3793,"count": 29},
        {"bin_start": 0.4,"bin_end": 0.5,"confidence": 0.4513,"accuracy": 0.4286,"count": 21},
        {"bin_start": 0.5,"bin_end": 0.6,"confidence": 0.5514,"accuracy": 0.65,"count": 20},
        {"bin_start": 0.6,"bin_end": 0.7,"confidence": 0.6608,"accuracy": 0.7576,"count": 33},
        {"bin_start": 0.7,"bin_end": 0.8,"confidence": 0.756,"accuracy": 0.7333,"count": 30},
        {"bin_start": 0.8,"bin_end": 0.9,"confidence": 0.8542,"accuracy": 0.871,"count": 93},
        {"bin_start": 0.9,"bin_end": 1.0,"confidence": 0.9396,"accuracy": 0.9185,"count": 135}
    ]},
]

AVAILABLE_ACTIVITIES = [m["nom_colonne"] for m in ALL_MODELS]
ALL_MODULES = [
    "Prédictions DL (ESM-2 + CNN + MC Dropout)  — toujours actif",
    "Paramètres physico-chimiques (MW, pI, VHSE, Z-Scales, Cruciani, MS-WHIM)  — toujours actif",
    "Génération SMILES",
    "XAI — Alanine Scanning top-10",
    "Clustering PepFuNN : UMAP + Treemap",
    "Extraction FASTA + SignalP 6.0",
    "ESMFold2 — Prédiction structure 3D peptides (.cif + pLDDT)",
]
# ============================================================
# FONCTIONS CŒUR — NE PAS MODIFIER
# (esm_embeddings, generate_XAI_model, plot_XAI_motif, plot_global_XAI)
# ============================================================
# Fonction ESM-2 : convertit une séquence peptidique en vecteur numérique de 1280 dimensions
# Utilise la couche 33 (dernière) du modèle esm2_t33_650M_UR50D — ne pas modifier
# Entrée  : liste de tuples (id, séquence)
# Sortie  : DataFrame (N_séquences x 1280) représentant les embeddings
MAX_LEN = 50
def esm_embeddings_token_level(esm2, esm2_alphabet, peptide_sequence_list, max_len=MAX_LEN):
    """Retourne un tenseur (1, MAX_LEN, 1280) pour une séquence."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(38) #Nombre de threads aloués pour les opérations
    esm2 = esm2.eval().to(device)
    batch_converter = esm2_alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
    batch_lens = (batch_tokens != esm2_alphabet.padding_idx).sum().item()
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad(): # Désactive le calcul de gradient (inférence uniquement, économise la mémoire)
        results = esm2(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33].cpu().numpy()[0,1:batch_lens - 1] # Tenseur (N, L, 1280) rapatrié sur CPU
    sequence_representations = token_representations.shape[0]
    if sequence_representations > max_len:
        token_representations = token_representations[:max_len, :]
    else:
        padding = np.zeros((max_len - sequence_representations, 1280), dtype=np.float32)
        token_representations = np.concatenate([token_representations, padding], axis=0)
    return token_representations[np.newaxis, ...] # Retourne un tenseur (1, MAX_LEN, 1280)

# Fonction XAI : Alanine Scanning informatique
# Remplace chaque acide aminé par 'X' (acide aminé inconnu) et mesure l'impact sur la prédiction
# Un impact positif = l'AA contribue à l'activité, négatif = il la diminue
# Entrée  : séquence, modèles ESM+CNN, scaler, nom de l'activité
# Sortie  : liste des scores d'impact par position + score de base
def generate_XAI_model(sequence, esm_model, alphabet, cnn_model, activity_name):
    # Alanine Scanning Informatique (MC Dropout)
    logging.info(f"Début XAI pour la séquence : {sequence}")
    # Score de référence : prédiction sur la séquence originale non mutée
    baseline_esm = esm_embeddings_token_level(esm_model, alphabet, sequence)  # (1, MAX_LEN, 1280)
    baseline_pred  = cnn_model.predict(baseline_esm, batch_size=16, verbose=0)
    baseline_score  = np.ravel(baseline_pred)[-1]

    impact_score = []
    # Boucle de mutation : chaque position est remplacée par 'X' tour à tour
    for i in range(len(sequence)):
        seq_alinise    = sequence[:i] + "X" + sequence[i + 1:]
        esm_alinise    = esm_embeddings_token_level(esm_model, alphabet, seq_alinise)
        mc_alinise_pred = []
        for _ in range(mc_passes):
            mc_alinise_pred.append((cnn_model(esm_alinise, training=True).numpy()) for _ in range(mc_passes))
        score_alinise = np.ravel(np.mean(mc_alinise_pred, axis=0))[-1]
        # Δ score : positif = AA important, négatif = AA défavorable
        impact = baseline_score - score_alinise
        impact_score.append(impact)
    return impact_score, baseline_score

def plot_XAI_motif(sequence, impact_score, baseline_score, activity_name, out_path):
    df_XAI = pd.DataFrame({
        "Position":    list(range(1, len(sequence) + 1)),
        "Acide_Aminé": list(sequence),
        "Score_Impact": impact_score,
    })
    fig = px.bar(
        df_XAI, x="Acide_Aminé", y="Score_Impact",
        color="Score_Impact", color_continuous_scale="RdBu_R", color_continuous_midpoint=0,
        text="Acide_Aminé",
        title=f"Motif Actif : {activity_name}, Score Global : {baseline_score:.3f}",
    )
    fig.update_traces(textposition="outside", textfont_size=14)
    fig.update_layout(template="simple_white", xaxis_title="Sequence Peptidique", yaxis_title="Score d'impact")
    out_path_Heat_XAI = os.path.join(DIR_GRAPHS, f"Heatmap_XAI_{sequence}.html")
    fig.write_html(out_path_Heat_XAI)
    logging.info(f"Heatmap XAI sauvegardée : {out_path_Heat_XAI}")

def plot_global_XAI(global_impacts, activity_name):
    aa_nom, impact_moyen = [], []
    for aa, list_impacts in global_impacts.items():
        aa_nom.append(aa)
        impact_moyen.append(np.mean(list_impacts) if list_impacts else 0.0)
    df_global_XAI = (
        pd.DataFrame({"Acide_Aminé": aa_nom, "Impact_Moyen": impact_moyen})
        .sort_values(by="Impact_Moyen", ascending=False)
    )
    fig = px.bar(
        df_global_XAI, x="Acide_Aminé", y="Impact_Moyen",
        color="Impact_Moyen", color_continuous_scale="RdBu_R", color_continuous_midpoint=0,
        text="Acide_Aminé",
        title=f"Motif actif récurrent : {activity_name}",
    )
    fig.update_traces(textposition="outside", textfont_size=14)
    fig.update_layout(template="simple_white", xaxis_title="Acide Aminé", yaxis_title="Impact Moyen d'activité")
    out_path_global_XAI = os.path.join(DIR_GRAPHS, f"Motif_global_XAI_{activity_name}.html")
    fig.write_html(out_path_global_XAI)
    logging.info(f"Motif global XAI sauvegardé : {out_path_global_XAI}")
# ============================================================
# ESM-FOLD / Docking
# ============================================================
def run_esmfold_module(
    dataset,
    esm2_model=None,
    sheet_name: str = "",
    esmfold_activities: list | None = None,
    activity_threshold: float = 0.95,
    top_n: int | None = None,
):
    """
    Module ESM-Fold — prédiction de structure 3D (concurrent direct d'AlphaFold), continuité pipeline.

    Sélectionne, par activité de `esmfold_activities`, les peptides dont la colonne
    Peptide_<activité> >= `activity_threshold` (au plus `top_n`, triés proba décroissante),
    les replie via ESMFold2 (biohub/ESMFold2) en local sur GPU, écrit un .cif par séquence unique
    + un manifeste Excel, et reporte la pLDDT moyenne dans `dataset` (colonne ESMFold_pLDDT).

    Returns:
        dict { dataset, esm_model, alphabet, n_predicted, n_failed, pdb_dir }
    """
    activities = esmfold_activities or []
    pdb_dir = os.path.join(DIR_ESMFOLD, str(sheet_name) if sheet_name else "run")
    os.makedirs(pdb_dir, exist_ok=True)

    def _clean(seq) -> str:
        c = re.sub(r"\(.*?\)", "", str(seq))
        return re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "", c.upper())

    # ── Libère l'ESM-2 entrant pour réduire le pic VRAM avant chargement ESMFold ──
    if esm2_model is not None:
        try:
            del esm2_model
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── Sélection des peptides à replier (seuil + top_n par activité) ─────────────
    selection = {}  # clean_seq -> {"orig", "activities": set, "max_proba"}
    for act in activities:
        col = f"Peptide_{act}"
        if col not in dataset.columns:
            logging.warning(f"ESM-Fold : colonne '{col}' absente — activité '{act}' ignorée")
            continue
        sub = dataset[dataset[col] >= activity_threshold].copy()
        if sub.empty:
            logging.info(f"ESM-Fold : aucun peptide >= {activity_threshold} pour '{act}'")
            continue
        sub = sub.sort_values(by=col, ascending=False)
        if top_n:
            sub = sub.head(int(top_n))
        for _, row in sub.iterrows():
            cseq = _clean(row["Peptide"])
            if not cseq:
                continue
            rec = selection.setdefault(cseq, {"orig": str(row["Peptide"]), "activities": set(), "max_proba": 0.0})
            rec["activities"].add(act)
            rec["max_proba"] = max(rec["max_proba"], float(row[col]))

    n_selected = len(selection)
    logging.info(f"ESM-Fold : {n_selected} séquence(s) unique(s) sélectionnée(s) (feuille '{sheet_name}')")

    n_predicted, n_failed = 0, 0
    plddt_map, records = {}, []

    # ── Chargement ESMFold2 (biohub/ESMFold2, GPU si disponible, repli CPU sinon) ──
    fold_model, fold_builder, device = None, None, "cpu"
    if n_selected > 0:
        try:
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
            from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
            fold_model = ESMFold2Model.from_pretrained("biohub/ESMFold2").eval()
            if torch.cuda.is_available():
                try:
                    fold_model = fold_model.cuda()
                    device = "cuda"
                except RuntimeError as e:
                    logging.warning(f"ESM-Fold : GPU indisponible, repli CPU ({e})")
                    fold_model = fold_model.cpu()
                    torch.cuda.empty_cache()
            fold_builder = ESMFold2InputBuilder()
            logging.info(f"ESM-Fold : ESMFold2 (biohub/ESMFold2) chargé sur {device}")
        except Exception as e:
            logging.error(f"ESM-Fold : échec chargement ESMFold2 : {e}", exc_info=True)
            fold_model = None
    else:
        logging.info("ESM-Fold : aucune séquence sélectionnée — repliement ignoré")

    # ── Repliement séquence par séquence (échecs isolés, non bloquants) ───────────
    if fold_model is not None:
        for idx, (cseq, rec) in enumerate(selection.items(), start=1):
            try:
                spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=cseq)])
                with torch.no_grad():
                    result = fold_builder.fold(
                        fold_model, spi,
                        num_loops=20, num_sampling_steps=100, num_diffusion_samples=1, seed=0,
                    )
                plddt = float(result.plddt.mean())
                fname = f"{idx:03d}_{cseq[:30]}.cif"
                with open(os.path.join(pdb_dir, fname), "w", encoding="utf-8") as fh:
                    fh.write(result.complex.to_mmcif())
                plddt_map[cseq] = plddt
                records.append({
                    "Peptide": rec["orig"],
                    "Clean_Sequence": cseq,
                    "Length": len(cseq),
                    "Activities": ", ".join(sorted(rec["activities"])),
                    "Max_Proba": round(rec["max_proba"], 4),
                    "Mean_pLDDT": None if np.isnan(plddt) else round(plddt, 2),
                    "Structure_File": fname,
                })
                n_predicted += 1
                logging.info(f"ESM-Fold [{idx}/{n_selected}] {cseq[:20]}… pLDDT={plddt:.3f} → {fname}")
            except Exception as e:
                n_failed += 1
                logging.error(f"ESM-Fold : échec repliement '{cseq[:20]}…' : {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        n_failed = n_selected

    # ── Manifeste Excel (peptide ↔ activités ↔ pLDDT ↔ PDB) ───────────────────────
    if records:
        try:
            manifest_path = os.path.join(pdb_dir, f"ESMFold_manifest_{sheet_name}.xlsx")
            pd.DataFrame(records).to_excel(manifest_path, index=False)
            logging.info(f"ESM-Fold : manifeste → {manifest_path}")
        except Exception as e:
            logging.warning(f"ESM-Fold : écriture manifeste échouée : {e}")

    # ── Report pLDDT moyenne dans le dataset ──────────────────────────────────────
    try:
        dataset["ESMFold_pLDDT"] = dataset["Peptide"].map(lambda s: plddt_map.get(_clean(s)))
    except Exception as e:
        logging.warning(f"ESM-Fold : ajout colonne ESMFold_pLDDT échoué : {e}")

    # ── Libère ESMFold, recharge ESM-2 pour la suite du pipeline ──────────────────
    if fold_model is not None:
        del fold_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logging.info("ESM-Fold : rechargement ESM-2 (esm2_t33_650M_UR50D) pour la suite du pipeline")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    return {
        "dataset": dataset,
        "esm_model": esm_model,
        "alphabet": alphabet,
        "n_predicted": n_predicted,
        "n_failed": n_failed,
        "pdb_dir": pdb_dir,
    }
# ============================================================
# PLATT SCALING
# ============================================================
def apply_platt(probas: np.ndarray, platt_a: float, platt_b: float, eps: float = 1e-7) -> np.ndarray:  
    """
    Recalibre probabilités MC Dropout via Platt scaling.
    calibrated_p = sigmoid(a * logit(p) + b)
    a, b stockés dans ALL_MODELS, calculés par compute_calibration.py.
    """
    p_clamp = np.clip(probas, eps, 1.0 - eps)
    logits = scipy_logit(p_clamp)
    return sigmoid(platt_a * logits + platt_b)
# ============================================================
# NOTIFICATION NTFY — ping fin de run
# ============================================================
def send_ntfy_notification(message: str, title: str = "PROTEOGEN", priority: str = "default", tags: str = "") -> bool:
    topic = os.environ.get("PROTEOGEN_NTFY_TOPIC") or "proteogen_UgXhicIH"
    base_url = os.environ.get("PROTEOGEN_NTFY_URL", "https://ntfy.sh").rstrip("/")
    url = f"{base_url}/{topic}"
    headers = {
        "Title": title.encode("utf-8"),
        "Priority": priority,
    }
    if tags:
        headers["Tags"] = tags
    try:
        req = urllib.request.Request(url, data=message.encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        logging.warning(f"ntfy notification échouée : {e}")
        return False

def _ntfy_on_critical_error(label: str):
    """Décorateur : ping ntfy si exception non gérée, puis re-raise."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                try:
                    send_ntfy_notification(
                        message=f"{label} — crash {type(e).__name__} : {e}",
                        title=f"PROTEOGEN — ERREUR {label}",
                        priority="urgent",
                        tags="rotating_light,x",
                    )
                except Exception:
                    pass
                raise
        return wrapper
    return deco
# ============================================================
# CONTRÔLE QUALITÉ — CALIBRATION BRIER / ECE
# ============================================================
def quality_control(activity: str, aspect: str = "all") -> dict:
    """
    Retourne métriques calibration post-hoc pour une activité.
    Args:
        activity : nom activité (ex: "Anti_Bacterien")
        aspect   : "calibration" | "coverage" | "uncertainty" | "all"
                   Seul "calibration" et "all" implémentés ici.
    Returns:
        dict avec brier, ece, n_val, ece_bins, reliability_diagram_path (si généré)
    """
    # Recherche modèle
    model_cfg = None
    for m in ALL_MODELS:
        if m["nom_colonne"] == activity:
            model_cfg = m
            break
    if model_cfg is None:
        return {"error": f"Activité '{activity}' non trouvée dans ALL_MODELS"}
    result: dict = {"activity": activity, "aspect": aspect}
    if aspect in ("calibration", "all"):
        brier    = model_cfg.get("brier")
        ece      = model_cfg.get("ece")
        n_val    = model_cfg.get("n_val")
        ece_bins = model_cfg.get("ece_bins")
        if brier is None or ece is None:
            result["calibration"] = {
                "status": "non_calculé",
                "message": (
                    f"Calibration pas encore calculée pour {activity}. "
                    f"Lancer compute_calibration.py avec validation set puis "
                    f"reporter brier/ece/ece_bins dans ALL_MODELS."
                ),
            }
        else:
            # Interprétation automatique
            if brier < 0.05:
                brier_grade = "excellent"
            elif brier < 0.10:
                brier_grade = "bon"
            elif brier < 0.20:
                brier_grade = "modéré"
            else:
                brier_grade = "faible"
            if ece < 0.03:
                ece_grade = "très bien calibré"
            elif ece < 0.07:
                ece_grade = "bien calibré"
            elif ece < 0.15:
                ece_grade = "calibration modérée"
            else:
                ece_grade = "mal calibré"
            result["calibration"] = {
                "status":      "calculé",
                "brier":       brier,
                "brier_grade": brier_grade,
                "ece":         ece,
                "ece_grade":   ece_grade,
                "n_val":       n_val,
            }
            # Génération reliability diagram si bins disponibles
            if ece_bins:
                os.makedirs(DIR_GRAPHS, exist_ok=True)
                diagram_path = os.path.join(DIR_GRAPHS, f"Reliability_{activity}.html")
                _plot_reliability_diagram(ece_bins, activity, brier, ece, diagram_path)
                result["calibration"]["reliability_diagram"] = diagram_path
                logging.info(f"Reliability diagram généré : {diagram_path}")
    return result

def quality_control_all(activities: list | None = None) -> list:
    """
    QC calibration pour plusieurs activités (ou toutes si None).
    Retourne liste de dicts quality_control().
    """
    targets = activities if activities else AVAILABLE_ACTIVITIES
    return [quality_control(act, aspect="calibration") for act in targets]

def _plot_reliability_diagram(bins_data: list, activity: str, brier: float, ece: float, out_path: str):
    """Reliability diagram Plotly → HTML."""
    confidences = [b["confidence"] for b in bins_data if b["count"] > 0]
    accuracies  = [b["accuracy"]   for b in bins_data if b["count"] > 0]
    counts      = [b["count"]      for b in bins_data if b["count"] > 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Calibration parfaite",
    ))
    fig.add_trace(go.Bar(
        x=confidences, y=accuracies,
        width=0.08,
        marker_color="rgba(55, 128, 191, 0.7)",
        marker_line=dict(color="rgba(55, 128, 191, 1)", width=1),
        name="Accuracy observée",
        text=[f"n={c}" for c in counts],
        textposition="outside",
        textfont_size=9,
    ))
    fig.update_layout(
        title=f"Reliability Diagram — {activity}<br>"
              f"<sub>Brier = {brier:.4f} | ECE = {ece:.4f}</sub>",
        xaxis_title="Confiance moyenne (prédite)",
        yaxis_title="Accuracy moyenne (observée)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        template="simple_white",
        width=700, height=550,
        legend=dict(x=0.02, y=0.98),
    )
    fig.write_html(out_path)
# ============================================================
# DASHBOARD CLIENT — HTML interactif autonome (sortie pipeline)
# Style DROID (rapport Peptidomique) : fond crème, carte papier,
# typographie Calibri/JetBrains Mono/Cambria, palette ok/warn/no.
# ============================================================
_DROID_PALETTE = {
    "ok":   (0x4F, 0x8A, 0x4F),
    "warn": (0xC5, 0x8A, 0x2E),
    "no":   (0xB0, 0x47, 0x3F),
}

def _droid_lerp(a, b, t):
    return int(round(a * (1 - t) + b * t))

def _droid_bar_color(p):
    p = max(0.0, min(1.0, p))
    if p < 0.5:
        t = p / 0.5
        r = _droid_lerp(_DROID_PALETTE["no"][0],   _DROID_PALETTE["warn"][0], t)
        g = _droid_lerp(_DROID_PALETTE["no"][1],   _DROID_PALETTE["warn"][1], t)
        b = _droid_lerp(_DROID_PALETTE["no"][2],   _DROID_PALETTE["warn"][2], t)
    else:
        t = (p - 0.5) / 0.5
        r = _droid_lerp(_DROID_PALETTE["warn"][0], _DROID_PALETTE["ok"][0],   t)
        g = _droid_lerp(_DROID_PALETTE["warn"][1], _DROID_PALETTE["ok"][1],   t)
        b = _droid_lerp(_DROID_PALETTE["warn"][2], _DROID_PALETTE["ok"][2],   t)
    return f"rgb({r},{g},{b})"

_DROID_LOGO_SVG = """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <g transform="translate(100,100) scale(2)">
    <path d="M 2.6948e-15 -44 L 38.1051 -22 L 38.1051 22 L 7.0816e-15 44 L -38.1051 22 L -38.1051 -22 Z"
      fill="none" stroke="#3a3530" stroke-width="0.9" stroke-dasharray="2 3" opacity="0.5"/>
    <path d="M 2.2046e-15 -36 L 31.1769 -18 L 31.1769 18 L 5.7941e-15 36 L -31.1769 18 L -31.1769 -18 Z"
      fill="#3a3530"/>
    <path d="M 1.4697e-15 -24 L 20.7846 -12 L 20.7846 12 L 3.8627e-15 24 L -20.7846 12 L -20.7846 -12 Z"
      fill="none" stroke="#f4ede0" stroke-width="1.2"/>
    <circle cx="0" cy="0" r="8" fill="#C58A2E"/>
    <circle cx="0" cy="0" r="3" fill="#f4ede0"/>
    <circle cx="0" cy="-36" r="3.4" fill="#f4ede0"/>
    <circle cx="31.1769" cy="-18" r="3.4" fill="#f4ede0"/>
    <circle cx="31.1769" cy="18" r="3.4" fill="#f4ede0"/>
    <circle cx="0" cy="36" r="3.4" fill="#f4ede0"/>
    <circle cx="-31.1769" cy="18" r="3.4" fill="#f4ede0"/>
    <circle cx="-31.1769" cy="-18" r="3.4" fill="#f4ede0"/>
    <circle cx="0" cy="-44" r="2.2" fill="#C58A2E"/>
    <circle cx="38.1051" cy="-22" r="2.2" fill="#C58A2E"/>
    <circle cx="38.1051" cy="22" r="2.2" fill="#C58A2E"/>
    <circle cx="0" cy="44" r="2.2" fill="#C58A2E"/>
    <circle cx="-38.1051" cy="22" r="2.2" fill="#C58A2E"/>
    <circle cx="-38.1051" cy="-22" r="2.2" fill="#C58A2E"/>
  </g>
</svg>"""

# Filtre multi-actifs (intersection score >= 0.90, max 5 activités cochées)
_DASHBOARD_FILTER_JS = r"""
function setupActivityFilters() {
  // Filtre DataTables global : lit l'état des chips de la table en cours
  $.fn.dataTable.ext.search.push(function(settings, searchData, dataIndex, rowData) {
    var box = document.querySelector('[data-filter-for="' + settings.sTableId + '"]');
    if (!box) return true;
    var checked = box.querySelectorAll('input.act-filter:checked');
    if (checked.length === 0) return true;
    for (var i = 0; i < checked.length; i++) {
      var di = parseInt(checked[i].dataset.di, 10);
      var v = rowData[di];
      if (v == null || v < 0.90) return false;
    }
    return true;
  });
  document.querySelectorAll('.activity-filter').forEach(function(box) {
    var tid = box.dataset.filterFor;
    var dt = jQuery('#' + tid).DataTable();
    var checks = box.querySelectorAll('input.act-filter');
    var chips = box.querySelectorAll('.af-chip');
    var countEl = box.querySelector('.af-count');
    var resetBtn = box.querySelector('.af-reset');
    function refresh() {
      var n = 0;
      checks.forEach(function(c){ if (c.checked) n++; });
      checks.forEach(function(c, i){
        var chip = chips[i];
        chip.classList.toggle('checked', c.checked);
        var locked = !c.checked && n >= 5;
        c.disabled = locked;
        chip.classList.toggle('locked', locked);
      });
      if (countEl) countEl.textContent = n + '/5';
      dt.draw();
    }
    checks.forEach(function(c){ c.addEventListener('change', refresh); });
    if (resetBtn) {
      resetBtn.addEventListener('click', function(){
        checks.forEach(function(c){ c.checked = false; });
        refresh();
      });
    }
  });
}
"""

# ── Radar multi-activités par peptide (SVG maison, zéro dépendance) ──────────
# Glyphe radar défini une seule fois (<symbol>), référencé par <use> dans chaque ligne (≈45 o/ligne au lieu de ≈620 o)
_RADAR_SYMBOL = (
    '<svg width="0" height="0" style="position:absolute" aria-hidden="true">'
    '<symbol id="proteogen-radar-ic" viewBox="0 0 24 24">'
    '<polygon points="12,3 20,9 17,19 7,19 4,9" fill="none" stroke="currentColor" stroke-width="1.4"/>'
    '<polygon points="12,7.5 16.5,10.8 15,16 9,16 7.5,10.8" fill="none" stroke="currentColor" stroke-width="1" opacity="0.6"/>'
    '<line x1="12" y1="12" x2="12" y2="3" stroke="currentColor" stroke-width="0.9" opacity="0.5"/>'
    '<line x1="12" y1="12" x2="20" y2="9" stroke="currentColor" stroke-width="0.9" opacity="0.5"/>'
    '<line x1="12" y1="12" x2="4" y2="9" stroke="currentColor" stroke-width="0.9" opacity="0.5"/>'
    '<circle cx="12" cy="12" r="1.3" fill="currentColor"/></symbol></svg>'
)
_RADAR_GLYPH = '<svg class="ri-ic" aria-hidden="true"><use href="#proteogen-radar-ic"/></svg>'

_RADAR_CSS = r"""
.pep-cell{ display:flex; align-items:center; gap:8px; }
.pep-name{ font-variant-ligatures:none; }
.ri-ic{ width:15px; height:15px; }
.radar-btn{
  flex:0 0 auto; display:inline-flex; align-items:center; justify-content:center;
  width:24px; height:24px; padding:0; border:1px solid var(--ink-faint,#8a8a8a);
  border-radius:6px; background:transparent; color:var(--ink-soft,#555);
  cursor:pointer; line-height:0; transition:all .12s ease;
}
.radar-btn:hover{ color:var(--accent,#B0473F); border-color:var(--accent,#B0473F);
  background:rgba(176,71,63,.06); }
.radar-btn:focus-visible{ outline:2px solid var(--accent,#B0473F); outline-offset:1px; }
.radar-modal{
  position:fixed; inset:0; z-index:9999; display:none;
  align-items:center; justify-content:center;
  background:rgba(26,26,26,.55); padding:24px;
}
.radar-modal.open{ display:flex; }
.radar-modal-card{
  background:var(--paper,#fdfcfa); color:var(--ink,#1a1a1a);
  border:1px solid var(--ink-faint,#8a8a8a); border-radius:14px;
  box-shadow:0 18px 50px rgba(0,0,0,.28); width:min(520px,94vw);
  max-height:92vh; overflow:auto; padding:18px 20px 14px;
}
.radar-modal-head{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.radar-modal-kicker{ font-size:11px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--ink-faint,#8a8a8a); }
.radar-title{ font-weight:700; font-size:16px; color:var(--accent,#B0473F);
  margin-right:auto; word-break:break-all; }
.radar-close{ flex:0 0 auto; width:30px; height:30px; border:none; border-radius:8px;
  background:transparent; color:var(--ink-soft,#555); font-size:22px; line-height:1; cursor:pointer; }
.radar-close:hover{ background:rgba(0,0,0,.06); color:var(--ink,#1a1a1a); }
.radar-canvas{ display:flex; justify-content:center; }
.radar-svg{ width:100%; max-width:460px; height:auto; }
.radar-grid{ fill:none; stroke:var(--ink-faint,#8a8a8a); stroke-opacity:.35; stroke-width:1; }
.radar-axis{ stroke:var(--ink-faint,#8a8a8a); stroke-opacity:.4; stroke-width:1; }
.radar-label{ font-size:10px; fill:var(--ink-soft,#555); font-family:inherit; }
.radar-area{ fill:rgba(176,71,63,.18); stroke:var(--accent,#B0473F); stroke-width:1.8; stroke-linejoin:round; }
.radar-dot{ fill:var(--accent,#B0473F); }
.radar-dot.hi{ fill:#fff; stroke:var(--accent,#B0473F); stroke-width:2; }
.radar-foot{ margin-top:8px; font-size:11px; color:var(--ink-faint,#8a8a8a); text-align:center; }
"""

_RADAR_MODAL = r"""
<div id="radarModal" class="radar-modal" role="dialog" aria-modal="true" aria-label="Radar multi-activités">
  <div class="radar-modal-card">
    <div class="radar-modal-head">
      <span class="radar-modal-kicker">Profil multi-activités</span>
      <span class="radar-title"></span>
      <button type="button" class="radar-close" aria-label="Fermer">&times;</button>
    </div>
    <div class="radar-canvas"></div>
    <div class="radar-foot">Rayon = probabilité calibrée (centre 0 → bord 1). Points pleins = score ≥ 0.90.</div>
  </div>
</div>
"""

_RADAR_JS = r"""
function _radarEsc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function _radarBuildSVG(labels, values, unc){
  var N = labels.length, size = 460, cx = size/2, cy = size/2, R = 165, rings = 4;
  function pt(i, r){ var a = -Math.PI/2 + i*2*Math.PI/N; return [cx + r*Math.cos(a), cy + r*Math.sin(a)]; }
  var parts = ['<svg viewBox="0 0 '+size+' '+size+'" xmlns="http://www.w3.org/2000/svg" class="radar-svg">'];
  for (var k=1; k<=rings; k++){
    var rr = R*k/rings, poly = [];
    for (var i=0;i<N;i++){ var p=pt(i,rr); poly.push(p[0].toFixed(1)+','+p[1].toFixed(1)); }
    parts.push('<polygon class="radar-grid" points="'+poly.join(' ')+'"/>');
  }
  for (var i=0;i<N;i++){
    var pe = pt(i, R);
    parts.push('<line class="radar-axis" x1="'+cx+'" y1="'+cy+'" x2="'+pe[0].toFixed(1)+'" y2="'+pe[1].toFixed(1)+'"/>');
    var pl = pt(i, R+18), anchor = (Math.abs(pl[0]-cx) < 6) ? 'middle' : (pl[0] > cx ? 'start' : 'end');
    parts.push('<text class="radar-label" x="'+pl[0].toFixed(1)+'" y="'+pl[1].toFixed(1)+'" text-anchor="'+anchor+'">'+_radarEsc(labels[i])+'</text>');
  }
  var vpoly = [], dots = [];
  for (var i=0;i<N;i++){
    var raw = values[i], v = (raw==null || isNaN(raw)) ? 0 : Math.max(0, Math.min(1, raw));
    var p = pt(i, R*v); vpoly.push(p[0].toFixed(1)+','+p[1].toFixed(1));
    var hi = (raw!=null) && (raw >= 0.90);
    var u = (unc && unc[i]!=null) ? ' \u00b1'+Number(unc[i]).toFixed(3) : '';
    var lbl = _radarEsc(labels[i])+' : '+(raw==null?'N/A':Number(raw).toFixed(3))+u;
    dots.push('<circle class="radar-dot'+(hi?' hi':'')+'" cx="'+p[0].toFixed(1)+'" cy="'+p[1].toFixed(1)+'" r="'+(hi?4:3)+'"><title>'+lbl+'</title></circle>');
  }
  parts.push('<polygon class="radar-area" points="'+vpoly.join(' ')+'"/>');
  parts.push(dots.join(''));
  parts.push('</svg>');
  return parts.join('');
}
function _radarOpen(btn){
  var tr = btn.closest('tr');
  if (!tr) return;
  var dt = $(tr).closest('table').DataTable();
  var rowData = dt.row(tr).data();
  if (!rowData) return;
  var values = [];
  for (var i = 0; i < RADAR_LABELS.length; i++){
    var v = rowData[ACT_DATA_OFFSET + 2 * i];
    values.push((v == null || isNaN(v)) ? null : v);
  }
  var modal = document.getElementById('radarModal');
  if (!modal) return;
  modal.querySelector('.radar-title').textContent = rowData[0];
  modal.querySelector('.radar-canvas').innerHTML = _radarBuildSVG(RADAR_LABELS, values, null);
  modal.classList.add('open');
}
function _radarClose(){ var m = document.getElementById('radarModal'); if (m) m.classList.remove('open'); }
function setupRadar(){
  document.addEventListener('click', function(e){
    var btn = e.target.closest ? e.target.closest('.radar-btn') : null;
    if (btn){ e.preventDefault(); _radarOpen(btn); return; }
    if (e.target.matches && (e.target.matches('.radar-modal') || e.target.matches('.radar-close'))) _radarClose();
  });
  document.addEventListener('keydown', function(e){ if (e.key === 'Escape') _radarClose(); });
}
"""


# ── Rendu client des cellules (DataTables columns.render) — miroir JS de _bar_html ──────────
_DASHBOARD_RENDER_JS = r"""
var _RADAR_GLYPH_HTML = '<svg class="ri-ic" aria-hidden="true"><use href="#proteogen-radar-ic"/></svg>';
function _esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function _pepCell(pep){
  return '<span class="pep-cell"><button type="button" class="radar-btn" title="Profil radar multi-activités" aria-label="Profil radar">'
    + _RADAR_GLYPH_HTML + '</button><span class="pep-name">' + _esc(pep) + '</span></span>';
}
function _confClass(u){
  if (u == null || isNaN(u)) return 'cf-na';
  if (u < 0.05) return 'cf-h';
  if (u < 0.15) return 'cf-m';
  return 'cf-l';
}
function _barCell(p, u){
  if (p == null || isNaN(p)) return '<span class="bar-na">N/A</span>';
  var k = (p >= 1) ? 9 : Math.max(0, Math.floor(p * 10));
  var width = Math.max(2, Math.round(p * 200));
  var us = (u == null || isNaN(u)) ? '' : ' \u00b1' + Number(u).toFixed(3);
  return '<div class="bar-cell"><div class="bar-wrap"><div class="bar-fill bf' + k + '" style="width:' + width + '%"></div></div>'
    + '<span class="bar-val">' + Number(p).toFixed(3) + us + '</span>'
    + '<span class="badge ' + _confClass(u) + '">\u25cf</span></div>';
}
function buildColumns(){
  var cols = [{ data: 0, render: function(d, t, row){ return (t === 'display') ? _pepCell(d) : d; } }];
  if (HAS_ACC) cols.push({ data: 1 });
  for (var i = 0; i < RADAR_LABELS.length; i++){
    (function(di, ui){
      cols.push({ data: di, render: function(d, t, row){
        if (t === 'sort' || t === 'type') return (d == null ? -1 : d);
        if (t === 'filter') return d;
        return _barCell(d, row[ui]);
      }});
    })(ACT_DATA_OFFSET + 2 * i, ACT_DATA_OFFSET + 2 * i + 1);
  }
  return cols;
}
"""


def _generate_client_dashboard(df_global, model_a_tester, output_dir, graphs_id, date_run):
    """
    Dashboard HTML standalone : onglets par feuille, tableau filtrable
    (DataTables CDN), barres proba colorées ± incertitude, badge confiance
    MC Dropout (vert/orange/rouge). Aucune dépendance serveur.
    """
    import json
    import html as _html
    activities = [m["nom_colonne"] for m in model_a_tester]
    prob_cols  = [f"Peptide_{a}" for a in activities]
    keep = [(a, pc, f"Incertitude_{a}") for a, pc in zip(activities, prob_cols) if pc in df_global.columns]
    activities = [k[0] for k in keep]
    prob_cols  = [k[1] for k in keep]
    unc_cols   = [k[2] for k in keep]
    if "Sheet" in df_global.columns:
        sheets = [s for s in df_global["Sheet"].dropna().unique().tolist()]
    else:
        sheets = ["All"]
        df_global = df_global.assign(Sheet="All")
    n_pep    = len(df_global)
    n_act    = len(activities)
    n_sheets = len(sheets)
    radar_labels_json = json.dumps(activities, ensure_ascii=False)
    def _bar_class(unc):
        if pd.isna(unc):
            return "cf-na"
        if unc < 0.05:
            return "cf-h"
        if unc < 0.15:
            return "cf-m"
        return "cf-l"
    def _bar_html(prob, unc):
        if pd.isna(prob):
            return '<span class="bar-na">N/A</span>'
        p = float(prob)
        k = 9 if p >= 1 else max(0, int(p * 10))   # bucket couleur -> classe CSS (.bf0..bf9)
        width = max(2, int(round(p * 200)))
        unc_str = f" ±{float(unc):.3f}" if not pd.isna(unc) else ""
        return (
            f'<div class="bar-cell">'
            f'<div class="bar-wrap"><div class="bar-fill bf{k}" style="width:{width}%"></div></div>'
            f'<span class="bar-val">{p:.3f}{unc_str}</span>'
            f'<span class="badge {_bar_class(unc)}">●</span>'
            f'</div>'
        )
    sheet_tables_html = []
    table_ids = []
    sheet_data_js = []
    has_accession = "Accession" in df_global.columns
    act_col_offset = 1 + (1 if has_accession else 0)
    bar_color_css = "".join(f".bf{k}{{background:{_droid_bar_color((k + 0.5) / 10)};}}" for k in range(10))
    for s in sheets:
        df_s = df_global[df_global["Sheet"] == s]
        # Rendu piloté par données : aucune ligne <tr> émise. Tableau JS compact de nombres bruts,
        # DataTables construit les cellules visibles via columns.render (deferRender).
        tid_data = "table_" + re.sub(r'[^A-Za-z0-9]', '_', str(s))
        data_rows = []
        for _, row in df_s.iterrows():
            rec: list[str | None] = [str(row.get("Peptide", ""))]
            if has_accession:
                rec.append(str(row.get("Accession", "")))
            for pc, uc in zip(prob_cols, unc_cols):
                p = row.get(pc, np.nan)
                u = row.get(uc, np.nan) if uc in df_s.columns else np.nan
                rec.append(None if pd.isna(p) else str(round(float(p), 5)))
                rec.append(None if pd.isna(u) else str(round(float(u), 5)))
            data_rows.append(rec)
        sheet_data_js.append(f'SHEET_DATA["{tid_data}"] = {json.dumps(data_rows, ensure_ascii=False)};')
        header_cells = ['<th>Peptide</th>']
        if has_accession:
            header_cells.append('<th>Accession</th>')
        header_cells += [f'<th>{a}</th>' for a in activities]
        tid = "table_" + re.sub(r'[^A-Za-z0-9]', '_', str(s))
        table_ids.append(tid)
        chip_html = "".join(
            f'<label class="af-chip"><input type="checkbox" class="act-filter" '
            f'data-di="{act_col_offset + 2 * i}" value="{a}"> <span>{a}</span></label>'
            for i, a in enumerate(activities)
        )
        act_filter_html = (
            f'<div class="activity-filter" data-filter-for="{tid}">'
            f'<span class="af-label">Multi-actifs · score ≥ 0.90</span>'
            f'<span class="af-count">0/5</span>'
            f'<button type="button" class="af-reset">décocher tout</button>'
            f'{chip_html}'
            f'</div>'
        ) if activities else ''
        sheet_tables_html.append(
            f'<div class="tab-content" id="tab_{tid}" style="display:none">'
            f'<h3>Feuille « {s} » — {len(df_s)} peptides</h3>'
            f'{act_filter_html}'
            f'<table id="{tid}" class="display nowrap" style="width:100%">'
            f'<thead><tr>{"".join(header_cells)}</tr></thead>'
            f'<tbody></tbody>'
            f'</table></div>'
        )
    tabs_buttons = "".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" data-target="tab_{tid}">{s}</button>'
        for i, (s, tid) in enumerate(zip(sheets, table_ids))
    )
    dt_init_js = "\n      ".join(
        f"$('#{tid}').DataTable({{data: SHEET_DATA['{tid}'], columns: buildColumns(), "
        f"pageLength: 25, scrollX: true, order: [], deferRender: true}});"
        for tid in table_ids
    )
    sheet_data_block = "\n".join(sheet_data_js)
    has_acc_js = "true" if has_accession else "false"
    first_tab_js = (
        f'document.getElementById("tab_{table_ids[0]}").style.display = "block";'
        if table_ids else ""
    )
    out_path = os.path.join(output_dir, f"PROTEOGEN_Dashboard_{graphs_id}_{date_run}.html")
    dashboard_fname = f"PROTEOGEN_Dashboard_{graphs_id}_{date_run}.html"
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>DROID · PROTEOGEN Dashboard — {graphs_id}</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<style>
/* ============================================================
   Dashboard PROTEOGEN — style DROID
   Inspiré de figures/shared.css (rapport DROID, Peptidomique)
   ============================================================ */
:root{{
  --c-ok:    #4F8A4F;
  --c-no:    #B0473F;
  --c-warn:  #C58A2E;
  --c-info:  #4A6FA5;

  --ink:       #1a1a1a;
  --ink-soft:  #555;
  --ink-faint: #8a8a8a;
  --ink-ghost: #b7b3aa;

  --paper:     #fdfcfa;
  --paper-2:   #f4ede0;
  --paper-3:   #ebe5d6;
  --rule:      #d8d4cc;
  --rule-soft: #e8e3d8;

  --font-sans: "Calibri","Carlito","Trebuchet MS",system-ui,sans-serif;
  --font-mono: "JetBrains Mono","SFMono-Regular",Consolas,monospace;
  --font-serif:"Cambria","Source Serif Pro",Georgia,serif;
}}
*{{ box-sizing:border-box; }}
html,body{{
  margin:0; padding:0;
  background:#eceae4;
  font-family:var(--font-sans);
  color:var(--ink);
  font-size:14px;
}}
.stage{{
  display:flex; flex-direction:column; align-items:center; gap:20px;
  padding:28px 16px 56px;
}}
.toolbar{{
  display:flex; gap:12px; align-items:center; flex-wrap:wrap;
  font-family:var(--font-mono); font-size:11px; color:var(--ink-soft);
  max-width:90vw;
}}
.toolbar .crumb{{ color:var(--ink-faint); }}
.sheet{{
  width:min(1240px, 96vw);
  background:var(--paper);
  box-shadow:0 1px 0 rgba(0,0,0,.04), 0 8px 24px rgba(0,0,0,.08);
  padding:32px 40px 28px;
  display:flex; flex-direction:column;
}}
.fig-header{{
  display:flex; justify-content:space-between; align-items:flex-start;
  gap:24px;
  border-bottom:1px solid var(--rule);
  padding-bottom:18px; margin-bottom:22px;
}}
.fig-header-main{{ flex:1; min-width:0; }}
.fig-id{{
  font-family:var(--font-mono); font-size:11px;
  color:var(--ink-faint); letter-spacing:.08em; text-transform:uppercase;
}}
.fig-title{{
  font-size:22px; font-weight:600; margin:6px 0 4px;
  letter-spacing:.01em; line-height:1.2;
}}
.fig-sub{{
  font-size:13px; color:var(--ink-soft);
  max-width:78ch; line-height:1.5; margin:0;
}}
.fig-meta{{
  font-family:var(--font-mono); font-size:11px;
  color:var(--ink-faint); text-align:right; line-height:1.7;
  white-space:nowrap;
}}
.fig-header-logo{{
  flex-shrink:0;
  display:flex; flex-direction:column; align-items:center; gap:4px;
  margin-left:8px;
}}
.fig-header-logo svg{{ display:block; width:64px; height:64px; }}
.fig-header-logo-label{{
  font-family:var(--font-mono); font-size:8.5px;
  color:var(--ink-faint); letter-spacing:.12em; text-transform:uppercase;
}}
.summary{{
  display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
  gap:14px;
  margin-bottom:18px;
}}
.summary .card{{
  background:var(--paper-2);
  border:1px solid var(--rule);
  padding:14px 18px 12px;
  display:flex; flex-direction:column; gap:4px;
}}
.summary .card .v{{
  font-family:var(--font-sans);
  font-size:28px; font-weight:700; color:var(--ink);
  line-height:1.05; letter-spacing:.01em;
}}
.summary .card .l{{
  font-family:var(--font-mono); font-size:10.5px;
  color:var(--ink-faint); letter-spacing:.1em; text-transform:uppercase;
}}
.legend{{
  margin-bottom:14px;
  border-top:1px solid var(--rule-soft);
  border-bottom:1px solid var(--rule-soft);
  padding:10px 0;
  font-family:var(--font-mono); font-size:10.5px;
  color:var(--ink-faint); letter-spacing:.04em;
  display:flex; flex-wrap:wrap; align-items:center; gap:6px 18px;
}}
.legend .legend-label{{
  text-transform:uppercase; letter-spacing:.12em;
  color:var(--ink-soft);
}}
.legend .sep{{ color:var(--ink-ghost); }}
.legend .dot{{
  display:inline-flex; align-items:center; gap:6px;
  color:var(--ink-soft);
}}
.legend .dot::before{{
  content:""; display:inline-block; width:8px; height:8px; border-radius:50%;
  background:currentColor;
}}
.legend .dot.ok{{ color:var(--c-ok); }}
.legend .dot.warn{{ color:var(--c-warn); }}
.legend .dot.no{{ color:var(--c-no); }}
.legend .dot span{{ color:var(--ink-soft); }}
.tabs{{
  display:flex; gap:2px; flex-wrap:wrap;
  border-bottom:1px solid var(--rule);
  margin-bottom:18px;
}}
.tab-btn{{
  background:transparent; border:none;
  padding:10px 14px 9px;
  font-family:var(--font-mono); font-size:10.5px;
  color:var(--ink-faint); letter-spacing:.1em; text-transform:uppercase;
  cursor:pointer;
  border-bottom:2px solid transparent;
  margin-bottom:-1px;
}}
.tab-btn:hover{{ color:var(--ink); }}
.tab-btn.active{{
  color:var(--ink);
  border-bottom-color:var(--c-warn);
  font-weight:600;
}}
.tab-content{{ padding:6px 0 12px; }}
.tab-content h3{{
  font-family:var(--font-sans);
  font-size:13px; font-weight:600;
  color:var(--ink); margin:0 0 12px;
  padding-bottom:6px; border-bottom:1px dashed var(--rule);
  letter-spacing:.01em;
}}
.activity-filter{{
  display:flex; flex-wrap:wrap; align-items:center; gap:6px 10px;
  padding:8px 0 14px;
  font-family:var(--font-mono); font-size:10.5px;
  color:var(--ink-soft);
  border-bottom:1px dashed var(--rule);
  margin-bottom:12px;
}}
.activity-filter .af-label{{
  text-transform:uppercase; letter-spacing:.1em;
  color:var(--ink-faint);
}}
.activity-filter .af-count{{
  font-weight:600; color:var(--c-warn);
  font-family:var(--font-mono); letter-spacing:.05em;
  padding:2px 7px; background:var(--paper-2);
  border:1px solid var(--rule); border-radius:2px;
}}
.activity-filter .af-reset{{
  font:inherit; color:var(--ink-soft); background:transparent;
  border:1px solid var(--rule); border-radius:2px;
  padding:3px 9px; cursor:pointer;
  text-transform:uppercase; letter-spacing:.05em;
}}
.activity-filter .af-reset:hover{{
  background:var(--paper-2); color:var(--ink);
}}
.activity-filter .af-chip{{
  display:inline-flex; align-items:center; gap:5px;
  padding:3px 9px; border:1px solid var(--rule);
  background:var(--paper); border-radius:2px; cursor:pointer;
  user-select:none;
  transition:background .15s, border-color .15s, color .15s;
}}
.activity-filter .af-chip input{{
  accent-color:var(--c-warn);
  margin:0; cursor:pointer;
}}
.activity-filter .af-chip.checked{{
  background:var(--paper-3);
  border-color:var(--c-warn);
  color:var(--ink); font-weight:600;
}}
.activity-filter .af-chip.locked{{
  opacity:.35; cursor:not-allowed;
}}
.activity-filter .af-chip.locked input{{
  cursor:not-allowed;
}}
.bar-cell{{
  display:flex; align-items:center; gap:8px; min-width:170px;
}}
.bar-wrap{{
  background:var(--paper-3);
  height:10px; flex:1; border-radius:1px; overflow:hidden;
  min-width:60px;
  border:1px solid var(--rule-soft);
}}
.bar-fill{{ height:100%; transition:width .2s; }}
.bar-val{{
  font-family:var(--font-mono); font-size:11px;
  color:var(--ink-soft); white-space:nowrap;
}}
.badge{{ font-size:13px; line-height:1; }}
.badge.cf-h{{ color:#4F8A4F; }} .badge.cf-m{{ color:#C58A2E; }} .badge.cf-l{{ color:#B0473F; }} .badge.cf-na{{ color:#9ca3af; }}
.bar-na{{ color:#b7b3aa; }}
{bar_color_css}
table.dataTable{{
  border-collapse:collapse !important;
  width:100% !important;
}}
table.dataTable thead th{{
  background:var(--paper-2) !important;
  border-bottom:1px solid var(--rule) !important;
  border-top:1px solid var(--rule) !important;
  font-family:var(--font-mono);
  font-size:10.5px; font-weight:600;
  color:var(--ink-soft);
  letter-spacing:.08em; text-transform:uppercase;
  padding:8px 10px !important;
}}
table.dataTable tbody td{{
  padding:6px 10px !important;
  font-family:var(--font-sans);
  font-size:12.5px; color:var(--ink);
  border-bottom:1px solid var(--rule-soft) !important;
  vertical-align:middle;
}}
table.dataTable tbody td:first-child{{
  font-family:var(--font-mono);
  font-size:11.5px;
  color:var(--ink);
}}
table.dataTable tbody td:nth-child(2){{
  font-family:var(--font-mono);
  font-size:11px;
  color:var(--ink-faint);
}}
table.dataTable tbody tr:hover td{{
  background:#f8f4ea !important;
}}
.dataTables_wrapper{{
  font-family:var(--font-mono); font-size:11px; color:var(--ink-soft);
}}
.dataTables_wrapper .dataTables_filter,
.dataTables_wrapper .dataTables_length{{
  margin-bottom:10px;
}}
.dataTables_wrapper .dataTables_filter label,
.dataTables_wrapper .dataTables_length label{{
  color:var(--ink-faint); letter-spacing:.04em;
  text-transform:uppercase; font-size:10px;
}}
.dataTables_wrapper .dataTables_filter input,
.dataTables_wrapper .dataTables_length select{{
  font-family:var(--font-mono); font-size:11px;
  color:var(--ink);
  border:1px solid var(--rule); border-radius:2px;
  padding:4px 8px; margin-left:6px;
  background:#fff;
}}
.dataTables_wrapper .dataTables_filter input:focus,
.dataTables_wrapper .dataTables_length select:focus{{
  outline:none; border-color:var(--c-warn);
}}
.dataTables_wrapper .dataTables_info{{
  color:var(--ink-faint); padding-top:14px;
  letter-spacing:.04em; text-transform:uppercase; font-size:10px;
}}
.dataTables_wrapper .dataTables_paginate{{
  padding-top:10px;
}}
.dataTables_wrapper .dataTables_paginate .paginate_button{{
  font-family:var(--font-mono); font-size:11px !important;
  color:var(--ink-soft) !important;
  border:1px solid transparent !important; border-radius:2px !important;
  padding:4px 9px !important; margin:0 1px !important;
  background:transparent !important;
}}
.dataTables_wrapper .dataTables_paginate .paginate_button:hover{{
  background:var(--paper-2) !important;
  border-color:var(--rule) !important;
  color:var(--ink) !important;
}}
.dataTables_wrapper .dataTables_paginate .paginate_button.current,
.dataTables_wrapper .dataTables_paginate .paginate_button.current:hover{{
  background:var(--ink) !important;
  border-color:var(--ink) !important;
  color:var(--paper-2) !important;
}}
.fig-footer{{
  margin-top:24px;
  display:flex; justify-content:space-between; align-items:flex-end;
  font-family:var(--font-mono); font-size:10.5px; color:var(--ink-faint);
  border-top:1px solid var(--rule); padding-top:10px;
  letter-spacing:.04em;
}}
.fig-footer .filename{{ letter-spacing:.06em; }}
@media print{{
  html,body{{ background:#fff; }}
  .stage{{ padding:0; gap:0; }}
  .toolbar{{ display:none; }}
  .sheet{{ box-shadow:none; }}
}}
{_RADAR_CSS}
</style>
</head>
<body>
<div class="stage">
  <div class="sheet">
    <header class="fig-header">
      <div class="fig-header-main">
        <div class="fig-id">Dashboard — sortie pipeline · activités biologiques prédites</div>
        <h1 class="fig-title">PROTEOGEN — Dashboard activités biologiques</h1>
        <p class="fig-sub">Probabilités d'activité biologique par peptide, calibrées via Platt et accompagnées de l'incertitude MC&nbsp;Dropout (σ). Tri par colonne, recherche plein texte et filtrage par feuille d'entrée.</p>
      </div>
      <div class="fig-meta">Run · {graphs_id}<br/>Exécution · {date_run}<br/>PROTEOGEN · pipeline<br/>v1 — sortie ARDF</div>
      <div class="fig-header-logo">
        {_DROID_LOGO_SVG}
        <div class="fig-header-logo-label">DROID</div>
      </div>
    </header>
    <section class="summary">
      <div class="card"><div class="v">{n_pep}</div><div class="l">Peptides</div></div>
      <div class="card"><div class="v">{n_act}</div><div class="l">Activités prédites</div></div>
      <div class="card"><div class="v">{n_sheets}</div><div class="l">Feuilles</div></div>
    </section>
    <div class="legend">
      <span class="legend-label">Confiance MC Dropout</span>
      <span class="sep">·</span>
      <span class="dot ok"><span>haute (σ &lt; 0.05)</span></span>
      <span class="dot warn"><span>moyenne (0.05 – 0.15)</span></span>
      <span class="dot no"><span>faible (σ ≥ 0.15)</span></span>
      <span class="sep">·</span>
      <span class="legend-label">Tri colonne</span><span>clic en-tête</span>
      <span class="sep">·</span>
      <span class="legend-label">Recherche</span><span>champ en haut à droite</span>
    </div>
    <div class="tabs">{tabs_buttons}</div>
    {"".join(sheet_tables_html)}
    {_RADAR_SYMBOL}
    {_RADAR_MODAL}
    <footer class="fig-footer">
      <span>DROID Team · Peptidomique · sortie pipeline ARDF</span>
      <span class="filename">{dashboard_fname}</span>
    </footer>
  </div>
</div>
<script>
var RADAR_LABELS = {radar_labels_json};
var ACT_DATA_OFFSET = {act_col_offset};
var HAS_ACC = {has_acc_js};
var SHEET_DATA = {{}};
{sheet_data_block}
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
    btn.classList.add('active');
    document.getElementById(btn.dataset.target).style.display = 'block';
  }});
}});
{first_tab_js}
{_DASHBOARD_RENDER_JS}
{_DASHBOARD_FILTER_JS}
{_RADAR_JS}
$(document).ready(function() {{
      {dt_init_js}
      setupActivityFilters();
      setupRadar();
}});
</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
# ============================================================
# UTILITAIRES — INSPECTION FICHIER XLSX (amont pipeline)
# ============================================================
def get_file_info(file_path: str) -> dict:
    """
    Inspecte un fichier Excel sans lancer la pipeline (aucune dépendance ESM/Keras).
    Détecte feuilles '*_Peptides', colonne 'Peptide', compte séquences valides,
    repère colonnes 'Peptide_*'/'Incertitude_*' (= fichier déjà traité, candidat reload).
    Args:
        file_path : Chemin vers fichier Excel (.xlsx ou .xls).
    Returns:
        dict {
            file_path, file_name, file_size_mb, format,
            n_sheets, sheet_names,
            peptide_sheets : list[{name, n_peptides, n_empty, has_peptide_col, peptide_col_name, columns}],
            other_sheets   : list[{name, n_rows, columns}],
            total_peptides, already_processed,
            detected_activities, warnings, error
        }
    """
    info = {
        "file_path": file_path, "file_name": None, "file_size_mb": None, "format": None,
        "n_sheets": 0, "sheet_names": [],
        "peptide_sheets": [], "other_sheets": [],
        "total_peptides": 0, "already_processed": False,
        "detected_activities": [], "warnings": [], "error": None,
    }
    if not file_path or not isinstance(file_path, str):
        info["error"] = "Chemin de fichier vide ou invalide."
        return info
    if not os.path.exists(file_path):
        info["error"] = f"Fichier introuvable : {file_path}"
        return info
    info["file_name"]   = os.path.basename(file_path)
    info["file_size_mb"] = round(os.path.getsize(file_path) / (1024 * 1024), 3)
    ext = os.path.splitext(file_path)[1].lower()
    info["format"] = ext.lstrip(".")
    if ext not in (".xlsx", ".xls"):
        info["error"] = f"Format non supporté : '{ext}'. Attendu : .xlsx ou .xls."
        return info
    try:
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        info["error"] = f"Erreur ouverture Excel : {e}"
        return info
    info["n_sheets"]    = len(xls.sheet_names)
    info["sheet_names"] = list(xls.sheet_names)
    if info["n_sheets"] == 0:
        info["warnings"].append("Fichier Excel sans aucune feuille.")
        return info
    peptide_sheet_names = [s for s in xls.sheet_names if "_Peptides" in str(s)]
    if not peptide_sheet_names:
        info["warnings"].append(
            "Aucune feuille avec suffixe '_Peptides'. La pipeline traitera toutes les feuilles."
        )
        peptide_sheet_names = list(xls.sheet_names)
    detected_acts = set()
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, na_filter=True)
        except Exception as e:
            info["warnings"].append(f"Feuille '{sheet_name}' illisible : {e}")
            continue
        cols = list(df.columns)
        peptide_col = next((c for c in cols if str(c).strip().lower() == "peptide"), None)
        pred_cols = [c for c in cols if str(c).startswith("Peptide_")]
        if pred_cols:
            info["already_processed"] = True
            detected_acts.update(str(c).replace("Peptide_", "", 1) for c in pred_cols)
        if sheet_name in peptide_sheet_names:
            if peptide_col is not None:
                series = df[peptide_col].astype(str).str.strip()
                n_empty    = int(series.isin(["", "nan", "None"]).sum())
                n_peptides = int(len(series) - n_empty)
            else:
                n_empty, n_peptides = 0, 0
                info["warnings"].append(
                    f"Feuille '{sheet_name}' sans colonne 'Peptide' — pipeline incompatible."
                )
            info["peptide_sheets"].append({
                "name": sheet_name,
                "n_peptides":      n_peptides,
                "n_empty":         n_empty,
                "has_peptide_col": peptide_col is not None,
                "peptide_col_name": peptide_col,
                "columns": cols,
            })
            info["total_peptides"] += n_peptides
        else:
            info["other_sheets"].append({
                "name": sheet_name, "n_rows": int(len(df)), "columns": cols,
            })
    info["detected_activities"] = sorted(detected_acts)
    if info["total_peptides"] == 0 and info["peptide_sheets"]:
        info["warnings"].append("Aucun peptide trouvé dans les feuilles candidates.")
    return info
# ============================================================
# UTILITAIRES — GESTION DES RUNS
# ============================================================
def _resolve_run_path(run_id: str) -> str:
    """
    Résout un run_id en chemin XLSX. Convention: run_id = nom du fichier sans extension.
    Cherche dans CWD avec/sans .xlsx, et avec préfixe 'PROTEOGEN_Results_' si absent.
    Retourne le chemin si trouvé, sinon str vide.
    """
    if not run_id:
        return ""
    candidates = [
        run_id if run_id.endswith(".xlsx") else f"{run_id}.xlsx",
        run_id if run_id.startswith("PROTEOGEN_Results_") else f"PROTEOGEN_Results_{run_id}.xlsx",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return ""

def load_run(run_id: str) -> dict:
    """
    Recharge un run PROTEOGEN précédent à partir de son XLSX résultat.
    Reconstruit un dict compatible avec display_pipeline_results (output_excel, html_files,
    log_file, summary) et l'enrichit d'un inventaire (activités prédites, peptides, feuilles).
    Args:
        run_id : nom du XLSX résultat (avec ou sans .xlsx, avec ou sans préfixe 'PROTEOGEN_Results_').
    Returns:
        dict {
            output_excel, html_files, log_file, summary,
            run_id, run_path, base_id,
            n_sheets, sheet_names,
            peptide_sheets : list[{name, n_peptides, columns}],
            n_peptides, detected_activities,
            warnings, error
        }
    """
    result = {
        "output_excel": None, "html_files": [], "log_file": None, "summary": "",
        "run_id": run_id, "run_path": None, "base_id": None,
        "dashboard_html": None,
        "n_sheets": 0, "sheet_names": [], "peptide_sheets": [],
        "n_peptides": 0, "detected_activities": [],
        "warnings": [], "error": None,
    }
    if not run_id or not isinstance(run_id, str):
        result["error"] = "run_id vide ou invalide."
        return result
    run_path = _resolve_run_path(run_id)
    if not run_path:
        result["error"] = f"Run introuvable : '{run_id}'. Attendu dans le CWD."
        return result
    result["run_path"] = run_path
    result["output_excel"] = run_path
    base = os.path.splitext(os.path.basename(run_path))[0].replace("PROTEOGEN_Results_", "", 1)
    result["base_id"] = base
    dashboard_candidate = f"PROTEOGEN_Dashboard_{base}.html"
    if os.path.exists(dashboard_candidate):
        result["dashboard_html"] = os.path.abspath(dashboard_candidate)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", base)
    if date_match:
        log_path = os.path.join(DIR_LOGS, f"Prediction_1280D_Run_{date_match.group(1)}.log")
        if os.path.exists(log_path):
            result["log_file"] = log_path
        else:
            result["warnings"].append(f"Log absent : {log_path}")
    else:
        result["warnings"].append("Date introuvable dans le nom du run — log non résolu.")
    try:
        xls = pd.ExcelFile(run_path)
    except Exception as e:
        result["error"] = f"Lecture XLSX impossible : {e}"
        return result
    result["n_sheets"] = len(xls.sheet_names)
    result["sheet_names"] = list(xls.sheet_names)
    # HTML nommés via graphs_id (input basename, sans date), sheet_name, ou activité.
    # base (avec date suffixe) ne match aucun. Tokens élargis : base, base_short, sheets.
    base_short = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", "", base)
    html_tokens = {t for t in {base, base_short, *xls.sheet_names} if t and len(t) >= 4}
    html_files = []
    for search_dir in [DIR_GRAPHS, DIR_CLUSTER]:
        if os.path.isdir(search_dir):
            for fname in sorted(os.listdir(search_dir)):
                if fname.endswith(".html") and any(tok in fname for tok in html_tokens):
                    html_files.append(os.path.join(search_dir, fname))
    result["html_files"] = html_files
    detected_acts = set()
    total_pep = 0
    for s in xls.sheet_names:
        try:
            df = pd.read_excel(run_path, sheet_name=s, na_filter=True)
        except Exception as e:
            result["warnings"].append(f"Feuille '{s}' illisible : {e}")
            continue
        pred_cols = [c for c in df.columns if str(c).startswith("Peptide_")]
        detected_acts.update(str(c).replace("Peptide_", "", 1) for c in pred_cols)
        if "_Peptides" in str(s) and "Peptide" in df.columns:
            series = df["Peptide"].astype(str).str.strip()
            n_p = int(len(series) - series.isin(["", "nan", "None"]).sum())
            total_pep += n_p
            result["peptide_sheets"].append({
                "name": s, "n_peptides": n_p, "columns": list(df.columns),
            })
    result["detected_activities"] = sorted(detected_acts)
    result["n_peptides"] = total_pep
    if not result["detected_activities"]:
        result["warnings"].append("Aucune colonne 'Peptide_*' détectée — run potentiellement non traité.")
    result["summary"] = (
        f"Run '{base}' rechargé : {total_pep} peptide(s), "
        f"{len(result['detected_activities'])} activité(s) prédite(s), "
        f"{len(html_files)} graphique(s) HTML."
    )
    return result
# ============================================================
# CLUSTERING ISOLÉ — POST-PIPELINE (par feuille)
# ============================================================
def run_clustering(run_id: str, cutoff: float = 0.7) -> dict:
    """
    Lance le clustering PepFuNN + UMAP + Treemap PAR FEUILLE sur un run existant.
    Chaque feuille '*_Peptides' du XLSX est clusterisée indépendamment.
    Args:
        run_id : nom du XLSX résultat (avec ou sans .xlsx, avec ou sans préfixe PROTEOGEN_Results_).
        cutoff : seuil de similarité Tanimoto pour Butina (défaut 0.7).
    Returns:
        dict {
            run_id, run_path, cutoff,
            n_sequences (total uniques toutes feuilles),
            n_clusters (somme par feuille),
            clusters_per_sheet (dict sheet -> nb clusters),
            umap_html (None, legacy compat),
            umap_sheets_html (list[str] : UMAP par feuille),
            treemap_global_html (None, legacy compat),
            treemap_sheets_html (list[str] : treemap par feuille),
            master_xlsx, warnings: list, error: str|None
        }
    """
    result = {
        "run_id": run_id, "run_path": None, "cutoff": cutoff,
        "n_sequences": 0, "n_clusters": 0,
        "clusters_per_sheet": {},
        "umap_html": None, "umap_sheets_html": [],
        "treemap_global_html": None, "treemap_sheets_html": [],
        "master_xlsx": None, "warnings": [], "error": None,
    }
    run_path = _resolve_run_path(run_id)
    if not run_path:
        result["error"] = f"Run introuvable : '{run_id}'. Attendu dans le CWD."
        return result
    result["run_path"] = run_path
    if not (0.0 < cutoff < 1.0):
        result["error"] = f"cutoff invalide : {cutoff}. Attendu dans ]0, 1[."
        return result

    os.makedirs(DIR_CLUSTER, exist_ok=True)
    graphs_id = os.path.splitext(os.path.basename(run_path))[0].replace("PROTEOGEN_Results_", "")
    try:
        xls = pd.ExcelFile(run_path)
    except Exception as e:
        result["error"] = f"Lecture XLSX impossible : {e}"
        return result
    peptide_sheets = [s for s in xls.sheet_names if "_Peptides" in str(s)] or [
        s for s in xls.sheet_names if s != "Matériel_et_Méthodes"
    ]

    master_frames = []
    for sheet in peptide_sheets:
        try:
            df_s = pd.read_excel(run_path, sheet_name=sheet, na_filter=True)
        except Exception as e:
            result["warnings"].append(f"Feuille '{sheet}' lecture échouée : {e}")
            continue
        df_s["Sheet"] = sheet
        if "Peptide" not in df_s.columns:
            result["warnings"].append(f"Feuille '{sheet}' : colonne 'Peptide' absente, ignorée.")
            continue
        if "Seq_Clean" not in df_s.columns:
            df_s["Seq_Clean"] = (
                df_s["Peptide"].astype(str)
                .apply(lambda s: re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "",
                                        re.sub(r"\(.*?\)", "", s).upper()))
            )
        df_s = df_s[df_s["Seq_Clean"].str.len() > 0]
        df_s = df_s.drop_duplicates(subset=["Seq_Clean"]).reset_index(drop=True)
        if len(df_s) < 2:
            result["warnings"].append(f"Feuille '{sheet}' : < 2 séquences uniques, clustering sauté.")
            continue
        result["n_sequences"] += int(len(df_s))

        liste_acc = (
            df_s["Accession"].astype(str).tolist()
            if "Accession" in df_s.columns
            else [f"{sheet}_seq_{i}" for i in range(len(df_s))]
        )
        liste_seq = df_s["Seq_Clean"].tolist()

        try:
            sim_sheet = Clustering_PepFuNN_1280D_UMAP.simClustering(
                ids=liste_acc, sequences=liste_seq
            )
            sim_sheet.run_clustering(cutoff=cutoff)
            n_clu = len(sim_sheet.clusters)
            result["clusters_per_sheet"][sheet] = n_clu
            result["n_clusters"] += n_clu
            logging.info(f"[run_clustering] '{sheet}' : {n_clu} clusters @ cutoff={cutoff}")
        except Exception as e:
            result["warnings"].append(f"Clustering '{sheet}' échec : {e}")
            continue

        try:
            umap_path = os.path.join(DIR_CLUSTER, f"PepFuNN_UMAP-{sheet}-{graphs_id}.html")
            sim_sheet.plot_sim_space_plotly(add_cluster_color=True, out_name=umap_path)
            result["umap_sheets_html"].append(umap_path)
        except Exception as e:
            result["warnings"].append(f"UMAP '{sheet}' non généré : {e}")

        try:
            dict_cluster = {}
            for cid, indices in enumerate(sim_sheet.clusters):
                cname = f"Cluster_{cid}" if cid < 35 else "Others"
                for mol_idx in indices:
                    dict_cluster[sim_sheet.sequences[mol_idx]] = cname
            df_s["Cluster_Sheet"] = df_s["Seq_Clean"].map(dict_cluster).fillna("Others/Singleton")
        except Exception as e:
            result["warnings"].append(f"Mapping clusters '{sheet}' partiel : {e}")
            df_s["Cluster_Sheet"] = "Others/Singleton"

        pred_cols = [c for c in df_s.columns if str(c).startswith("Peptide_") and not str(c).endswith("_raw")]
        if pred_cols:
            df_s["Activité_Dominante"] = (
                df_s[pred_cols].idxmax(axis=1).str.replace("Peptide_", "", regex=False)
            )
            df_s["Proba_Max"] = df_s[pred_cols].max(axis=1)
        else:
            df_s["Activité_Dominante"] = "Inconnu"
            df_s["Proba_Max"] = 0.0
            result["warnings"].append(f"Feuille '{sheet}' : aucune colonne 'Peptide_*' — treemap sans activité dominante.")

        try:
            df_tm = (
                df_s.groupby(["Cluster_Sheet", "Activité_Dominante"], as_index=False)
                .agg(Nbr_Peptides=("Seq_Clean", "count"), Proba_Moyenne=("Proba_Max", "mean"))
            )
            fig_s = px.treemap(
                df_tm, path=["Cluster_Sheet", "Activité_Dominante"],
                values="Nbr_Peptides", color="Proba_Moyenne",
                color_continuous_scale="RdBu",
                title=f"Treemap — {sheet} (cutoff={cutoff})",
                hover_data={"Nbr_Peptides": True, "Proba_Moyenne": ":.4f"},
            )
            tm_path = os.path.join(DIR_CLUSTER, f"Treemap_{sheet}-{graphs_id}_c{cutoff}.html")
            fig_s.write_html(tm_path)
            result["treemap_sheets_html"].append(tm_path)
        except Exception as e:
            result["warnings"].append(f"Treemap '{sheet}' non généré : {e}")

        master_frames.append(df_s)

    if not master_frames:
        result["error"] = "Aucune feuille clusterisable (< 2 séquences uniques partout)."
        return result

    master_df = pd.concat(master_frames, ignore_index=True)
    master_path = os.path.join(DIR_CLUSTER, f"Self_Ids_Sequence_PepFuNN_{graphs_id}.xlsx")
    try:
        master_df.to_excel(master_path, index=False)
        result["master_xlsx"] = master_path
    except Exception as e:
        result["warnings"].append(f"Master XLSX non sauvegardé : {e}")

    if not result["error"]:
        per_sheet = ", ".join(f"{s}={n}" for s, n in result["clusters_per_sheet"].items()) or "aucune"
        cluster_summary = (
            f"Clustering PepFuNN terminé — run {result['run_id']} (cutoff={cutoff}). "
            f"{result['n_sequences']} séquences, {result['n_clusters']} clusters total. "
            f"Par feuille : {per_sheet}."
            + (f" Master XLSX : {result['master_xlsx']}" if result.get("master_xlsx") else "")
        )
        send_ntfy_notification(
            message=cluster_summary,
            title="PROTEOGEN — clustering terminé",
            priority="high",
            tags="white_check_mark",
        )
    return result
# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
@_ntfy_on_critical_error("run pipeline")
def run_proteogen_pipeline(
    input_file: str,
    target_activities: list | None = None,
    run_smiles: bool = False,
    run_clustering: bool = False,
    clustering_cutoff: float = 0.7,
    run_signalp: list | None = None,
    run_xai: bool = False,
    run_esmfold: bool = False,
    esmfold_activities: list | None = None,
    esmfold_threshold: float = 0.95,
    esmfold_top_n: int | None = None,
) -> dict:
    """
    Exécute le pipeline complet PROTEOGEN.
    Args:
        input_file       : Chemin vers le fichier Excel d'entrée (.xlsx).
                           Doit contenir des feuilles nommées '*_Peptides' avec une colonne 'Peptide'.
        target_activities: Liste des activités biologiques à prédire. None = toutes.
                           Exemples : ["Anti_Bacterien", "Quorum_Sensing", "Neuropeptide"]
        run_smiles       : True pour générer les colonnes SMILES (RDKit). Peut être lent.
        run_clustering   : True pour lancer le clustering PepFuNN (UMAP + Treemap).
        run_signalp      : Liste d'activités pour l'extraction FASTA + SignalP 6.0. None ou [] = ignoré.
                           Exemples : ["Neuropeptide"] ou ["Neuropeptide", "Anti_Cancer"]
        run_xai          : True pour lancer l'analyse XAI (Alanine Scanning) sur le top 10 de chaque activité.
        esmfold_activities : Liste d'activités pour lesquelles lancer le module ESM-Fold. None ou [] = ignoré.
    Returns:
        dict avec les clés :
            output_excel (str)     : chemin du fichier Excel des résultats
            html_files  (list[str]): chemins des fichiers HTML Plotly générés
            log_file    (str)      : chemin du fichier de log
            summary     (str)      : résumé textuel de l'exécution
    """
    """ 
    Exécute un pipeline de prédiction axé sur les microbes
    Args:
        input_file       : Chemin vers le fichier Excel d'entrée (.xlsx).
                           Doit contenir des feuilles nommées '*_Peptides' avec une colonne 'Peptide'.
        target_activities: Liste des activités biologiques à prédire. None = Anti_bactérien, Quorum_Sensing, Anti_Fongique, Anti_Parasitique, Anti_Biofilm, Anti_AMRSA.
        run_smiles       : False
        run_clustering   : True pour lancer le clustering PepFuNN (UMAP + Treemap).
        run_signalp      : Liste d'activités pour l'extraction FASTA + SignalP 6.0. None ou [] = ignoré.
                           Exemples : ["Anti_Bactérien"] ou ["Anti_Bactérien", "Anti_Fongique"]
        run_xai          : True pour lancer l'analyse XAI (Alanine Scanning) sur le top 10 de chaque activité.
    Returns:
        dict avec les clés :
            output_excel (str)     : chemin du fichier Excel des résultats
            html_files  (list[str]): chemins des fichiers HTML Plotly générés
            log_file    (str)      : chemin du fichier de log
            summary     (str)      : résumé textuel de l'exécution
    
    """
    # Création des dossiers de sortie
    for d in [DIR_LOGS, DIR_GRAPHS, DIR_XLSX_Interest, DIR_FASTA_Interest, DIR_CLUSTER, DIR_SIGNALP, DIR_ESMFOLD]:
        os.makedirs(d, exist_ok=True)
    start_time = time.time()
    date_run   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fichier_log = os.path.join(DIR_LOGS, f"Prediction_1280D_Run_{date_run}.log")
    # Reconfiguration du logger racine pour cette exécution
    #for handler in logging.root.handlers[:]:
        #logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(fichier_log, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Script PROTEOGEN lancé")
    graphs_id   = os.path.splitext(os.path.basename(input_file))[0]
    sortie_file = f"PROTEOGEN_Results_{graphs_id}_{date_run}.xlsx"

    # Sélection des modèles
    if target_activities:
        model_a_tester = [m for m in ALL_MODELS if m["nom_colonne"] in target_activities]
        if not model_a_tester:
            logging.warning(f"Aucun modèle trouvé pour {target_activities}. Tous les modèles seront utilisés.")
            model_a_tester = ALL_MODELS
    else:
        model_a_tester = ALL_MODELS
    predict_column_global = [f"Peptide_{m['nom_colonne']}" for m in ALL_MODELS]
    _activities_listed = [m['nom_colonne'] for m in model_a_tester]
    logging.info(f"Activités sélectionnées : {_activities_listed}")
    send_ntfy_notification(
        message=f"Pipeline démarré sur {os.path.basename(input_file)}\nActivités : {', '.join(_activities_listed)}",
        title="PROTEOGEN — run démarré",
        priority="default",
        tags="rocket",
    )
    # ── Chargement ESM-2 ──────────────────────────────────────────────────────
    logging.info("Initialisation ESM-2 (esm2_t33_650M_UR50D)")
    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # ── Lecture du fichier Excel ──────────────────────────────────────────────
    logging.info(f"Lecture du fichier Excel : {input_file}")
    try:
        xls = pd.ExcelFile(input_file)
        peptide_sheet = [s for s in xls.sheet_names if "_Peptides" in str(s)]
        if not peptide_sheet:
            logging.warning("Aucune feuille '_Peptides'. Traitement de toutes les feuilles.")
            peptide_sheet = xls.sheet_names
    except FileNotFoundError:
        logging.error(f"Fichier introuvable : {input_file}", exc_info=True)
        raise
    All_sheet_results = []
    # ── Boucle principale par feuille ─────────────────────────────────────────
    with pd.ExcelWriter(sortie_file, engine="openpyxl") as writer:
        for sheet_name in peptide_sheet:
            logging.info(f"\n{'=' * 40}")
            logging.info(f"Traitement de la feuille : {sheet_name}")
            dataset = pd.read_excel(input_file, sheet_name=sheet_name, na_filter=True)
            dataset["Sheet"] = sheet_name
            sequence_list = dataset["Peptide"]
            # ── Calcul des embeddings ESM-2 ───────────────────────────────────
            logging.info("Calcul embeddings ESM-2 pour chaque peptide")
            embeddings_list = []
            valid_indices   = []
            for idx, seq in sequence_list.items():
                if _stop_event.is_set():
                    raise InterruptedError("Pipeline interrompu par l'utilisateur.")
                seq_sans_parentheses = re.sub(r"\(.*?\)", "", str(seq))
                clean_seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", seq_sans_parentheses.upper())
                if len(clean_seq) == 0:
                    continue
                one_seq_embeddings = esm_embeddings_token_level(model_esm, alphabet, [(clean_seq, clean_seq)]) #Prend compte du tenser L*50*1280D
                embeddings_list.append(one_seq_embeddings)
                valid_indices.append(idx)
            embeddings_memory  = np.concatenate(embeddings_list, axis=0) # Tenseur (N, MAX_LEN, 1280)
            # ── Boucle de prédiction Keras + Monte Carlo Dropout — NE PAS MODIFIER ──
            for config in model_a_tester:
                if _stop_event.is_set():
                    raise InterruptedError("Pipeline interrompu par l'utilisateur.")
                nom = config["nom_colonne"]
                logging.info(f"Analyse en cours : {nom.upper()}")
                try:
                    X_test_scaled = embeddings_memory
                    logging.info("Chargement du modèle CNN Keras")
                    cnn_model = load_model(config["model"])
                    logging.info("-> Modèle Keras chargé dans la RAM")
                    logging.info(f"Monte Carlo Dropout : {mc_passes} passes")
                    mc_prediction = []
                    for _ in range(mc_passes):
                        if _stop_event.is_set():
                            raise InterruptedError("Pipeline interrompu par l'utilisateur.")
                        batch_prediction = []
                        for i in range(0, len(X_test_scaled), 256):
                            batch_data = X_test_scaled[i: i + 256]
                            pred = cnn_model(batch_data, training=True).numpy()
                            batch_prediction.append(pred)
                        full_pred = np.vstack(batch_prediction)
                        mc_prediction.append(full_pred)
                    mc_prediction = np.stack(mc_prediction)
                    predicted_probability = mc_prediction.mean(axis=0)
                    uncertainty_score = mc_prediction.std(axis=0)
                    predicted_class  = []
                    uncertainty_class = []
                    for i in range(predicted_probability.shape[0]):
                        valeurs_plates = np.ravel(predicted_probability[i])
                        proba_positif  = valeurs_plates[-1]
                        predicted_class.append(round(float(proba_positif), 5))
                        uncert_plates  = np.ravel(uncertainty_score[i])
                        uncert_positif = uncert_plates[-1]
                        uncertainty_class.append(round(float(uncert_positif), 5))
                    # ── Platt Scaling ────────────────────
                    pa = config.get("platt_a")
                    pb = config.get("platt_b")
                    if pa is not None and pb is not None:
                        raw_probas = np.array(predicted_class)
                        calibrated_probas = apply_platt(raw_probas, pa, pb)
                        logging.info("Platt scaling appliqué aux probabilités MC Dropout")
                        nom_col_cal = f"Peptide_{nom}"
                        dataset.loc[valid_indices, nom_col_cal] = calibrated_probas
                        predicted_class = [round(float(p), 5) for p in calibrated_probas]
                    nom_col_uncert  = f"Incertitude_{nom}"
                    dataset.loc[valid_indices, nom_col_uncert] = uncertainty_class
                    logging.info(f"Colonne '{nom_col_cal}' ajoutée")
                    logging.info("Prédictions terminées")
                    # ── Gestion mémoire — NE PAS MODIFIER ────────────────────
                    try:
                        K.clear_session()
                        del cnn_model
                        gc.collect()
                        logging.info(f"Nettoyage RAM — modèle {nom}")
                    except Exception as e:
                        logging.error(f"Erreur nettoyage Keras pour {nom}", exc_info=True)
                except Exception as e:
                    logging.error(f"Erreur prédiction Keras pour {nom}", exc_info=True)
            # ── Module ESM-Fold (optionnel) ─────────────────────────────────────────────
            if run_esmfold:
                logging.info("=" * 40)
                logging.info("Lancement du module ESM-Fold - prédiction de structure 3D")
                send_ntfy_notification(
                    message=f"Module ESM-Fold démarré (feuille {sheet_name})",
                    title="PROTEOGEN — module démarré",
                    priority="default", tags="dna",
                )
                try:
                    esmfold_results = run_esmfold_module(
                        dataset=dataset,
                        esm2_model=model_esm,
                        sheet_name=str(sheet_name),
                        esmfold_activities=esmfold_activities,
                        activity_threshold=esmfold_threshold,
                        top_n = esmfold_top_n,
                    )
                    dataset = esmfold_results["dataset"]
                    model_esm = esmfold_results["esm_model"]
                    alphabet   = esmfold_results["alphabet"]
                    logging.info(
                        f"ESM-Fold terminé -"
                        f" {esmfold_results['n_predicted']} structures prédites, "
                        f" {esmfold_results['n_failed']} échecs"
                        f"PDB : {esmfold_results['pdb_dir']}"
                    )
                    send_ntfy_notification(
                        message=f"ESM-Fold terminé ({sheet_name}) : {esmfold_results['n_predicted']} OK / {esmfold_results['n_failed']} KO",
                        title="PROTEOGEN — module terminé",
                        priority="default", tags="white_check_mark",
                    )
                except Exception as e:
                    logging.error(f"Erreur lors du module ESM-Fold : {e}", exc_info=True)
                    send_ntfy_notification(
                        message=f"ESM-Fold ({sheet_name}) — crash {type(e).__name__} : {e}",
                        title="PROTEOGEN — ERREUR module",
                        priority="high", tags="warning",
                    )
            else:
                logging.info("Module ESM-Fold désactivé - aucune structure 3D prédite")
            # ── Paramètres physico-chimiques ──────────────────────────────────
            def run_physicochemical_analysis(df):
                mw, hydro, pi, charge, aliph, arom, h_don, xlogp = [], [], [], [], [], [], [], []
                vhse_list, zscale_list, cruciani_list, mswhim_list = [], [], [], []
                for seq in df["Peptide"]:
                    try:
                        clean = re.sub(r"\(.*?\)", "", str(seq))
                        clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "", clean.upper())
                        if not clean:
                            mw.append(np.nan); hydro.append(np.nan); pi.append(np.nan); charge.append(np.nan)
                            aliph.append(np.nan); arom.append(np.nan); h_don.append(np.nan); xlogp.append(np.nan)
                            vhse_list.append([np.nan] * 8); zscale_list.append([np.nan] * 5)
                            cruciani_list.append([np.nan] * 3); mswhim_list.append([np.nan] * 3)
                            continue
                        p = peptides.Peptide(clean)
                        mw.append(round(p.molecular_weight(), 5))
                        hydro.append(round(p.hydrophobicity(), 5))
                        pi.append(round(p.isoelectric_point(), 5))
                        charge.append(round(p.charge(pH=7), 5))
                        aliph.append(round(peptidy.descriptors.aliphatic_index(clean), 5))
                        arom.append(round(peptidy.descriptors.aromaticity(clean), 5))
                        h_don.append(round(peptidy.descriptors.n_h_donors(clean), 5))
                        xlogp.append(round(peptidy.descriptors.x_logp_energy(clean), 5))
                        vhse_list.append([round(v, 5) for v in p.vhse_scales()])
                        zscale_list.append([round(v, 5) for v in p.z_scales()])
                        cruciani_list.append([round(v, 5) for v in p.cruciani_properties()])
                        mswhim_list.append([round(v, 5) for v in p.ms_whim_scores()])
                    except Exception as e:
                        logging.warning(f"Erreur physico-chimique pour {seq} : {e}")
                        mw.append(np.nan); hydro.append(np.nan); pi.append(np.nan); charge.append(np.nan)
                        aliph.append(np.nan); arom.append(np.nan); h_don.append(np.nan); xlogp.append(np.nan)
                        vhse_list.append([np.nan] * 8); zscale_list.append([np.nan] * 5)
                        cruciani_list.append([np.nan] * 3); mswhim_list.append([np.nan] * 3)
                df["Molecular_Weight"] = mw
                df["Hydrophobicity"]   = hydro
                df["Isoelectric_Point"]= pi
                df["Charge_at_pH"]     = charge
                df["Aliphatic_Index"]  = aliph
                df["Aromatic_Index"]   = arom
                df["H_Donor"]          = h_don
                df["Sharing_Coefficient"] = xlogp
                df[["VHSE1","VHSE2","VHSE3","VHSE4","VHSE5","VHSE6","VHSE7","VHSE8"]]   = pd.DataFrame(vhse_list,    index=df.index)
                df[["Zscale_1","Zscale_2","Zscale_3","Zscale_4","Zscale_5"]]            = pd.DataFrame(zscale_list,  index=df.index)
                df[["Cruciani_1","Cruciani_2","Cruciani_3"]]                            = pd.DataFrame(cruciani_list,index=df.index)
                df[["MSWhim_1","MSWhim_2","MSWhim_3"]]                                  = pd.DataFrame(mswhim_list,  index=df.index)
                return df
            dataset = run_physicochemical_analysis(dataset)
            # ── Sauvegarde de la feuille ──────────────────────────────────────
            dataset.to_excel(writer, index=False, sheet_name=sheet_name)
            All_sheet_results.append(dataset)
            logging.info(f"Feuille '{sheet_name}' sauvegardée dans {sortie_file}")
            # ── Graphiques scatter ────────────────────────────────────────────
            try:
                path_pI = os.path.join(DIR_GRAPHS, f"Isoelectric_Point_vs_Molecular_Weight_{sheet_name}.png")
                dataset.plot.scatter(x="Molecular_Weight", y="Isoelectric_Point",
                                     title=f"pI vs MW — {sheet_name}")
                plt.savefig(path_pI, dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                logging.warning(f"Erreur graphique pI vs MW : {e}")

            try:
                path_hydro = os.path.join(DIR_GRAPHS, f"Hydrophobicity_vs_Molecular_Weight_{sheet_name}.png")
                dataset.plot.scatter(x="Molecular_Weight", y="Hydrophobicity",
                                     title=f"Hydrophobicité vs MW — {sheet_name}")
                plt.savefig(path_hydro, dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                logging.warning(f"Erreur graphique Hydro vs MW : {e}")
            # ── Radar VHSE ────────────────────────────────────────────────────
            try:
                vhse_cols = ["VHSE1","VHSE2","VHSE3","VHSE4","VHSE5","VHSE6","VHSE7","VHSE8"]
                if all(c in dataset.columns for c in vhse_cols):
                    vhse_data = dataset[vhse_cols].dropna()
                    if not vhse_data.empty:
                        mean_vhse  = vhse_data.mean().values
                        df_radar   = pd.DataFrame(dict(r=mean_vhse, theta=vhse_cols))
                        fig_radar  = px.line_polar(df_radar, r="r", theta="theta",
                                                   line_close=True, title=f"VHSE moyen — {sheet_name}")
                        fig_radar.update_traces(fill="toself")
                        radar_path = os.path.join(DIR_GRAPHS, f"Radar_VHSE_{sheet_name}.html")
                        fig_radar.write_html(radar_path)
            except Exception as e:
                logging.warning(f"Erreur Radar VHSE : {e}")
            # ── SignalP multi-activités (optionnel) ───────────────────────────
            signalp_activities = run_signalp if run_signalp else []
            if signalp_activities:
                logging.info(f"Extraction SignalP pour {len(signalp_activities)} activité(s) : {signalp_activities}")
                send_ntfy_notification(
                    message=f"Module SignalP démarré ({sheet_name}) : {', '.join(signalp_activities)}",
                    title="PROTEOGEN — module démarré",
                    priority="default", tags="mag",
                )
            else:
                logging.info("SignalP ignoré (run_signalp vide)")
            _signalp_done, _signalp_failed = 0, 0
            for signalp_act in signalp_activities:
                logging.info(f"-- SignalP : activité '{signalp_act}' / feuille '{sheet_name}'")
                df_sp           = dataset.copy()
                colonne_recherche = f"Peptide_{signalp_act}"
                threshold       = 0.95
                if colonne_recherche not in df_sp.columns:
                    logging.warning(f"Colonne '{colonne_recherche}' absente — activité ignorée")
                    continue
                peptides_interets = df_sp[df_sp[colonne_recherche] >= threshold].copy()
                if "Found By" in peptides_interets.columns:
                    peptides_interets = peptides_interets[peptides_interets["Found By"] == "DB Search"].copy()
                if "tag(>=0.0%)" not in peptides_interets.columns or "Accession" not in peptides_interets.columns:
                    logging.warning(f"Colonnes 'tag' ou 'Accession' absentes — SignalP ignoré pour '{signalp_act}'")
                    continue
                peptides_interets = peptides_interets[["tag(>=0.0%)", "Accession"]].copy()
                peptides_interets["Peptide_Sequence"] = (
                    peptides_interets["tag(>=0.0%)"].str.replace(r"[^A-Za-z]", "", regex=True)
                )
                nom_fichier_final  = f"Peptides_{signalp_act}_Accession_{sheet_name}.xlsx"
                path_xlsx_interets = os.path.join(DIR_XLSX_Interest, nom_fichier_final)
                peptides_interets.drop_duplicates(subset=["Accession"]).to_excel(path_xlsx_interets, index=False)
                logging.info(f"{len(peptides_interets)} peptides d'intérêt sauvegardés : {nom_fichier_final}")
                liste_acc_FASTA   = peptides_interets["Accession"].str.split(":").str[0].astype(str).tolist()
                proteome_entree   = "augustus.fasta"
                proteome_sortie   = f"Proteome_{signalp_act}_{sheet_name}.fasta"
                path_FASTA_output = os.path.join(DIR_FASTA_Interest, proteome_sortie)
                try:
                    seqs_interet = [r for r in SeqIO.parse(proteome_entree, "fasta")
                                    if r.id.split(":")[0] in liste_acc_FASTA]
                    SeqIO.write(seqs_interet, path_FASTA_output, "fasta")
                    logging.info(f"{len(seqs_interet)} séquences FASTA → {proteome_sortie}")
                    result_folder_signalP = os.path.join(DIR_SIGNALP, f"Pred_SignalP_{signalp_act}_{sheet_name}")
                    commande_signalp = [
                        "signalp6", "--fastafile", path_FASTA_output,
                        "--organism", "eukarya", "--output_dir", result_folder_signalP,
                        "--format", "none", "--mode", "slow-sequential",
                        "--torch_num_threads", "40", "--write_procs", "16",
                    ]
                    subprocess.run(commande_signalp, capture_output=True, text=True, check=True)
                    logging.info(f"SignalP terminé → {result_folder_signalP}")
                    _signalp_done += 1
                except FileNotFoundError:
                    logging.error(f"Protéome introuvable : {proteome_entree}")
                    _signalp_failed += 1
                except subprocess.CalledProcessError as e:
                    logging.error(f"Erreur SignalP pour '{signalp_act}' : {e.stderr}")
                    _signalp_failed += 1
            if signalp_activities:
                send_ntfy_notification(
                    message=f"SignalP terminé ({sheet_name}) : {_signalp_done} OK / {_signalp_failed} KO sur {len(signalp_activities)}",
                    title="PROTEOGEN — module terminé",
                    priority="default", tags="white_check_mark",
                )
            # ── Génération SMILES ─────────────────────────────────────────────
            if run_smiles:
                logging.info("Génération des structures SMILES avec gestion des PTMs via RDKit Reactions")
            else:
                logging.info("Génération SMILES ignorée (run_smiles=False)")
            from rdkit.Chem import AllChem #type: ignore
            def generate_smile_column(df):
                #Définir les PTM
                rxn_amidation = AllChem.ReactionFromSmarts('[CX3:1](=[O:2])[OH1]>>[CX3:1](=[O:2])[NH2]')
                rxn_pyroglutamate = AllChem.ReactionFromSmarts('[NH2:1]-[CH1:2](-[C:3]=[O:4])-[CH2:5]-[CH2:6]-[C:7](=[O:8])[NH2:9]>>[O:4]=[C:3]-[CH1:2]1-[NH1:1]-[C:7](=[O:8])-[CH2:6]-[CH2:5]1')
                rxn_deamidation = AllChem.ReactionFromSmarts('[CX4:1]-[CX3:2](=[O:3])[NH2]>>[CX4:1]-[CX3:2](=[O:3])[OH]')
                # Tag = PTM
                ptm_reactions = {
                    'Amidation': rxn_amidation,
                    'Pyroglutamate': rxn_pyroglutamate,
                    'Deamidation': rxn_deamidation
                }
                datasetS = generate_smile_column(dataset)
                def get_smile_with_ptm(seq):
                    seq_str = str(seq).upper()
                    #Récuperer la séquence 
                    clean_seq = re.sub(r'\(.*?\)', '', seq_str)
                    clean_seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', clean_seq)
                    if not clean_seq:
                        return None 
                    try:
                        mol = Chem.MolFromSequence(clean_seq)
                        if mol is None:
                            return None
                        #Analyser les tags de PTM dans la séquence et appliquer les réactions correspondantes
                        if '(-0.98)' in seq_str:
                            modified_mols = ptm_reactions['Amidation'].RunReactants((mol,))
                            if modified_mols:
                                mol = modified_mols[0][0]
                                Chem.SanitizeMol(mol)
                        if '(-17.02)' in seq_str:
                            modified_mols = ptm_reactions['Pyroglutamate'].RunReactants((mol,))
                            if modified_mols:
                                mol = modified_mols[0][0]
                                Chem.SanitizeMol(mol)
                        if '(+0.98)' in seq_str:
                            modified_mols = ptm_reactions['Deamidation'].RunReactants((mol,))
                            if modified_mols:
                                mol = modified_mols[0][0]
                                Chem.SanitizeMol(mol)
                        return Chem.MolToSmiles(mol)
                    except Exception as e:
                        logging.debug(f"SMILES failure for '{seq_str}' : {e}")
                        return None
                df['SMILES'] = df['Peptide'].apply(get_smile_with_ptm)
                return df
    # ── Matériel et méthodes sheet ─────────────────────────────────────────────────────
    try:
        logging.info("Génération de la feuille Matériel & Méthodes")
        method_rows = []
        # Section 1 — Embeddings
        method_rows.append({"Section": "Embeddings", "Paramètre": "Modèle", "Valeur": "ESM-2 (esm2_t33_650M_UR50D)", "Détail": "Facebook AI Research, 33 couches, 650M paramètres"})
        method_rows.append({"Section": "Embeddings", "Paramètre": "Dimension", "Valeur": "1280D token-level", "Détail": f"MAX_LEN = {MAX_LEN} résidus, padding zéro si plus court"})
        method_rows.append({"Section": "Embeddings", "Paramètre": "Couche extraction", "Valeur": "repr_layers=[33]", "Détail": "Dernière couche du transformer"})
        # Section 2 — Prédiction
        method_rows.append({"Section": "Prédiction", "Paramètre": "Architecture", "Valeur": "CNN 1D (Keras)", "Détail": "Token-level, classification binaire par activité"})
        method_rows.append({"Section": "Prédiction", "Paramètre": "Incertitude", "Valeur": f"MC Dropout ({mc_passes} passes)", "Détail": "Monte Carlo Dropout — moyenne = probabilité, écart-type = incertitude"})
        method_rows.append({"Section": "Prédiction", "Paramètre": "Batch size inférence", "Valeur": "256", "Détail": "Traitement par lots de 256 séquences"})
        # Section 3 — Calibration (par modèle)
        for config in model_a_tester:
            nom = config["nom_colonne"]
            pa = config.get("platt_a")
            pb = config.get("platt_b")
            brier = config.get("brier")
            ece = config.get("ece")
            n_val = config.get("n_val")
            if pa is not None and pb is not None:
                method_rows.append({
                    "Section": "Calibration",
                    "Paramètre": f"Platt Scaling — {nom}",
                    "Valeur": f"a={pa:.4f}, b={pb:.4f}",
                    "Détail": f"sigmoid(a·logit(p)+b) | Brier={brier:.4f}, ECE={ece:.4f}, n_val={n_val}"
                })
            else:
                method_rows.append({
                    "Section": "Calibration",
                    "Paramètre": f"Platt Scaling — {nom}",
                    "Valeur": "Non calibré",
                    "Détail": "Probabilités MC Dropout brutes"
                })
        # Section 4 — Physico-chimie
        method_rows.append({"Section": "Physico-chimie", "Paramètre": "Librairie", "Valeur": "peptides + peptidy", "Détail": "MW, hydrophobicité, pI, charge pH7, indice aliphatique, aromaticité, H-donors, XLogP"})
        method_rows.append({"Section": "Physico-chimie", "Paramètre": "Descripteurs", "Valeur": "VHSE(8), Z-scales(5), Cruciani(3), MS-WHIM(3)", "Détail": "19 descripteurs physico-chimiques par séquence"})
        # Section 5 — Modules optionnels
        if run_clustering:
            method_rows.append({"Section": "Clustering", "Paramètre": "Méthode", "Valeur": "PepFuNN + UMAP (par feuille)", "Détail": f"Cutoff similarité = {clustering_cutoff}, clustering indépendant par feuille, UMAP + Treemap par feuille"})
        if run_smiles:
            method_rows.append({"Section": "SMILES", "Paramètre": "Génération", "Valeur": "RDKit AllChem", "Détail": "Gestion PTMs (phosphorylation, acétylation, déamidation)"})
        if run_xai:
            method_rows.append({"Section": "XAI", "Paramètre": "Méthode", "Valeur": "Perturbation par résidu", "Détail": "Impact de chaque acide aminé sur la prédiction (top 10 peptides)"})
        if run_esmfold:
            method_rows.append({"Section": "ESMFold", "Paramètre": "Structure 3D", "Valeur": f"Seuil={esmfold_threshold}, top_n={esmfold_top_n}", "Détail": f"Activités : {esmfold_activities}"})
        if run_signalp:
            method_rows.append({"Section": "SignalP", "Paramètre": "Peptides signal", "Valeur": "SignalP 6.0 (eukarya, slow-sequential)", "Détail": f"Activités filtrées : {run_signalp}"})
        # Section 6 — Métadonnées run
        method_rows.append({"Section": "Métadonnées", "Paramètre": "Date exécution", "Valeur": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Détail": ""})
        method_rows.append({"Section": "Métadonnées", "Paramètre": "Fichier entrée", "Valeur": os.path.basename(input_file), "Détail": f"Feuilles : {', '.join(str(s) for s in peptide_sheet)}"})
        method_rows.append({"Section": "Métadonnées", "Paramètre": "Modèles exécutés", "Valeur": str(len(model_a_tester)), "Détail": ", ".join(c["nom_colonne"] for c in model_a_tester)})
        method_rows.append({"Section": "Métadonnées", "Paramètre": "Fichier sortie", "Valeur": os.path.basename(sortie_file), "Détail": ""})
        df_methods = pd.DataFrame(method_rows)
        df_methods.to_excel(writer, index=False, sheet_name="Matériel_et_Méthodes")
        logging.info("Feuille 'Matériel_et_Méthodes' ajoutée au XLSX")
    except Exception as e:
        logging.error(f"Erreur génération sheet Matériel & Méthodes : {e}", exc_info=True)
    # ── Assemblage global ─────────────────────────────────────────────────────
    try:
        df_global = pd.concat(All_sheet_results, ignore_index=True)
        logging.info(f"Total peptides assemblés : {len(df_global)}")
    except Exception as e:
        logging.error(f"Erreur assemblage feuilles : {e}", exc_info=True)
        df_global = pd.DataFrame()
    if not df_global.empty:
        # Nettoyage des séquences
        try:
            df_global["Seq_Clean"] = (
                df_global["Peptide"].astype(str)
                .apply(lambda s: re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "",
                                        re.sub(r"\(.*?\)", "", s).upper()))
            )
            df_global = df_global[df_global["Seq_Clean"].str.len() > 0]
            df_global = df_global.drop_duplicates(subset=["Seq_Clean"]).reset_index(drop=True)
            logging.info(f"Séquences uniques après nettoyage PTM : {len(df_global)}")
        except Exception as e:
            logging.error(f"Erreur nettoyage : {e}", exc_info=True)
        # ── XAI global (optionnel) ───────────────────────────────────────────
        if run_xai:
            logging.info("\n" + "=" * 40 + "\nLANCEMENT XAI GLOBAL\n" + "=" * 40)
            send_ntfy_notification(
                message=f"Module XAI global démarré ({len(model_a_tester)} activité(s))",
                title="PROTEOGEN — module démarré",
                priority="default", tags="bulb",
            )
        else:
            logging.info("XAI ignoré (run_xai=False)")
        _xai_done, _xai_failed = 0, 0
        for config in model_a_tester if run_xai else []:
            nom            = config["nom_colonne"]
            nom_col_tableau = f"Peptide_{nom}"
            if nom_col_tableau not in df_global.columns:
                continue
            logging.info(f"XAI top 10 pour : {nom}")
            top_10_df = df_global.sort_values(by=nom_col_tableau, ascending=False).head(10)
            try:
                cnn_model = load_model(config["model"])
                global_aa_impacts = {aa: [] for aa in "ACDEFGHIKLMNPQRSTVWY"}
                for _, row in top_10_df.iterrows():
                    seq_to_test = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X",
                                         re.sub(r"\(.*?\)", "", str(row["Peptide"])).upper())
                    if not seq_to_test:
                        continue
                    impacts, baseline = generate_XAI_model(
                        seq_to_test, model_esm, alphabet, cnn_model, nom
                    )
                    for aa, imp in zip(seq_to_test, impacts):
                        if aa in global_aa_impacts:
                            global_aa_impacts[aa].append(imp)
                plot_global_XAI(global_aa_impacts, nom)
                # ── Gestion mémoire — NE PAS MODIFIER ────────────────────────
                K.clear_session()
                del cnn_model
                gc.collect()
                _xai_done += 1
            except Exception as e:
                logging.error(f"Erreur XAI pour {nom} : {e}", exc_info=True)
                _xai_failed += 1
        if run_xai:
            send_ntfy_notification(
                message=f"XAI global terminé : {_xai_done} OK / {_xai_failed} KO sur {len(model_a_tester)}",
                title="PROTEOGEN — module terminé",
                priority="default", tags="white_check_mark",
            )
        # ── Clustering PepFuNN PAR FEUILLE (optionnel) ────────────────────────
        if run_clustering:
            logging.info("\n" + "=" * 60 + f"\nCLUSTERING PAR FEUILLE (cutoff={clustering_cutoff})\n" + "=" * 60)
            send_ntfy_notification(
                message=f"Module Clustering PepFuNN démarré (cutoff={clustering_cutoff})",
                title="PROTEOGEN — module démarré",
                priority="default", tags="link",
            )
            if not (0.0 < clustering_cutoff < 1.0):
                logging.error(f"clustering_cutoff invalide : {clustering_cutoff}. Attendu ]0, 1[. Clustering sauté.")
            elif "Sheet" not in df_global.columns:
                logging.warning("Colonne 'Sheet' absente de df_global — clustering par feuille impossible.")
            else:
                cluster_frames = []
                for sheet in df_global["Sheet"].dropna().unique():
                    logging.info(f"--- Clustering feuille : {sheet} ---")
                    try:
                        df_s = df_global[df_global["Sheet"] == sheet].copy()
                        df_s = df_s[df_s["Seq_Clean"].astype(str).str.len() > 0]
                        df_s = df_s.drop_duplicates(subset=["Seq_Clean"]).reset_index(drop=True)
                        if len(df_s) < 2:
                            logging.warning(f"Feuille '{sheet}' : < 2 séquences uniques, clustering sauté.")
                            continue
                        if "Accession" in df_s.columns:
                            liste_acc = df_s["Accession"].astype(str).tolist()
                        else:
                            liste_acc = [f"{sheet}_seq_{i}" for i in range(len(df_s))]
                        liste_seq = df_s["Seq_Clean"].tolist()
                        sim_sheet = Clustering_PepFuNN_1280D_UMAP.simClustering(
                            ids=liste_acc, sequences=liste_seq
                        )
                        sim_sheet.run_clustering(cutoff=clustering_cutoff)
                        logging.info(f"Feuille '{sheet}' : {len(sim_sheet.clusters)} clusters")
                        try:
                            output_umap = os.path.join(DIR_CLUSTER, f"PepFuNN_UMAP-{sheet}-{graphs_id}.html")
                            sim_sheet.plot_sim_space_plotly(add_cluster_color=True, out_name=output_umap)
                            logging.info(f"UMAP '{sheet}' : {output_umap}")
                        except Exception as e:
                            logging.error(f"Erreur UMAP '{sheet}' : {e}", exc_info=True)
                        try:
                            dict_cluster = {}
                            for cluster_id, cluster_indices in enumerate(sim_sheet.clusters):
                                cluster_name = f"Cluster_{cluster_id}" if cluster_id < 35 else "Others"
                                for mol_idx in cluster_indices:
                                    dict_cluster[sim_sheet.sequences[mol_idx]] = cluster_name
                            df_s["Cluster_Sheet"] = (
                                df_s["Seq_Clean"].map(dict_cluster).fillna("Others/Singleton")
                            )
                        except Exception as e:
                            logging.error(f"Erreur mapping clusters '{sheet}' : {e}", exc_info=True)
                            df_s["Cluster_Sheet"] = "Others/Singleton"
                        try:
                            cols_dispo = [c for c in predict_column_global if c in df_s.columns]
                            if cols_dispo:
                                df_s["Activité_Dominante"] = (
                                    df_s[cols_dispo].idxmax(axis=1)
                                    .str.replace("Peptide_", "", regex=False)
                                )
                                df_s["Proba_Max"] = df_s[cols_dispo].max(axis=1)
                            else:
                                df_s["Activité_Dominante"] = "Inconnu"
                                df_s["Proba_Max"] = 0.0
                            df_tm_s = (
                                df_s
                                .groupby(["Cluster_Sheet", "Activité_Dominante"], as_index=False)
                                .agg(Nbr_Peptides=("Seq_Clean", "count"), Proba_Moyenne=("Proba_Max", "mean"))
                            )
                            fig_s = px.treemap(
                                df_tm_s,
                                path=["Cluster_Sheet", "Activité_Dominante"],
                                values="Nbr_Peptides", color="Proba_Moyenne",
                                color_continuous_scale="RdBu",
                                title=f"Treemap — {sheet} (cutoff={clustering_cutoff})",
                                hover_data={"Nbr_Peptides": True, "Proba_Moyenne": ":.4f"},
                            )
                            output_tm_s = os.path.join(DIR_CLUSTER, f"Treemap_{sheet}-{graphs_id}.html")
                            fig_s.write_html(output_tm_s)
                            logging.info(f"Treemap '{sheet}' : {output_tm_s}")
                        except Exception as e:
                            logging.error(f"Erreur Treemap '{sheet}' : {e}", exc_info=True)
                        cluster_frames.append(df_s)
                    except Exception as e:
                        logging.error(f"Erreur clustering feuille '{sheet}' : {e}", exc_info=True)
                if cluster_frames:
                    try:
                        master_df = pd.concat(cluster_frames, ignore_index=True)
                        master_path = os.path.join(DIR_CLUSTER, f"Self_Ids_Sequence_PepFuNN_{graphs_id}.xlsx")
                        master_df.to_excel(master_path, index=False)
                        logging.info(f"Master XLSX clustering : {master_path}")
                    except Exception as e:
                        logging.error(f"Erreur sauvegarde master XLSX : {e}", exc_info=True)
                send_ntfy_notification(
                    message=f"Clustering PepFuNN terminé : {len(cluster_frames)} feuille(s) traitée(s)",
                    title="PROTEOGEN — module terminé",
                    priority="default", tags="white_check_mark",
                )
    # ── Collecte des fichiers HTML générés ────────────────────────────────────
    html_files_generated = []
    for search_dir in [DIR_GRAPHS, DIR_CLUSTER, DIR_DASHBOARD]:
        if os.path.isdir(search_dir):
            for fname in sorted(os.listdir(search_dir)):
                if not fname.endswith(".html"):
                    continue
                fpath = os.path.join(search_dir, fname)
                # Run actuel uniquement : on ne garde que les HTML (ré)écrits depuis
                # le début du run. Évite de ramasser les graphiques des runs
                # précédents encore présents dans DIR_GRAPHS / DIR_CLUSTER / DIR_DASHBOARD.
                if os.path.getmtime(fpath) >= start_time:
                    html_files_generated.append(fpath)
    # ── Dashboard client HTML interactif ──────────────────────────────────────
    dashboard_html = None
    try:
        if not df_global.empty:
            dashboard_html = _generate_client_dashboard(
                df_global=df_global,
                model_a_tester=model_a_tester,
                output_dir=DIR_DASHBOARD,
                graphs_id=graphs_id,
                date_run=date_run,
            )
            # NB : pas d'ajout à html_files_generated — le dashboard (souvent
            # >200 Mo) est exposé séparément via la clé 'dashboard_html' et ne doit
            # PAS être ré-embarqué dans les onglets (limite websocket Streamlit).
            logging.info(f"Dashboard client généré : {dashboard_html}")
        else:
            logging.warning("Dashboard client ignoré : df_global vide")
    except Exception as e:
        logging.error(f"Erreur génération dashboard client : {e}", exc_info=True)
    # ── Temps d'exécution ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    h, r    = divmod(elapsed, 3600)
    m, s    = divmod(r, 60)
    duration_str = f"{int(h)}h {int(m)}m {s:.2f}s"
    logging.info(f"Temps d'exécution total : {duration_str}")
    summary = (
        f"Pipeline PROTEOGEN terminé en {duration_str}. "
        f"{len(model_a_tester)} modèle(s) exécuté(s) sur {len(peptide_sheet)} feuille(s). "
        f"{len(html_files_generated)} graphique(s) HTML généré(s). "
        f"Résultats Excel : {sortie_file}"
        + (f" · Dashboard : {dashboard_html}" if dashboard_html else "")
    )
    send_ntfy_notification(
        message=summary,
        title="PROTEOGEN — run terminé",
        priority="high",
        tags="white_check_mark",
    )
    return {
        "output_excel":   sortie_file,
        "html_files":     html_files_generated,
        "dashboard_html": dashboard_html,
        "log_file":       fichier_log,
        "summary":        summary,
    }
