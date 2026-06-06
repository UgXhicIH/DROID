# Copyright (c) 2026 Morgan Letoux. All rights reserved.
# This file is part of PROTEOGEN/DROID.
# Unauthorized use, reproduction or distribution is strictly prohibited.
# See LICENSE for details.
"""
FineTune_ESM2_LoRA_for_CNN.py  —  Template générique (adapter TASK_NAME / EXCEL_FILE)

Pipeline en 3 phases pour petits datasets :
  Phase 1 – Fine-tuning ESM2-650M par DoRA (PyTorch + PEFT, use_dora=True)
             DoRA = Weight-Decomposed Low-Rank Adaptation (Liu et al., ICML 2024)
             Décomposition W = m · (V / ||V||c) : magnitude scalaire + direction LoRA-adaptée
             Convergence supérieure à LoRA à rang faible — adapté au contexte
             peptidomique (datasets petits, séquences courtes MAX_LEN=50)
             K-fold CV, EarlyStopping sur val_AUC, rang r=4
             Extraction Out-Of-Fold (OOF) : zéro fuite de données
  Phase 2 – Embeddings token-level OOF → fichiers .npy
             Meilleur adapter DoRA utilisé pour le test set externe
  Phase 3 – CNN multi-échelle sur embeddings OOF (Keras/TF)
             K-fold CV, EarlyStopping, seuil Youden par fold
             Évaluation finale sur test set externe et indépendant

Dépendances supplémentaires :
  pip install transformers "peft>=0.9.0"
"""

import os
import gc
import math
import logging
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import AutoModel, AutoTokenizer #type: ignore
from peft import LoraConfig, get_peft_model, PeftModel #type: ignore

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve, matthews_corrcoef
)

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

os.makedirs("Logs", exist_ok=True)
os.makedirs("ModelCNN_PROTEOGEN_V2", exist_ok=True)
os.makedirs("DoRA_Adapters", exist_ok=True)

# =============================================================================
# CONFIGURATION — modifier uniquement cette section
# =============================================================================
TASK_NAME    = "Opioid"         
EXCEL_FILE   = "Train_Opioid.xlsx"    

MAX_LEN      = 50     
K_FOLDS      = 10     
TEST_SIZE    = 0.20  
RANDOM_STATE = 123

LORA_R       = 4                              # rang de la décomposition LoRA sous-jacente à DoRA
LORA_ALPHA   = 8                              # scaling factor (alpha/r = 2)
LORA_DROPOUT = 0.10
LORA_TARGETS = ["query", "key", "value"]      # modules attention ESM2
USE_DORA     = True                           # active DoRA : magnitude × LoRA(direction)

# Fine-tuning DoRA (PyTorch)
LR_LORA      = 5e-5
BATCH_LORA   = 16                             # réduire si OOM — DoRA ~10-15% plus lourd que LoRA
EPOCHS_LORA  = 60
PATIENCE_LORA = 10

LR_CNN       = 3e-4
BATCH_CNN    = 64
EPOCHS_CNN   = 200
PATIENCE_CNN_ES = 15
PATIENCE_CNN_LR = 8

ESM2_HF     = "facebook/esm2_t33_650M_UR50D"
NPY_TRAIN   = f"embeddings_oof_{TASK_NAME}_train.npy"
NPY_TEST    = f"embeddings_oof_{TASK_NAME}_test.npy"
DORA_DIR    = f"DoRA_Adapters/{TASK_NAME}_best_fold"
MODEL_PATH  = f"ModelCNN_PROTEOGEN_V2/{TASK_NAME}_DoRA_CNN_model.keras"
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"Logs/FineTune_DoRA_{TASK_NAME}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device : {device}")


# =============================================================================
# Dataset PyTorch
# =============================================================================
class PeptideDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.enc = tokenizer(
            list(sequences),
            max_length=MAX_LEN + 2,  # +2 pour CLS / EOS
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# =============================================================================
# Modèle ESM2 + DoRA + tête de classification
# =============================================================================
class ESM2DoRAClassifier(nn.Module):
    def __init__(self, dora_encoder, num_classes=2, dropout=0.30):
        super().__init__()
        self.encoder = dora_encoder
        hidden = dora_encoder.config.hidden_size  # 1280 pour t33-650M
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(out.last_hidden_state, attention_mask)
        return self.head(pooled)

    @torch.no_grad()
    def extract_token_embeddings(self, input_ids, attention_mask):
        """Retourne (B, MAX_LEN, 1280) — strip CLS/EOS, pad/truncate."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state.cpu().numpy()  # (B, L+2, 1280)
        attn_np   = attention_mask.cpu().numpy()
        result = []
        for i in range(token_emb.shape[0]):
            seq_len = max(int(attn_np[i].sum()) - 2, 0)
            rep = token_emb[i, 1: seq_len + 1]           # (L, 1280)
            L = rep.shape[0]
            if L >= MAX_LEN:
                padded = rep[:MAX_LEN]
            else:
                pad = np.zeros((MAX_LEN - L, rep.shape[1]), dtype=np.float32)
                padded = np.concatenate([rep, pad], axis=0)
            result.append(padded.astype(np.float32))
        return np.array(result, dtype=np.float32)


# =============================================================================
# EarlyStopping PyTorch (sauvegarde en mémoire, pas sur disque)
# =============================================================================
class TorchEarlyStopping:
    def __init__(self, patience=10, mode="max"):
        self.patience  = patience
        self.mode      = mode
        self.counter   = 0
        self.best      = None
        self.triggered = False
        self._state    = None

    def step(self, score, model):
        better = (self.best is None) or \
                 (self.mode == "max" and score > self.best) or \
                 (self.mode == "min" and score < self.best)
        if better:
            self.best   = score
            self._state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True

    def restore(self, model):
        if self._state is not None:
            model.load_state_dict(self._state)


# =============================================================================
# Construction du modèle DoRA
# =============================================================================
def build_dora_model():
    base = AutoModel.from_pretrained(ESM2_HF)
    cfg  = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGETS,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        use_dora       = USE_DORA,        # DoRA : décomposition magnitude/direction
    )
    return ESM2DoRAClassifier(get_peft_model(base, cfg))


# =============================================================================
# Boucle d'entraînement DoRA (un epoch)
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(ids, mask), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    proba_all, label_all = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        prob = torch.softmax(model(ids, mask), dim=1)[:, 1].cpu().numpy()
        proba_all.extend(prob)
        label_all.extend(batch["labels"].numpy())
    proba_all = np.array(proba_all)
    label_all = np.array(label_all)
    auc = roc_auc_score(label_all, proba_all) if len(np.unique(label_all)) > 1 else 0.5
    return auc, proba_all, label_all


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    parts = []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        parts.append(model.extract_token_embeddings(ids, mask))
    return np.concatenate(parts, axis=0)


# =============================================================================
# Fine-tuning DoRA pour un fold
# =============================================================================
def run_dora_fold(seq_tr, y_tr, seq_val, y_val, tokenizer, fold_id):
    tr_ds  = PeptideDataset(seq_tr,  y_tr,  tokenizer)
    val_ds = PeptideDataset(seq_val, y_val, tokenizer)
    tr_ld  = DataLoader(tr_ds,  batch_size=BATCH_LORA, shuffle=True,  num_workers=0, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_LORA, shuffle=False, num_workers=0, pin_memory=True)

    model     = build_dora_model().to(device)
    optimizer = AdamW(model.parameters(), lr=LR_LORA, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7)
    criterion = nn.CrossEntropyLoss()
    es        = TorchEarlyStopping(patience=PATIENCE_LORA, mode="max")

    for epoch in range(1, EPOCHS_LORA + 1):
        loss    = train_one_epoch(model, tr_ld, optimizer, criterion)
        val_auc, _, _ = evaluate_loader(model, val_ld)
        scheduler.step(val_auc)
        es.step(val_auc, model)

        if epoch % 5 == 0 or es.triggered:
            logging.info(
                f"  [DoRA fold {fold_id}] ep {epoch:3d} | "
                f"loss={loss:.4f} | val_AUC={val_auc:.4f} | "
                f"best={es.best:.4f} | patience={es.counter}/{PATIENCE_LORA}"
            )
        if es.triggered:
            logging.info(f"  [DoRA fold {fold_id}] EarlyStopping — époque {epoch}")
            break

    es.restore(model)
    final_auc, _, _ = evaluate_loader(model, val_ld)
    logging.info(f"  [DoRA fold {fold_id}] AUC (best weights) = {final_auc:.4f}")
    return model, final_auc, val_ld


# =============================================================================
# Chargement données + split hold-out test
# =============================================================================
logging.info("=== Chargement dataset ===")
dataset   = pd.read_excel(EXCEL_FILE, na_filter=False)
sequences = list(dataset["sequence"])
y         = np.array(dataset["label"])
logging.info(f"Dataset : {len(y)} séquences | classes {dict(zip(*np.unique(y, return_counts=True)))}")

seq_train, seq_test, y_train, y_test = train_test_split(
    sequences, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
seq_train = np.array(seq_train)
seq_test  = np.array(seq_test)
logging.info(f"Train : {len(y_train)} | Test externe indépendant : {len(y_test)}")

# =============================================================================
# Phase 1 : K-fold DoRA fine-tuning + embeddings OOF
# =============================================================================
if os.path.exists(NPY_TRAIN) and os.path.exists(NPY_TEST):
    logging.info(f"\nEmbeddings OOF trouvés — Phase 1 ignorée")
    logging.info(f"  Train : {NPY_TRAIN}")
    logging.info(f"  Test  : {NPY_TEST}")
else:
    logging.info(f"\n=== PHASE 1 : K-fold DoRA Fine-tuning (r={LORA_R}, K={K_FOLDS}) ===")
    logging.info(f"Chargement tokenizer {ESM2_HF}")
    tokenizer = AutoTokenizer.from_pretrained(ESM2_HF)

    kf_dora    = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_emb    = np.zeros((len(y_train), MAX_LEN, 1280), dtype=np.float32)
    dora_aucs  = []
    best_auc   = -1.0

    for fold, (tr_idx, val_idx) in enumerate(kf_dora.split(seq_train, y_train)):
        logging.info(f"\n--- DoRA Fold {fold + 1}/{K_FOLDS} ---")
        model, fold_auc, val_ld = run_dora_fold(
            seq_train[tr_idx], y_train[tr_idx],
            seq_train[val_idx], y_train[val_idx],
            tokenizer, fold + 1,
        )
        dora_aucs.append(fold_auc)

        # Embeddings OOF sans fuite : modèle entraîné SANS ce fold
        logging.info(f"  Extraction embeddings OOF fold {fold + 1}")
        oof_emb[val_idx] = extract_embeddings(model, val_ld)

        # Sauvegarde du meilleur adapter DoRA (utilisé pour les embeddings test)
        if fold_auc > best_auc:
            best_auc = fold_auc
            model.encoder.save_pretrained(DORA_DIR)
            logging.info(f"  Meilleur fold ({fold_auc:.4f}) — adapter DoRA sauvegardé")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info(
        f"\nDoRA CV — AUC : {np.mean(dora_aucs):.4f} ± {np.std(dora_aucs):.4f}"
    )
    np.save(NPY_TRAIN, oof_emb)
    logging.info(f"OOF embeddings (train) : {NPY_TRAIN} — shape {oof_emb.shape}")

    # Embeddings test avec le meilleur adapter DoRA
    logging.info("\n=== Extraction embeddings test set (meilleur adapter DoRA) ===")
    base_esm  = AutoModel.from_pretrained(ESM2_HF)
    dora_best = PeftModel.from_pretrained(base_esm, DORA_DIR)
    clf_test  = ESM2DoRAClassifier(dora_best).to(device)
    clf_test.eval()

    test_ds = PeptideDataset(seq_test, y_test, tokenizer)
    test_ld = DataLoader(test_ds, batch_size=BATCH_LORA, shuffle=False, num_workers=0)
    X_test_emb = extract_embeddings(clf_test, test_ld)
    np.save(NPY_TEST, X_test_emb)
    logging.info(f"Embeddings test : {NPY_TEST} — shape {X_test_emb.shape}")

    del clf_test, dora_best, base_esm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =============================================================================
# Phase 2 : CNN multi-échelle sur embeddings OOF (Keras / TF)
# =============================================================================
logging.info("\n=== PHASE 2 : CNN multi-échelle sur embeddings OOF ===")

import keras                               # type: ignore
import tensorflow as tf                    # type: ignore
from keras.layers import (                 # type: ignore
    Input, Dense, Activation, BatchNormalization,
    Conv1D, Dropout, GlobalAveragePooling1D, Concatenate, Add,
)
from keras.models import Model, load_model  # type: ignore
from keras.optimizers import Adam           # type: ignore
from keras.callbacks import (               # type: ignore
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
)
from keras.regularizers import l2           # type: ignore
from keras.utils import to_categorical      # type: ignore

if tf.test.gpu_device_name():
    logging.info("GPU TF détecté")
else:
    logging.info("CPU TF uniquement")

X_emb      = np.load(NPY_TRAIN)    # (N_train, MAX_LEN, 1280)
X_test_emb = np.load(NPY_TEST)     # (N_test,  MAX_LEN, 1280)
logging.info(f"Embeddings train chargés : {X_emb.shape}")
logging.info(f"Embeddings test  chargés : {X_test_emb.shape}")


def build_cnn(input_shape=(MAX_LEN, 1280)):
    inp = Input(shape=input_shape)

    x3 = Conv1D(128, 3, padding="same", name="conv_k3")(inp)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    x3 = Dropout(0.20)(x3)

    x5 = Conv1D(128, 5, padding="same", name="conv_k5")(inp)
    x5 = BatchNormalization()(x5)
    x5 = Activation("relu")(x5)
    x5 = Dropout(0.20)(x5)

    x7 = Conv1D(128, 7, padding="same", name="conv_k7")(inp)
    x7 = BatchNormalization()(x7)
    x7 = Activation("relu")(x7)
    x7 = Dropout(0.20)(x7)

    x = Concatenate(name="merge")([x3, x5, x7])  # (MAX_LEN, 384)

    x_skip = Conv1D(256, 1, padding="same", name="skip_proj")(x)
    x_skip = BatchNormalization()(x_skip)

    x = Conv1D(256, 3, padding="same", name="conv_res1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(256, 3, padding="same", name="conv_res2")(x)
    x = BatchNormalization()(x)
    x = Add(name="residual")([x, x_skip])
    x = Activation("relu")(x)
    x = Dropout(0.30)(x)

    x = GlobalAveragePooling1D(name="gap")(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4), name="fc1")(x)
    x = Dropout(0.30)(x)
    x = Dense(2, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=x, name=f"ESM2_DoRA_CNN_{TASK_NAME}")


def train_cnn(X_tr, y_tr, X_val, y_val):
    model = build_cnn()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=LR_CNN),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    cbs = [
        EarlyStopping(monitor="val_auc", mode="max", patience=PATIENCE_CNN_ES,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                          patience=PATIENCE_CNN_LR, min_lr=1e-7, verbose=0),
        ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max",
                        save_best_only=True, verbose=0),
    ]
    model.fit(
        X_tr, to_categorical(y_tr, 2),
        validation_data=(X_val, to_categorical(y_val, 2)),
        epochs=EPOCHS_CNN, batch_size=BATCH_CNN,
        callbacks=cbs, verbose=0,
    )
    return load_model(MODEL_PATH)


kf_cnn    = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
metrics   = {m: [] for m in ["ACC", "BACC", "Sn", "Sp", "MCC", "AUC"]}
thresholds = []

logging.info(f"Début CNN K-fold CV ({K_FOLDS} folds)")

for fold, (tr_idx, val_idx) in enumerate(kf_cnn.split(X_emb, y_train)):
    logging.info(f"--- CNN Fold {fold + 1}/{K_FOLDS} ---")
    X_tr, X_val = X_emb[tr_idx], X_emb[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    best_model = train_cnn(X_tr, y_tr, X_val, y_val)
    proba = best_model.predict(X_val, batch_size=BATCH_CNN, verbose=0)[:, 1]

    fpr, tpr, thresh = roc_curve(y_val, proba)
    opt_thresh = float(thresh[np.argmax(tpr - fpr)])
    thresholds.append(opt_thresh)

    pred = (proba >= opt_thresh).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()

    Sn   = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    Sp   = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    BACC = 0.5 * Sn + 0.5 * Sp if not (math.isnan(Sn) or math.isnan(Sp)) else float("nan")

    metrics["ACC"].append((TP + TN) / (TP + TN + FP + FN))
    metrics["Sn"].append(Sn)
    metrics["Sp"].append(Sp)
    metrics["BACC"].append(BACC)
    metrics["MCC"].append(matthews_corrcoef(y_val, pred))
    metrics["AUC"].append(roc_auc_score(y_val, proba))

    logging.info(
        f"  seuil={opt_thresh:.3f} | AUC={metrics['AUC'][-1]:.4f} | "
        f"MCC={metrics['MCC'][-1]:.4f} | TP={TP} FP={FP} FN={FN} TN={TN}"
    )
    keras.backend.clear_session()
    gc.collect()

logging.info("\nRésultats CNN Cross-Validation (Moyenne ± Écart-Type) :")
for name, vals in metrics.items():
    logging.info(f"  {name:4s}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")
logging.info(f"  Seuil Youden moyen : {np.mean(thresholds):.4f} ± {np.std(thresholds):.4f}")

# =============================================================================
# Évaluation finale — test set externe et indépendant
# =============================================================================
logging.info("\n=== Évaluation finale — test set externe (jamais vu) ===")
final_model  = load_model(MODEL_PATH)
proba_test   = final_model.predict(X_test_emb, batch_size=BATCH_CNN, verbose=0)[:, 1]
mean_thresh  = np.mean(thresholds)
pred_test    = (proba_test >= mean_thresh).astype(int)

TN, FP, FN, TP = confusion_matrix(y_test, pred_test, labels=[0, 1]).ravel()
Sn_test = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
Sp_test = TN / (TN + FP) if (TN + FP) > 0 else float("nan")

logging.info(f"  seuil={mean_thresh:.4f} | TP={TP} FP={FP} FN={FN} TN={TN}")
logging.info(f"  ACC  : {(TP + TN) / (TP + TN + FP + FN):.4f}")
logging.info(f"  Sn   : {Sn_test:.4f}")
logging.info(f"  Sp   : {Sp_test:.4f}")
logging.info(f"  MCC  : {matthews_corrcoef(y_test, pred_test):.4f}")
logging.info(f"  AUC  : {roc_auc_score(y_test, proba_test):.4f}")
logging.info(f"  -> Seuil à utiliser dans le pipeline de prédiction : {mean_thresh:.4f}")
