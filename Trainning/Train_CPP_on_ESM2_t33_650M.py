import os
import gc
import math
import logging

import torch       # type: ignore
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import esm         # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearnex import patch_sklearn # type: ignore
patch_sklearn()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.makedirs("Logs", exist_ok=True)
os.makedirs('ModelCNN_PROTEOGEN_V3', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Logs/Token_Train_CPPV2.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ---------------------------------------------------------------------------
# Hyperparamètres globaux
# ---------------------------------------------------------------------------
MAX_LEN = 50
ESM_BATCH = 16
TRAIN_BATCH = 64
EPOCHS = 200
LR = 3e-4
PATIENCE_ES = 15
PATIENCE_LR = 8
NPY_PATH = 'dataset_train_esm2_t33_650M_CPP_token_level.npy'
MODEL_PATH = 'ModelCNN_PROTEOGEN_V3/CPP_PROTEOGEN_model.keras'

# ---------------------------------------------------------------------------
# Embeddings token-level (Conv1D sémantiquement correct sur positions AA)
# ---------------------------------------------------------------------------
def esm_embeddings_batch(esm2, esm2_alphabet, sequences, max_len=MAX_LEN, batch_size=ESM_BATCH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm2 = esm2.eval().to(device)
    batch_converter = esm2_alphabet.get_batch_converter()
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        batch_input = [(s, s) for s in batch_seqs]
        _, _, batch_tokens = batch_converter(batch_input)
        batch_lens = (batch_tokens != esm2_alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = esm2(batch_tokens, repr_layers=[33], return_contacts=False)

        token_reps = results["representations"][33].cpu().numpy()  # (B, L+2, 1280)

        for j, seq_len in enumerate(batch_lens):
            rep = token_reps[j, 1:seq_len - 1]  # strip BOS/EOS → (L, 1280)
            L = rep.shape[0]
            if L >= max_len:
                padded = rep[:max_len]
            else:
                pad = np.zeros((max_len - L, 1280), dtype=np.float32)
                padded = np.concatenate([rep, pad], axis=0)
            all_embeddings.append(padded)

        del batch_tokens, results, token_reps
        gc.collect()
        logging.info(f"  ESM2 : {min(i + batch_size, len(sequences))}/{len(sequences)} séquences")

    return np.array(all_embeddings, dtype=np.float32)  # (N, MAX_LEN, 1280)

# ---------------------------------------------------------------------------
# Chargement données
# ---------------------------------------------------------------------------
logging.info("Chargement ESM-2 t33 650M")
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

logging.info("Lecture du dataset")
dataset = pd.read_excel('Train_CPP.xlsx', na_filter=False)
sequence_list = list(dataset['sequence'])
y = np.array(dataset['label'])

logging.info(f"Dataset : {len(y)} séquences | classes {dict(zip(*np.unique(y, return_counts=True)))}")

# ---------------------------------------------------------------------------
# Génération embeddings (ou chargement si déjà calculés)
# ---------------------------------------------------------------------------
if os.path.exists(NPY_PATH):
    logging.info(f"Embeddings existants chargés : {NPY_PATH}")
    X = np.load(NPY_PATH)
else:
    logging.info("Génération des embeddings token-level")
    X = esm_embeddings_batch(esm_model, alphabet, sequence_list)
    np.save(NPY_PATH, X)
    logging.info(f"Embeddings sauvegardés : {NPY_PATH} — shape {X.shape}")

del esm_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

logging.info(f"Shape embeddings : {X.shape}")  # (N, MAX_LEN, 1280)

# ---------------------------------------------------------------------------
# Imports Keras/TF
# ---------------------------------------------------------------------------
import keras                            # type: ignore
import tensorflow as tf                 # type: ignore
from keras.layers import (              # type: ignore
    Input, Dense, Activation, BatchNormalization,
    Conv1D, Dropout, GlobalAveragePooling1D,
    Concatenate, Add
)
from keras.models import Model, load_model  # type: ignore
from keras.optimizers import Adam           # type: ignore
from keras.callbacks import (               # type: ignore
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from keras.regularizers import l2           # type: ignore
from keras.utils import to_categorical      # type: ignore
from sklearn.model_selection import train_test_split, StratifiedKFold  # type: ignore
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef  # type: ignore

if tf.test.gpu_device_name():
    logging.info('GPU trouvé')
else:
    logging.info("CPU uniquement")

# ---------------------------------------------------------------------------
# Architecture CNN sur token-level embeddings (MAX_LEN, 1280)
# ---------------------------------------------------------------------------
def build_cnn(input_shape=(MAX_LEN, 1280)):
    inp = Input(shape=input_shape)

    x3 = Conv1D(128, 3, padding='same', name='conv_k3')(inp)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.20)(x3)

    x5 = Conv1D(128, 5, padding='same', name='conv_k5')(inp)
    x5 = BatchNormalization()(x5)
    x5 = Activation('relu')(x5)
    x5 = Dropout(0.20)(x5)

    x7 = Conv1D(128, 7, padding='same', name='conv_k7')(inp)
    x7 = BatchNormalization()(x7)
    x7 = Activation('relu')(x7)
    x7 = Dropout(0.20)(x7)

    x = Concatenate(name='merge')([x3, x5, x7])  # (MAX_LEN, 384)

    x_skip = Conv1D(256, 1, padding='same', name='skip_proj')(x)
    x_skip = BatchNormalization()(x_skip)

    x = Conv1D(256, 3, padding='same', name='conv_res1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same', name='conv_res2')(x)
    x = BatchNormalization()(x)
    x = Add(name='residual')([x, x_skip])
    x = Activation('relu')(x)
    x = Dropout(0.30)(x)

    x = GlobalAveragePooling1D(name='gap')(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4), name='fc1')(x)
    x = Dropout(0.30)(x)
    x = Dense(2, activation='softmax', name='output')(x)

    return Model(inputs=inp, outputs=x, name='ESM_CNN_CPP')


def train_model(X_tr, y_tr, X_val, y_val, model_path):
    y_tr_cat = to_categorical(y_tr, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)

    model = build_cnn()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=LR),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=PATIENCE_ES,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5,
                          patience=PATIENCE_LR, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, monitor='val_auc', mode='max',
                        save_best_only=True, verbose=0)
    ]

    model.fit(X_tr, y_tr_cat, validation_data=(X_val, y_val_cat),
              epochs=EPOCHS, batch_size=TRAIN_BATCH,
              callbacks=callbacks, verbose=1)

    return load_model(model_path)

# ---------------------------------------------------------------------------
# Split train/test stratifié
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)
logging.info(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# Cross-validation 10-fold
# ---------------------------------------------------------------------------
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
metrics = {m: [] for m in ['ACC', 'BACC', 'Sn', 'Sp', 'MCC', 'AUC']}
thresholds = []

logging.info("Début cross-validation 10-fold")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    logging.info(f"--- Fold {fold + 1}/10 ---")
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    saved_model = train_model(X_tr, y_tr, X_val, y_val, MODEL_PATH)

    proba = saved_model.predict(X_val, batch_size=TRAIN_BATCH)[:, 1]
    y_true = y_val

    fpr, tpr, thresh = roc_curve(y_true, proba)
    opt_thresh = float(thresh[np.argmax(tpr - fpr)])
    thresholds.append(opt_thresh)

    pred_class = (proba >= opt_thresh).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_true, pred_class, labels=[0, 1]).ravel()
    logging.info(f"  seuil={opt_thresh:.3f} | TP={TP} FP={FP} FN={FN} TN={TN}")

    Sn = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
    Sp = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
    BACC = 0.5 * Sn + 0.5 * Sp if not (math.isnan(Sn) or math.isnan(Sp)) else float('nan')

    metrics['ACC'].append((TP + TN) / (TP + TN + FP + FN))
    metrics['Sn'].append(Sn)
    metrics['Sp'].append(Sp)
    metrics['BACC'].append(BACC)
    metrics['MCC'].append(matthews_corrcoef(y_true, pred_class))
    metrics['AUC'].append(roc_auc_score(y_true, proba))

    keras.backend.clear_session()
    gc.collect()

logging.info("\nRésultats Cross-Validation (Moyenne ± Écart-Type) :")
for name, vals in metrics.items():
    logging.info(f"{name:4s}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")
logging.info(f"Seuil Youden moyen : {np.mean(thresholds):.4f} ± {np.std(thresholds):.4f}")

# ---------------------------------------------------------------------------
# Évaluation finale sur hold-out test set
# ---------------------------------------------------------------------------
logging.info("\nÉvaluation finale sur X_test (hold-out 20%)")
final_model = load_model(MODEL_PATH)
proba_test = final_model.predict(X_test, batch_size=TRAIN_BATCH)[:, 1]
mean_thresh = np.mean(thresholds)
pred_test = (proba_test >= mean_thresh).astype(int)

TN, FP, FN, TP = confusion_matrix(y_test, pred_test, labels=[0, 1]).ravel()
Sn_test = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
Sp_test = TN / (TN + FP) if (TN + FP) > 0 else float('nan')

logging.info(f"  seuil={mean_thresh:.4f} | TP={TP} FP={FP} FN={FN} TN={TN}")
logging.info(f"  ACC  : {(TP + TN) / (TP + TN + FP + FN):.4f}")
logging.info(f"  Sn   : {Sn_test:.4f}")
logging.info(f"  Sp   : {Sp_test:.4f}")
logging.info(f"  MCC  : {matthews_corrcoef(y_test, pred_test):.4f}")
logging.info(f"  AUC  : {roc_auc_score(y_test, proba_test):.4f}")
logging.info(f"  -> Seuil à utiliser dans le pipeline de prédiction : {mean_thresh:.4f}")
