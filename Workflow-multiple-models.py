"""
Multi-Model Benchmark - Molecular Property Prediction (pEC50)
=============================================================
Modelli: Chemprop (MPNN), AttentiveFP, GIN, Random Forest, XGBoost
HPO    : Optuna
Salva  : modelli su disco + predizioni su test originale e file blinded

OTTIMIZZAZIONI rispetto alla versione originale:
- MorganGenerator al posto di GetMorganFingerprintAsBitVect (fix deprecation warning)
- Fingerprint calcolati in batch con GetFingerprints() (parallelizzato da RDKit)
- num_workers=0 nel DataLoader PyG (fix crash Windows multiprocessing)
- n_jobs=1 per Optuna su modelli PyG/Chemprop (evita spawn conflicts su Windows)
- n_jobs=-1 solo per RF/XGBoost (thread-safe, nessun problema)
- Optuna pruning (MedianPruner) per tagliare trial cattivi in anticipo
- ReduceLROnPlateau con patience più aggressiva in HPO
- pin_memory=True nel DataLoader se CUDA disponibile
- torch.set_float32_matmul_precision per speedup su GPU Ampere+
- Tutto il codice eseguibile sotto if __name__ == "__main__" (fix Windows spawn)
"""

# ═══════════════════════════════════════════════════════════
#  MODIFICA QUESTI PARAMETRI
# ═══════════════════════════════════════════════════════════

CSV_TRAIN   = r"C:\Users\Alessio Macorano\Desktop\calcoli\training_set_new.csv"
CSV_TEST    = r"C:\Users\Alessio Macorano\Desktop\calcoli\unblinded_clean.csv"
CSV_BLINDED = r"C:\Users\Alessio Macorano\Desktop\calcoli\pxr-challenge_TEST_BLINDED.csv"
SMILES_COL  = "SMILES"
TARGET_COL  = "pEC50"
N_TRIALS    = 30
EPOCHS_HPO  = 30
EPOCHS_FINAL = 50
SAVE_DIR    = r"C:\Users\Alessio Macorano\Desktop\calcoli\Risultati_Benchmark"
VAL_FRAC    = 0.1
FP_NBITS    = 2048
FP_RADIUS   = 2
RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
import optuna
from optuna.pruners import MedianPruner

# ── Thread tuning (prima di importare tutto il resto) ──────
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(1)
# Speedup su GPU Ampere+ (RTX 30xx/40xx): abilita TF32
torch.set_float32_matmul_precision("high")

np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import DataStructs
# ── FIX DEPRECATION WARNING: usa MorganGenerator ──────────
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from chemprop import data as cpdata, models as cpmodels, nn as cpnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import AttentiveFP, GINConv, global_mean_pool
import torch.nn as tnn
import torch.nn.functional as F

# Genera il generator UNA SOLA VOLTA (oggetto riutilizzabile, più veloce)
_MORGAN_GEN = GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)

PIN_MEMORY = torch.cuda.is_available()  # pin_memory solo se hai GPU


# ─────────────────────────────────────────────
# UTILITY - METRICHE
# ─────────────────────────────────────────────

def rae(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.mean(y_true)))
    return float(num / den) if den != 0 else np.nan


def compute_metrics(y_true, y_pred, split_name=""):
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae  = mean_absolute_error(y_true, y_pred)
    rae_ = rae(y_true, y_pred)
    sp   = stats.spearmanr(y_true, y_pred)
    kd   = stats.kendalltau(y_true, y_pred)
    return {
        f"{split_name}_R2":         round(r2,   6),
        f"{split_name}_RMSE":       round(rmse, 6),
        f"{split_name}_MAE":        round(mae,  6),
        f"{split_name}_RAE":        round(rae_, 6),
        f"{split_name}_Spearman":   round(sp.statistic, 6),
        f"{split_name}_Spearman_p": sp.pvalue,
        f"{split_name}_Kendall":    round(kd.statistic, 6),
        f"{split_name}_Kendall_p":  kd.pvalue,
    }


def print_metrics(metrics, model_name, split):
    pref = split + "_"
    print(f"  [{model_name} | {split}]"
          f"  R²={metrics[pref+'R2']:.4f}"
          f"  RMSE={metrics[pref+'RMSE']:.4f}"
          f"  MAE={metrics[pref+'MAE']:.4f}"
          f"  Spearman={metrics[pref+'Spearman']:.4f}")


# ─────────────────────────────────────────────
# 2. MORGAN FINGERPRINTS (ottimizzato)
# ─────────────────────────────────────────────

def smiles_to_fp(smiles_list):
    """
    Usa MorganGenerator + GetFingerprints() batch (più veloce del loop singolo).
    GetFingerprints parallelizza internamente con numThreads.
    """
    mols = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_idx.append(i)

    fps_rdkit = _MORGAN_GEN.GetFingerprints(mols, numThreads=os.cpu_count())

    result = np.zeros((len(smiles_list), FP_NBITS), dtype=np.float32)
    for out_i, rdkit_fp in zip(valid_idx, fps_rdkit):
        DataStructs.ConvertToNumpyArray(rdkit_fp, result[out_i])
    return result


# ─────────────────────────────────────────────
# 3. PYGEOMETRIC UTILS
# ─────────────────────────────────────────────

def mol_to_pyg(smi, y=None):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            int(atom.GetHybridization()), int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
            int(atom.IsInRing()), atom.GetMass() / 100.0,
        ])
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = [float(bond.GetBondTypeAsDouble()), int(bond.GetIsConjugated()),
              int(bond.IsInRing()), int(bond.GetStereo())]
        edge_index += [[i, j], [j, i]]
        edge_attr  += [bf, bf]
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([[y]], dtype=torch.float)
    return data


def build_pyg_dataset(smiles_list, targets=None):
    dataset = []
    for i, smi in enumerate(smiles_list):
        y = targets[i] if targets is not None else None
        d = mol_to_pyg(smi, y)
        if d is not None:
            dataset.append(d)
    return dataset


N_ATOM_FEAT = 9
N_EDGE_FEAT = 4


def pyg_loader(dataset, batch_size=64, shuffle=False):
    """
    num_workers=0 è obbligatorio su Windows per evitare il crash
    multiprocessing con Optuna n_jobs > 1.
    pin_memory=True accelera il trasferimento CPU→GPU se disponibile.
    """
    return PyGLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )


# ─────────────────────────────────────────────
# 4. MODELLI PYTORCH GEOMETRIC
# ─────────────────────────────────────────────

class AttentiveFPModel(tnn.Module):
    def __init__(self, hidden_channels, num_layers, num_timesteps, dropout):
        super().__init__()
        self.gnn = AttentiveFP(
            in_channels=N_ATOM_FEAT, hidden_channels=hidden_channels,
            out_channels=1, edge_dim=N_EDGE_FEAT,
            num_layers=num_layers, num_timesteps=num_timesteps, dropout=dropout)

    def forward(self, data):
        return self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)


class GINModel(tnn.Module):
    def __init__(self, hidden_dim, n_layers, dropout):
        super().__init__()
        self.convs = tnn.ModuleList()
        self.bns   = tnn.ModuleList()
        in_dim = N_ATOM_FEAT
        for _ in range(n_layers):
            self.convs.append(GINConv(tnn.Sequential(
                tnn.Linear(in_dim, hidden_dim), tnn.ReLU(),
                tnn.Linear(hidden_dim, hidden_dim))))
            self.bns.append(tnn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        self.dropout = dropout
        self.head = tnn.Sequential(
            tnn.Linear(hidden_dim, hidden_dim // 2), tnn.ReLU(),
            tnn.Dropout(dropout), tnn.Linear(hidden_dim // 2, 1))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(global_mean_pool(x, batch))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_pyg(model, loader, optimizer):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(batch).squeeze(-1)
        loss = F.mse_loss(out, batch.y.squeeze(-1))
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def predict_pyg(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        preds.append(model(batch).squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def evaluate_pyg(model, loader):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        preds.append(model(batch).squeeze(-1).cpu().numpy())
        trues.append(batch.y.squeeze(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


def fit_pyg(model, ds_train, ds_val, epochs, lr, batch_size=64,
            trial=None):
    """
    Aggiunto supporto opzionale al pruning Optuna:
    se `trial` è passato, riporta l'RMSE intermedio e solleva
    TrialPruned se il trial è chiaramente cattivo.
    """
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Patience più aggressiva durante HPO per velocizzare
    patience = 3 if trial is not None else 5
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=patience, factor=0.5)
    val_loader = pyg_loader(ds_val, batch_size)
    for epoch in range(epochs):
        train_pyg(model, pyg_loader(ds_train, batch_size, shuffle=True), opt)
        vp, vt = evaluate_pyg(model, val_loader)
        val_rmse = float(np.sqrt(np.mean((vp - vt) ** 2)))
        sch.step(val_rmse)
        # Pruning Optuna: segnala il valore intermedio e interrompe se necessario
        if trial is not None:
            trial.report(val_rmse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return model


# ─────────────────────────────────────────────
# 5. CHEMPROP UTILS
# ─────────────────────────────────────────────

def cp_dataset_with_targets(smiles_list, target_array):
    tgt = target_array.reshape(-1, 1)
    return cpdata.MoleculeDataset(
        [cpdata.MoleculeDatapoint.from_smi(s, tgt[i])
         for i, s in enumerate(smiles_list)])


def cp_dataset_inference(smiles_list):
    return cpdata.MoleculeDataset(
        [cpdata.MoleculeDatapoint.from_smi(s) for s in smiles_list])


def cp_loader(ds, shuffle=False, batch_size=64):
    return cpdata.build_dataloader(ds, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=0)


def build_chemprop(params):
    mp   = cpnn.BondMessagePassing(d_h=params["d_h"], depth=params["depth"],
                                   dropout=params["dropout"])
    pred = cpnn.RegressionFFN(input_dim=params["d_h"], hidden_dim=params["hidden_dim"],
                              n_layers=params["n_layers"], dropout=params["dropout"])
    return cpmodels.MPNN(message_passing=mp, agg=cpnn.MeanAggregation(),
                         predictor=pred, max_lr=params["max_lr"])


def cp_trainer(epochs):
    return pl.Trainer(max_epochs=epochs, enable_progress_bar=False,
                      enable_model_summary=False, logger=False, accelerator="auto")


def cp_predict(trainer, model, loader):
    return torch.cat(trainer.predict(model, loader)).numpy().flatten()


# ═══════════════════════════════════════════════════════════
# TUTTO IL CODICE ESEGUIBILE VA QUI SOTTO
# Questo è il fix fondamentale per Windows multiprocessing
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ─────────────────────────────────────────────
    # 1. CARICAMENTO DATI
    # ─────────────────────────────────────────────
    print("\nCarico dati...")
    df_train   = pd.read_csv(CSV_TRAIN)
    df_test    = pd.read_csv(CSV_TEST)
    df_blinded = pd.read_csv(CSV_BLINDED)
    print(f"  Train: {len(df_train)} | Test: {len(df_test)} | Blinded: {len(df_blinded)}")

    smiles_train   = df_train[SMILES_COL].tolist()
    targets_train  = df_train[TARGET_COL].values.astype(float)
    smiles_test    = df_test[SMILES_COL].tolist()
    targets_test   = df_test[TARGET_COL].values.astype(float)
    smiles_blinded = df_blinded[SMILES_COL].tolist()

    n         = len(smiles_train)
    idx       = np.random.permutation(n)
    n_val     = int(n * VAL_FRAC)
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    smiles_tr   = [smiles_train[i] for i in train_idx]
    targets_tr  = targets_train[train_idx]
    smiles_val  = [smiles_train[i] for i in val_idx]
    targets_val = targets_train[val_idx]

    print(f"  Train HPO: {len(smiles_tr)} | Val: {len(smiles_val)}")
    print(f"  Device: {DEVICE}")

    # ─────────────────────────────────────────────
    # 2. MORGAN FINGERPRINTS
    # ─────────────────────────────────────────────
    print("\nCalcolo Morgan fingerprints...")
    X_tr      = smiles_to_fp(smiles_tr)
    X_val     = smiles_to_fp(smiles_val)
    X_all     = smiles_to_fp(smiles_train)
    X_test    = smiles_to_fp(smiles_test)
    X_blinded = smiles_to_fp(smiles_blinded)
    print("  Done.")

    # ─────────────────────────────────────────────
    # 3. PYGEOMETRIC DATASET
    # ─────────────────────────────────────────────
    print("\nCostruzione dataset PyG...")
    pyg_tr      = build_pyg_dataset(smiles_tr,      targets_tr)
    pyg_val     = build_pyg_dataset(smiles_val,     targets_val)
    pyg_all     = build_pyg_dataset(smiles_train,   targets_train)
    pyg_test    = build_pyg_dataset(smiles_test,    targets_test)
    pyg_blinded = build_pyg_dataset(smiles_blinded)
    print("  Done.")

    all_results = {}

    ds_cp_tr  = cp_dataset_with_targets(smiles_tr,    targets_tr)
    ds_cp_val = cp_dataset_with_targets(smiles_val,   targets_val)
    ds_cp_all = cp_dataset_with_targets(smiles_train, targets_train)
    ds_cp_tst = cp_dataset_with_targets(smiles_test,  targets_test)

    # ── Chemprop ──────────────────────────────────
    print("\n" + "=" * 60)
    print("HPO: Chemprop (MPNN)")
    print("=" * 60)

    def obj_chemprop(trial):
        p = {"depth":      trial.suggest_int("depth", 2, 6),
             "d_h":        trial.suggest_categorical("d_h", [64, 128, 256, 512]),
             "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
             "n_layers":   trial.suggest_int("n_layers", 1, 3),
             "dropout":    trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
             "max_lr":     trial.suggest_float("max_lr", 1e-4, 1e-3, log=True)}
        m = build_chemprop(p)
        t = cp_trainer(EPOCHS_HPO)
        t.fit(m, cp_loader(ds_cp_tr, shuffle=True), cp_loader(ds_cp_val))
        pr = cp_predict(t, m, cp_loader(ds_cp_val))
        y  = np.array([ds_cp_val[i].y[0] for i in range(len(ds_cp_val))])
        return float(np.sqrt(np.mean((pr - y) ** 2)))

    # n_jobs=1 per Chemprop: usa già Lightning/multiprocessing internamente
    study_cp = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study_cp.optimize(obj_chemprop, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)
    best_cp = study_cp.best_trial.params
    print(f"  Best val RMSE: {study_cp.best_trial.value:.4f} | {best_cp}")

    final_cp = build_chemprop(best_cp)
    tr_final = cp_trainer(EPOCHS_FINAL)
    tr_final.fit(final_cp, cp_loader(ds_cp_all, shuffle=True))

    p_tr_cp  = cp_predict(tr_final, final_cp, cp_loader(ds_cp_tr))
    p_val_cp = cp_predict(tr_final, final_cp, cp_loader(ds_cp_val))
    p_tst_cp = cp_predict(tr_final, final_cp, cp_loader(ds_cp_tst))

    m_cp = {}
    m_cp.update(compute_metrics(targets_tr,   p_tr_cp,  "train"))
    m_cp.update(compute_metrics(targets_val,  p_val_cp, "val"))
    m_cp.update(compute_metrics(targets_test, p_tst_cp, "test"))
    print_metrics(m_cp, "Chemprop", "train")
    print_metrics(m_cp, "Chemprop", "val")
    print_metrics(m_cp, "Chemprop", "test")
    all_results["Chemprop_MPNN"] = {"best_params": best_cp,
                                    "hpo_best_val_rmse": study_cp.best_trial.value,
                                    "metrics": m_cp}
    study_cp.trials_dataframe().to_csv(os.path.join(SAVE_DIR, "trials_chemprop.csv"), index=False)
    tr_final.save_checkpoint(os.path.join(SAVE_DIR, "model_chemprop.ckpt"))

    # ── AttentiveFP ───────────────────────────────
    print("\n" + "=" * 60)
    print("HPO: AttentiveFP")
    print("=" * 60)

    def obj_afp(trial):
        p = {"hidden_channels": trial.suggest_categorical("hidden_channels", [64, 128, 256]),
             "num_layers":      trial.suggest_int("num_layers", 2, 5),
             "num_timesteps":   trial.suggest_int("num_timesteps", 2, 5),
             "dropout":         trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
             "lr":              trial.suggest_float("lr", 1e-4, 1e-3, log=True),
             "batch_size":      trial.suggest_categorical("batch_size", [32, 64, 128])}
        m = AttentiveFPModel(p["hidden_channels"], p["num_layers"],
                             p["num_timesteps"], p["dropout"])
        # Passa trial per abilitare il pruning epoch-by-epoch
        fit_pyg(m, pyg_tr, pyg_val, EPOCHS_HPO, p["lr"], p["batch_size"], trial=trial)
        vp, vt = evaluate_pyg(m, pyg_loader(pyg_val))
        return float(np.sqrt(np.mean((vp - vt) ** 2)))

    # n_jobs=1 per PyG: DataLoader con num_workers=0 non tollera spawn multipli
    study_afp = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study_afp.optimize(obj_afp, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)
    best_afp = study_afp.best_trial.params
    print(f"  Best val RMSE: {study_afp.best_trial.value:.4f} | {best_afp}")

    final_afp = AttentiveFPModel(best_afp["hidden_channels"], best_afp["num_layers"],
                                 best_afp["num_timesteps"], best_afp["dropout"])
    fit_pyg(final_afp, pyg_all, pyg_val, EPOCHS_FINAL,
            best_afp["lr"], best_afp["batch_size"])

    p_tr_afp  = predict_pyg(final_afp, pyg_loader(pyg_tr))
    p_val_afp = predict_pyg(final_afp, pyg_loader(pyg_val))
    p_tst_afp = predict_pyg(final_afp, pyg_loader(pyg_test))

    m_afp = {}
    m_afp.update(compute_metrics(targets_tr,   p_tr_afp,  "train"))
    m_afp.update(compute_metrics(targets_val,  p_val_afp, "val"))
    m_afp.update(compute_metrics(targets_test, p_tst_afp, "test"))
    print_metrics(m_afp, "AttentiveFP", "train")
    print_metrics(m_afp, "AttentiveFP", "val")
    print_metrics(m_afp, "AttentiveFP", "test")
    all_results["AttentiveFP"] = {"best_params": best_afp,
                                  "hpo_best_val_rmse": study_afp.best_trial.value,
                                  "metrics": m_afp}
    study_afp.trials_dataframe().to_csv(os.path.join(SAVE_DIR, "trials_attentivefp.csv"), index=False)
    torch.save({"params": best_afp, "state_dict": final_afp.state_dict()},
               os.path.join(SAVE_DIR, "model_afp.pt"))

    # ── GIN ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("HPO: GIN")
    print("=" * 60)

    def obj_gin(trial):
        p = {"hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
             "n_layers":   trial.suggest_int("n_layers", 2, 5),
             "dropout":    trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
             "lr":         trial.suggest_float("lr", 1e-4, 1e-3, log=True),
             "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128])}
        m = GINModel(p["hidden_dim"], p["n_layers"], p["dropout"])
        fit_pyg(m, pyg_tr, pyg_val, EPOCHS_HPO, p["lr"], p["batch_size"], trial=trial)
        vp, vt = evaluate_pyg(m, pyg_loader(pyg_val))
        return float(np.sqrt(np.mean((vp - vt) ** 2)))

    study_gin = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study_gin.optimize(obj_gin, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)
    best_gin = study_gin.best_trial.params
    print(f"  Best val RMSE: {study_gin.best_trial.value:.4f} | {best_gin}")

    final_gin = GINModel(best_gin["hidden_dim"], best_gin["n_layers"], best_gin["dropout"])
    fit_pyg(final_gin, pyg_all, pyg_val, EPOCHS_FINAL,
            best_gin["lr"], best_gin["batch_size"])

    p_tr_gin  = predict_pyg(final_gin, pyg_loader(pyg_tr))
    p_val_gin = predict_pyg(final_gin, pyg_loader(pyg_val))
    p_tst_gin = predict_pyg(final_gin, pyg_loader(pyg_test))

    m_gin = {}
    m_gin.update(compute_metrics(targets_tr,   p_tr_gin,  "train"))
    m_gin.update(compute_metrics(targets_val,  p_val_gin, "val"))
    m_gin.update(compute_metrics(targets_test, p_tst_gin, "test"))
    print_metrics(m_gin, "GIN", "train")
    print_metrics(m_gin, "GIN", "val")
    print_metrics(m_gin, "GIN", "test")
    all_results["GIN"] = {"best_params": best_gin,
                          "hpo_best_val_rmse": study_gin.best_trial.value,
                          "metrics": m_gin}
    study_gin.trials_dataframe().to_csv(os.path.join(SAVE_DIR, "trials_gin.csv"), index=False)
    torch.save({"params": best_gin, "state_dict": final_gin.state_dict()},
               os.path.join(SAVE_DIR, "model_gin.pt"))

    # ── Random Forest ─────────────────────────────
    print("\n" + "=" * 60)
    print("HPO: Random Forest")
    print("=" * 60)

    def obj_rf(trial):
        p = {"n_estimators":      trial.suggest_int("n_estimators", 100, 800, step=100),
             "max_depth":         trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
             "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
             "max_features":      trial.suggest_categorical("max_features",
                                                             ["sqrt", "log2", 0.3, 0.5, 0.7])}
        m = RandomForestRegressor(**p, random_state=RANDOM_SEED, n_jobs=-1)
        m.fit(X_tr, targets_tr)
        return float(np.sqrt(np.mean((m.predict(X_val) - targets_val) ** 2)))

    # RF/XGBoost: n_jobs=-1 è sicuro, usano thread non processi
    study_rf = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study_rf.optimize(obj_rf, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)
    best_rf = study_rf.best_trial.params
    print(f"  Best val RMSE: {study_rf.best_trial.value:.4f} | {best_rf}")

    final_rf = RandomForestRegressor(**best_rf, random_state=RANDOM_SEED, n_jobs=-1)
    final_rf.fit(X_all, targets_train)

    p_tr_rf  = final_rf.predict(X_tr)
    p_val_rf = final_rf.predict(X_val)
    p_tst_rf = final_rf.predict(X_test)

    m_rf = {}
    m_rf.update(compute_metrics(targets_tr,   p_tr_rf,  "train"))
    m_rf.update(compute_metrics(targets_val,  p_val_rf, "val"))
    m_rf.update(compute_metrics(targets_test, p_tst_rf, "test"))
    print_metrics(m_rf, "RF", "train")
    print_metrics(m_rf, "RF", "val")
    print_metrics(m_rf, "RF", "test")
    all_results["RandomForest"] = {"best_params": best_rf,
                                   "hpo_best_val_rmse": study_rf.best_trial.value,
                                   "metrics": m_rf}
    study_rf.trials_dataframe().to_csv(os.path.join(SAVE_DIR, "trials_rf.csv"), index=False)
    with open(os.path.join(SAVE_DIR, "model_rf.pkl"), "wb") as f:
        pickle.dump(final_rf, f)

    # ── XGBoost ───────────────────────────────────
    print("\n" + "=" * 60)
    print("HPO: XGBoost")
    print("=" * 60)

    def obj_xgb(trial):
        p = {"n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=100),
             "max_depth":        trial.suggest_int("max_depth", 3, 10),
             "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
             "subsample":        trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0, step=0.1),
             "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
             "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
             "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)}
        m = XGBRegressor(**p, random_state=RANDOM_SEED, tree_method="hist",
                         verbosity=0, n_jobs=-1)
        m.fit(X_tr, targets_tr, eval_set=[(X_val, targets_val)], verbose=False)
        return float(np.sqrt(np.mean((m.predict(X_val) - targets_val) ** 2)))

    study_xgb = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study_xgb.optimize(obj_xgb, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)
    best_xgb = study_xgb.best_trial.params
    print(f"  Best val RMSE: {study_xgb.best_trial.value:.4f} | {best_xgb}")

    final_xgb = XGBRegressor(**best_xgb, random_state=RANDOM_SEED,
                              tree_method="hist", verbosity=0, n_jobs=-1)
    final_xgb.fit(X_all, targets_train)

    p_tr_xgb  = final_xgb.predict(X_tr)
    p_val_xgb = final_xgb.predict(X_val)
    p_tst_xgb = final_xgb.predict(X_test)

    m_xgb = {}
    m_xgb.update(compute_metrics(targets_tr,   p_tr_xgb,  "train"))
    m_xgb.update(compute_metrics(targets_val,  p_val_xgb, "val"))
    m_xgb.update(compute_metrics(targets_test, p_tst_xgb, "test"))
    print_metrics(m_xgb, "XGBoost", "train")
    print_metrics(m_xgb, "XGBoost", "val")
    print_metrics(m_xgb, "XGBoost", "test")
    all_results["XGBoost"] = {"best_params": best_xgb,
                              "hpo_best_val_rmse": study_xgb.best_trial.value,
                              "metrics": m_xgb}
    study_xgb.trials_dataframe().to_csv(os.path.join(SAVE_DIR, "trials_xgboost.csv"), index=False)
    with open(os.path.join(SAVE_DIR, "model_xgb.pkl"), "wb") as f:
        pickle.dump(final_xgb, f)

    # ─────────────────────────────────────────────
    # 7. SALVATAGGIO PREDIZIONI TEST
    # ─────────────────────────────────────────────
    print("\nSalvo predizioni test...")
    pd.DataFrame({
        "SMILES":           smiles_test,
        TARGET_COL:         targets_test,
        "Chemprop_pred":    p_tst_cp,
        "AttentiveFP_pred": p_tst_afp,
        "GIN_pred":         p_tst_gin,
        "RF_pred":          p_tst_rf,
        "XGBoost_pred":     p_tst_xgb,
    }).to_csv(os.path.join(SAVE_DIR, "predictions_test.csv"), index=False)

    # ─────────────────────────────────────────────
    # 8. RIEPILOGO FINALE
    # ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RIEPILOGO METRICHE TEST SET")
    print("=" * 70)
    print(f"{'Modello':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'RAE':>8} {'Spearman':>10}")
    print("-" * 70)
    for name, res in all_results.items():
        m = res["metrics"]
        print(f"{name:<20}"
              f"  {m['test_R2']:>7.4f}"
              f"  {m['test_RMSE']:>7.4f}"
              f"  {m['test_MAE']:>7.4f}"
              f"  {m['test_RAE']:>7.4f}"
              f"  {m['test_Spearman']:>9.4f}")
    print("=" * 70)

    with open(os.path.join(SAVE_DIR, "benchmark_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    rows = []
    for name, res in all_results.items():
        row = {"model": name, "hpo_best_val_rmse": res["hpo_best_val_rmse"]}
        row.update(res["metrics"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(SAVE_DIR, "benchmark_summary.csv"), index=False)

    with open(os.path.join(SAVE_DIR, "best_params.json"), "w") as f:
        json.dump({"chemprop": best_cp, "afp": best_afp, "gin": best_gin,
                   "rf": best_rf, "xgb": best_xgb}, f, indent=2, default=str)

    print(f"\nModelli salvati in: {SAVE_DIR}")
    print("Benchmark completato!")

    # ─────────────────────────────────────────────
    # 9. INFERENZA FILE BLINDED
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INFERENZA FILE BLINDED")
    print("=" * 60)

    ds_cp_blinded = cp_dataset_inference(smiles_blinded)

    preds_blinded = {
        "Chemprop_MPNN": cp_predict(tr_final, final_cp, cp_loader(ds_cp_blinded)),
        "AttentiveFP":   predict_pyg(final_afp, pyg_loader(pyg_blinded)),
        "GIN":           predict_pyg(final_gin, pyg_loader(pyg_blinded)),
        "RandomForest":  final_rf.predict(X_blinded),
        "XGBoost":       final_xgb.predict(X_blinded),
    }

    best_model_name = min(all_results, key=lambda x: all_results[x]["metrics"]["test_RMSE"])
    print(f"Miglior modello (test RMSE): {best_model_name}")

    df_blinded["pEC50_pred"] = preds_blinded[best_model_name]
    out_path = os.path.join(SAVE_DIR, "predictions_blinded.csv")
    df_blinded.to_csv(out_path, index=False)
    print(f"Predizioni blinded salvate in: {out_path}")