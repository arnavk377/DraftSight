"""
Walk-forward backtesting with college football features for NFL draft 2-year AV.

Models:
  1. Regression Splines (SplineTransformer on pick + Ridge)
  2. XGBoost
  3. TabNet
  4. CatBoost
  5. Random Forest
  6. LightGBM

Input:
  - src/data/raw/draft_picks.csv
  - data/clean_cfb/05_04_all_players_2004_2024.csv
  - scraping_av/data/*_av.csv

Output:
  - poc_outputs_v3/  (CSV + scatter plots + feature importance)
"""

import os
import glob
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt

from src.model_v3.tabnet import TabNetRegressor
from src.model_v3.mlp_embeddings import MLPWithEmbeddings
from src.model_v3.ft_transformer import FTTransformer
from src.model_v3.data_loader import (
    load_cfb, load_draft, load_av_from_year_files, build_two_year_labels,
    join_cfb_to_draft, NUM_COLS, CAT_COLS, DRAFT_NUM_COLS, DRAFT_CAT_COLS,
)

# ── Config ───────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DRAFT_CSV = os.path.join(REPO_ROOT, "src", "data", "raw", "draft_picks.csv")
CFB_CSV   = os.path.join(REPO_ROOT, "data", "clean_cfb", "05_04_all_players_2004_2024.csv")
AV_DIR    = os.path.join(REPO_ROOT, "scraping_av", "data")
OUT_DIR   = os.path.join(REPO_ROOT, "poc_outputs_v3")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Join options
JOIN_ON_COLLEGE = False   # also require college name to match
USE_FUZZY       = False   # use fuzzy name matching
FUZZY_THRESHOLD = 85      # minimum score (0-100) for fuzzy match

WALK_FORWARD_START_YEAR = 2005  # first year with CFB data coverage

# TabNet hyperparameters
TABNET_CFG = dict(
    n_d=16, n_a=16, n_steps=4, gamma=1.5,
    n_shared=2, n_step_dep=2, vbs=64, momentum=0.02, lambda_sparse=1e-3,
)
TRAIN_CFG = dict(
    lr=0.02, lr_decay=0.95, lr_decay_steps=200,
    batch_size=256, max_epochs=500, patience=50, val_fraction=0.2,
)

CATBOOST_CFG = dict(
    iterations=500, learning_rate=0.05, depth=6,
    l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
)

RF_CFG = dict(
    n_estimators=500, max_depth=None, min_samples_leaf=5,
    max_features=0.5, n_jobs=-1, random_state=SEED,
)

LGB_CFG = dict(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, n_jobs=-1, random_state=SEED, verbose=-1,
)

# MLP with entity embeddings
MLPE_CFG = dict(hidden_dims=(256, 128, 64), dropout=0.3)

# FT-Transformer
FTT_CFG = dict(d_token=32, n_heads=4, n_layers=3, dropout=0.1)

# Shared training config for embed-based neural models
EMBED_TRAIN_CFG = dict(
    lr=1e-3, batch_size=128, max_epochs=300, patience=30, val_fraction=0.15,
)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def build_spline_preprocessor(df: pd.DataFrame):
    """SplineTransformer on pick, median imputation on other numerics, OHE on cats."""
    num_cols = [c for c in NUM_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    transformers = []
    if "pick" in num_cols:
        transformers.append(("pick_spline", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("spl", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
        ]), ["pick"]))
        rest = [c for c in num_cols if c != "pick"]
        if rest:
            transformers.append(("num", SimpleImputer(strategy="median"), rest))
    elif num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))

    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def build_tree_preprocessor(df: pd.DataFrame):
    """Median imputation on numerics, OHE on cats (no spline — trees don't need it)."""
    num_cols = [c for c in NUM_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def build_embed_data(df_tr: pd.DataFrame, df_te: pd.DataFrame):
    """Prepare separate numeric and ordinal-encoded categorical arrays for embedding models."""
    num_cols = [c for c in NUM_COLS if c in df_tr.columns]
    cat_cols = [c for c in CAT_COLS if c in df_tr.columns]

    num_imp = SimpleImputer(strategy="median")
    X_num_tr = num_imp.fit_transform(df_tr[num_cols]).astype(np.float32)
    X_num_te = num_imp.transform(df_te[num_cols]).astype(np.float32)

    cat_tr = df_tr[cat_cols].fillna("__nan__").astype(str)
    cat_te = df_te[cat_cols].fillna("__nan__").astype(str)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    X_cat_tr = enc.fit_transform(cat_tr)
    X_cat_te = enc.transform(cat_te)

    # Map unknowns (-1) to the last slot; cardinalities include that extra slot
    cardinalities = [len(cats) + 1 for cats in enc.categories_]
    for j in range(X_cat_tr.shape[1]):
        X_cat_tr[X_cat_tr[:, j] == -1, j] = cardinalities[j] - 1
        X_cat_te[X_cat_te[:, j] == -1, j] = cardinalities[j] - 1

    return X_num_tr, X_num_te, X_cat_tr, X_cat_te, cardinalities


def build_catboost_data(df_tr: pd.DataFrame, df_te: pd.DataFrame):
    """Prepare DataFrames for CatBoost: fill cat NaN with 'nan', return cat feature indices."""
    num_cols = [c for c in NUM_COLS if c in df_tr.columns]
    cat_cols = [c for c in CAT_COLS if c in df_tr.columns]
    all_cols = num_cols + cat_cols
    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    X_tr = df_tr[all_cols].copy()
    X_te = df_te[all_cols].copy()
    for c in cat_cols:
        X_tr[c] = X_tr[c].fillna("nan").astype(str)
        X_te[c] = X_te[c].fillna("nan").astype(str)
    return X_tr, X_te, cat_indices, all_cols


def get_feature_names(preprocessor, num_cols, cat_cols) -> list[str]:
    names = list(num_cols)
    if cat_cols and "cat" in preprocessor.named_transformers_:
        names += list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    return names


# ── Metrics ──────────────────────────────────────────────────────────────────

def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sp   = float(spearmanr(y_true, y_pred).correlation)
    return mae, rmse, sp


# ── TabNet training loop ──────────────────────────────────────────────────────

def train_tabnet(X_tr, y_tr, n_features, verbose=True):
    model = TabNetRegressor(n_features=n_features, **TABNET_CFG).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CFG["lr"])
    scheduler = ExponentialLR(optimizer, gamma=TRAIN_CFG["lr_decay"])

    n_val = max(1, int(len(X_tr) * TRAIN_CFG["val_fraction"]))
    Xt, Xv = X_tr[:-n_val], X_tr[-n_val:]
    yt, yv = y_tr[:-n_val], y_tr[-n_val:]

    loader = DataLoader(
        TensorDataset(torch.tensor(Xt, dtype=torch.float32),
                      torch.tensor(yt, dtype=torch.float32)),
        batch_size=TRAIN_CFG["batch_size"], shuffle=True, drop_last=True,
    )
    Xv_t = torch.tensor(Xv, dtype=torch.float32).to(DEVICE)
    yv_t = torch.tensor(yv, dtype=torch.float32).to(DEVICE)

    best_val, best_state, patience_ctr = float("inf"), None, 0
    train_hist, val_hist = [], []
    step = 0

    for epoch in range(TRAIN_CFG["max_epochs"]):
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred, sp, _ = model(Xb)
            loss = model.loss(pred, yb, sp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            step += 1
            if step % TRAIN_CFG["lr_decay_steps"] == 0:
                scheduler.step()
        train_hist.append(ep_loss / max(1, len(loader)))

        model.eval()
        with torch.no_grad():
            vp, vsp, _ = model(Xv_t)
            vl = model.loss(vp, yv_t, vsp).item()
        val_hist.append(vl)

        if vl < best_val:
            best_val, patience_ctr = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1:4d}  train={train_hist[-1]:.3f}  val={vl:.3f}  "
                  f"best={best_val:.3f}  patience={patience_ctr}/{TRAIN_CFG['patience']}")

        if patience_ctr >= TRAIN_CFG["patience"]:
            if verbose:
                print(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, train_hist, val_hist


# ── Embed-model training loop (shared by MLPE and FTTransformer) ─────────────

def train_embed_model(model, X_num_tr, X_cat_tr, y_tr, verbose=False):
    n_val = max(1, int(len(y_tr) * EMBED_TRAIN_CFG["val_fraction"]))
    Xn_t, Xn_v = X_num_tr[:-n_val], X_num_tr[-n_val:]
    Xc_t, Xc_v = X_cat_tr[:-n_val], X_cat_tr[-n_val:]
    yt, yv = y_tr[:-n_val], y_tr[-n_val:]

    loader = DataLoader(
        TensorDataset(
            torch.tensor(Xn_t, dtype=torch.float32),
            torch.tensor(Xc_t, dtype=torch.long),
            torch.tensor(yt, dtype=torch.float32),
        ),
        batch_size=EMBED_TRAIN_CFG["batch_size"], shuffle=True, drop_last=False,
    )
    Xn_v_t = torch.tensor(Xn_v, dtype=torch.float32).to(DEVICE)
    Xc_v_t = torch.tensor(Xc_v, dtype=torch.long).to(DEVICE)
    yv_t   = torch.tensor(yv,  dtype=torch.float32).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=EMBED_TRAIN_CFG["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EMBED_TRAIN_CFG["max_epochs"])

    best_val, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(EMBED_TRAIN_CFG["max_epochs"]):
        model.train()
        for Xb_n, Xb_c, yb in loader:
            Xb_n, Xb_c, yb = Xb_n.to(DEVICE), Xb_c.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            F.mse_loss(model(Xb_n, Xb_c), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(Xn_v_t, Xc_v_t), yv_t).item()

        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:4d}  val={val_loss:.3f}  best={best_val:.3f}  "
                  f"patience={patience_ctr}/{EMBED_TRAIN_CFG['patience']}")

        if patience_ctr >= EMBED_TRAIN_CFG["patience"]:
            if verbose:
                print(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


# ── Plotting ──────────────────────────────────────────────────────────────────

def _nice_ax(ax):
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def save_pred_vs_actual(y_true, y_pred, out_path, draft_year, model_name):
    mae, rmse, sp = eval_metrics(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=220)
    ax.scatter(y_true, y_pred, s=22, alpha=0.65, edgecolor="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], lw=1.2, alpha=0.85)
    ax.set_title(f"{model_name}: {draft_year} Draft Class", fontsize=13, pad=10)
    ax.set_xlabel(f"Actual 2-Year AV  (AV{draft_year} + AV{draft_year+1})", fontsize=11)
    ax.set_ylabel("Predicted 2-Year AV", fontsize=11)
    ax.text(0.02, 0.98, f"MAE={mae:.2f}  RMSE={rmse:.2f}  Spearman={sp:.2f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.9, ec="0.85"))
    _nice_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_feature_importance(importance, feature_names, out_path, title="Feature Importance"):
    idx = np.argsort(importance)[::-1][:25]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.barh([feature_names[i] for i in reversed(idx)], importance[list(reversed(idx))], color="#4C72B0")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    _nice_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_loss_curves(train_hist, val_hist, out_path, test_year):
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=180)
    ax.plot(train_hist, label="Train", lw=1.4)
    ax.plot(val_hist,   label="Val",   lw=1.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE + λ·sparsity)")
    ax.set_title(f"TabNet Training Curves — Test Year {test_year}", fontsize=13, pad=10)
    ax.legend(fontsize=10)
    _nice_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load data
    print("Loading draft:", DRAFT_CSV)
    draft = load_draft(DRAFT_CSV)

    print("Loading CFB:", CFB_CSV)
    cfb = load_cfb(CFB_CSV)

    print("Loading AVs:", AV_DIR)
    av_long = load_av_from_year_files(AV_DIR)
    labels  = build_two_year_labels(av_long)

    # Join CFB stats onto draft
    merged, join_stats = join_cfb_to_draft(
        draft, cfb,
        join_on_college=JOIN_ON_COLLEGE,
        use_fuzzy=USE_FUZZY,
        fuzzy_threshold=FUZZY_THRESHOLD,
    )
    print(f"Join stats: {join_stats}")

    # Attach AV labels
    df = merged.rename(columns={"season": "draft_season"}).merge(
        labels, on=["draft_season", "pfr_player_id"], how="left"
    )
    df = df.dropna(subset=["av_2yr"]).copy()
    df["av_2yr"] = df["av_2yr"].astype(float)

    set_seeds(SEED)

    feat_cols = [c for c in (NUM_COLS + CAT_COLS) if c in df.columns]
    X_raw = df[feat_cols].copy()
    y = df["av_2yr"].values
    years = df["draft_season"].values

    unique_years = sorted(np.unique(years))
    test_years   = [yr for yr in unique_years if yr > WALK_FORWARD_START_YEAR]
    if not test_years:
        raise ValueError(f"No test years after {WALK_FORWARD_START_YEAR}")

    results = []
    latest_year = max(test_years)
    last_tabnet_state = {}

    for test_year in test_years:
        tr_mask = (years >= WALK_FORWARD_START_YEAR) & (years < test_year)
        te_mask = years == test_year
        X_tr_raw, X_te_raw = X_raw[tr_mask], X_raw[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]

        print(f"\n=== Test {test_year}  train={tr_mask.sum()}  test={te_mask.sum()} ===")

        # ── Spline Ridge ────────────────────────────────────────────────────
        pre_s, _, _ = build_spline_preprocessor(X_tr_raw)
        spline_model = Pipeline([("pre", pre_s), ("reg", Ridge(alpha=1.0))])
        spline_model.fit(X_tr_raw, y_tr)
        pred_s = spline_model.predict(X_te_raw)
        mae_s, rmse_s, sp_s = eval_metrics(y_te, pred_s)
        print(f"  Spline  MAE={mae_s:.3f}  RMSE={rmse_s:.3f}  Spearman={sp_s:.3f}")

        # ── XGBoost ─────────────────────────────────────────────────────────
        pre_x, _, _ = build_tree_preprocessor(X_tr_raw)
        xgb_model = Pipeline([
            ("pre", pre_x),
            ("xgb", XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                                  subsample=0.8, colsample_bytree=0.8,
                                  reg_lambda=1.0, random_state=42)),
        ])
        xgb_model.fit(X_tr_raw, y_tr)
        pred_x = xgb_model.predict(X_te_raw)
        mae_x, rmse_x, sp_x = eval_metrics(y_te, pred_x)
        print(f"  XGBoost MAE={mae_x:.3f}  RMSE={rmse_x:.3f}  Spearman={sp_x:.3f}")

        # ── CatBoost ────────────────────────────────────────────────────────
        X_tr_cb, X_te_cb, cat_idx_cb, cb_feat_names = build_catboost_data(X_tr_raw, X_te_raw)
        cb_model = CatBoostRegressor(**CATBOOST_CFG, cat_features=cat_idx_cb)
        cb_model.fit(X_tr_cb, y_tr)
        pred_cb = cb_model.predict(X_te_cb)
        mae_cb, rmse_cb, sp_cb = eval_metrics(y_te, pred_cb)
        print(f"  CatBoost MAE={mae_cb:.3f}  RMSE={rmse_cb:.3f}  Spearman={sp_cb:.3f}")

        # ── Random Forest ────────────────────────────────────────────────────
        pre_rf, _num_rf, _cat_rf = build_tree_preprocessor(X_tr_raw)
        rf_model = Pipeline([
            ("pre", pre_rf),
            ("rf",  RandomForestRegressor(**RF_CFG)),
        ])
        rf_model.fit(X_tr_raw, y_tr)
        pred_rf = rf_model.predict(X_te_raw)
        mae_rf, rmse_rf, sp_rf = eval_metrics(y_te, pred_rf)
        print(f"  RF      MAE={mae_rf:.3f}  RMSE={rmse_rf:.3f}  Spearman={sp_rf:.3f}")

        # ── LightGBM ─────────────────────────────────────────────────────────
        pre_lgb, _num_lgb, _cat_lgb = build_tree_preprocessor(X_tr_raw)
        lgb_model = Pipeline([
            ("pre", pre_lgb),
            ("lgb", lgb.LGBMRegressor(**LGB_CFG)),
        ])
        lgb_model.fit(X_tr_raw, y_tr)
        pred_lgb = lgb_model.predict(X_te_raw)
        mae_lgb, rmse_lgb, sp_lgb = eval_metrics(y_te, pred_lgb)
        print(f"  LightGBM MAE={mae_lgb:.3f}  RMSE={rmse_lgb:.3f}  Spearman={sp_lgb:.3f}")

        # ── MLPE + FT-Transformer (shared embed preprocessing) ───────────────
        X_num_tr, X_num_te, X_cat_tr, X_cat_te, cardinalities = build_embed_data(
            X_tr_raw, X_te_raw)

        mlpe = MLPWithEmbeddings(X_num_tr.shape[1], cardinalities, **MLPE_CFG).to(DEVICE)
        mlpe = train_embed_model(mlpe, X_num_tr, X_cat_tr, y_tr, verbose=False)
        mlpe.eval()
        with torch.no_grad():
            pred_mlpe = mlpe(
                torch.tensor(X_num_te, dtype=torch.float32).to(DEVICE),
                torch.tensor(X_cat_te, dtype=torch.long).to(DEVICE),
            ).cpu().numpy()
        mae_mlpe, rmse_mlpe, sp_mlpe = eval_metrics(y_te, pred_mlpe)
        print(f"  MLPE    MAE={mae_mlpe:.3f}  RMSE={rmse_mlpe:.3f}  Spearman={sp_mlpe:.3f}")

        ftt = FTTransformer(X_num_tr.shape[1], cardinalities, **FTT_CFG).to(DEVICE)
        ftt = train_embed_model(ftt, X_num_tr, X_cat_tr, y_tr, verbose=False)
        ftt.eval()
        with torch.no_grad():
            pred_ftt = ftt(
                torch.tensor(X_num_te, dtype=torch.float32).to(DEVICE),
                torch.tensor(X_cat_te, dtype=torch.long).to(DEVICE),
            ).cpu().numpy()
        mae_ftt, rmse_ftt, sp_ftt = eval_metrics(y_te, pred_ftt)
        print(f"  FTT     MAE={mae_ftt:.3f}  RMSE={rmse_ftt:.3f}  Spearman={sp_ftt:.3f}")

        # ── TabNet ──────────────────────────────────────────────────────────
        pre_t, _num_t, _cat_t = build_tree_preprocessor(X_tr_raw)
        X_tr_t = pre_t.fit_transform(X_tr_raw).astype(np.float32)
        X_te_t = pre_t.transform(X_te_raw).astype(np.float32)
        feat_names_t = get_feature_names(pre_t, _num_t, _cat_t)

        tabnet, train_hist, val_hist = train_tabnet(X_tr_t, y_tr, X_tr_t.shape[1], verbose=True)
        tabnet.eval()
        with torch.no_grad():
            Xte_tensor = torch.tensor(X_te_t, dtype=torch.float32).to(DEVICE)
            pred_raw, _, masks = tabnet(Xte_tensor)
            pred_t = pred_raw.cpu().numpy()
            importance = tabnet.encoder.aggregate_importance(masks).cpu().numpy()
        mae_t, rmse_t, sp_t = eval_metrics(y_te, pred_t)
        print(f"  TabNet  MAE={mae_t:.3f}  RMSE={rmse_t:.3f}  Spearman={sp_t:.3f}")

        results.append({
            "test_year": int(test_year), "n_test": int(te_mask.sum()),
            "spline_mae":   mae_s,    "spline_rmse":   rmse_s,    "spline_spearman":   sp_s,
            "xgb_mae":      mae_x,    "xgb_rmse":      rmse_x,    "xgb_spearman":      sp_x,
            "catboost_mae": mae_cb,   "catboost_rmse": rmse_cb,   "catboost_spearman": sp_cb,
            "rf_mae":       mae_rf,   "rf_rmse":       rmse_rf,   "rf_spearman":       sp_rf,
            "lgb_mae":      mae_lgb,  "lgb_rmse":      rmse_lgb,  "lgb_spearman":      sp_lgb,
            "mlpe_mae":     mae_mlpe, "mlpe_rmse":     rmse_mlpe, "mlpe_spearman":     sp_mlpe,
            "ftt_mae":      mae_ftt,  "ftt_rmse":      rmse_ftt,  "ftt_spearman":      sp_ftt,
            "tabnet_mae":   mae_t,    "tabnet_rmse":   rmse_t,    "tabnet_spearman":   sp_t,
        })

        if test_year == latest_year:
            for y_pred, name in [
                (pred_s,    "Spline Ridge"),
                (pred_x,    "XGBoost"),
                (pred_cb,   "CatBoost"),
                (pred_rf,   "Random Forest"),
                (pred_lgb,  "LightGBM"),
                (pred_mlpe, "MLPE"),
                (pred_ftt,  "FTTransformer"),
                (pred_t,    "TabNet"),
            ]:
                slug = name.lower().replace(" ", "_")
                save_pred_vs_actual(y_te, y_pred,
                    os.path.join(OUT_DIR, f"plot_{slug}_pred_vs_actual_{test_year}.png"),
                    test_year, name)
            save_feature_importance(importance, feat_names_t,
                os.path.join(OUT_DIR, "plot_tabnet_feature_importance.png"),
                title=f"TabNet Feature Importance — trained through {test_year-1}")
            save_loss_curves(train_hist, val_hist,
                os.path.join(OUT_DIR, f"plot_tabnet_loss_curves_{test_year}.png"),
                test_year)
            xgb_imp = xgb_model.named_steps["xgb"].feature_importances_
            xgb_feat_names = get_feature_names(
                xgb_model.named_steps["pre"], _num_t, _cat_t)
            save_feature_importance(xgb_imp, xgb_feat_names,
                os.path.join(OUT_DIR, "plot_xgb_feature_importance.png"),
                title=f"XGBoost Feature Importance — trained through {test_year-1}")
            save_feature_importance(
                cb_model.get_feature_importance(), cb_feat_names,
                os.path.join(OUT_DIR, "plot_catboost_feature_importance.png"),
                title=f"CatBoost Feature Importance — trained through {test_year-1}")
            rf_feat_names = get_feature_names(rf_model.named_steps["pre"], _num_rf, _cat_rf)
            save_feature_importance(
                rf_model.named_steps["rf"].feature_importances_, rf_feat_names,
                os.path.join(OUT_DIR, "plot_rf_feature_importance.png"),
                title=f"Random Forest Feature Importance — trained through {test_year-1}")
            lgb_feat_names = get_feature_names(lgb_model.named_steps["pre"], _num_lgb, _cat_lgb)
            save_feature_importance(
                lgb_model.named_steps["lgb"].feature_importances_, lgb_feat_names,
                os.path.join(OUT_DIR, "plot_lgb_feature_importance.png"),
                title=f"LightGBM Feature Importance — trained through {test_year-1}")

    res_df = pd.DataFrame(results)
    print("\nWalk-forward results:")
    print(res_df.to_string(index=False))
    out_csv = os.path.join(OUT_DIR, "walkforward_results_v3.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nResults: {out_csv}")
    print(f"Plots:   {OUT_DIR}")


if __name__ == "__main__":
    main()
