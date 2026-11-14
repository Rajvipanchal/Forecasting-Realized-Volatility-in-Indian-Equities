"""Pooled LSTM volatility models (HAR-RV-LSTM and EGARCH-LSTM).

One network is trained on all stocks together. Each input is the last LOOKBACK (22)
trading days of a feature set for one stock; a learned stock embedding lets the
network adjust the level of volatility for each name. The network outputs a
next-day realized-variance forecast and is trained with the QLIKE loss.

    HAR-RV-LSTM   inputs per day: log RV, log RV_5, log RV_22
    EGARCH-LSTM   inputs per day: log EGARCH volatility, EGARCH standardized shock
    HAR-EGARCH-LSTM  both sets together (ablation: does EGARCH add anything to HAR?)
"""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config as C

FEATURE_SETS = {
    "HAR-RV-LSTM": ["RV", "RV_5", "RV_22"],
    "EGARCH-LSTM": ["EGARCH_Vol", "EGARCH_Resid"],
    "HAR-EGARCH-LSTM": ["RV", "RV_5", "RV_22", "EGARCH_Vol", "EGARCH_Resid"],   # ablation
}
LOG_FEATURES = {"RV", "RV_5", "RV_22", "EGARCH_Vol"}


# ----------------------------------------------------------------------------- model
class VolatilityLSTM(nn.Module):
    """2-layer LSTM (64 units) + stock embedding (8) + dense head (64-32-1).

    The output is exp(z + c), where c is the average log RV of the training targets.
    The exponential keeps every forecast positive, which QLIKE requires.
    """

    def __init__(self, n_features: int, n_stocks: int, log_rv_mean: float,
                 hidden: int = 64, layers: int = 2, emb_dim: int = 8, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(n_stocks, emb_dim)
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + emb_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.register_buffer("c", torch.tensor(float(log_rv_mean)))

    def forward(self, x, stock_id):
        _, (h, _) = self.lstm(x)
        z = self.head(torch.cat([h[-1], self.embedding(stock_id)], dim=1)).squeeze(-1)
        return torch.exp(z.clamp(-15, 15) + self.c)


class QLIKELoss(nn.Module):
    """QLIKE = RV/h - log(RV/h) - 1 (Patton, 2011). Zero only when h = RV."""

    def forward(self, h, rv):
        ratio = rv / h
        return (ratio - torch.log(ratio) - 1).mean()


# ------------------------------------------------------------------------ sequences
class Scaler:
    """Log-transform (where listed) and standardize, fitted on training rows only."""

    def __init__(self, features):
        self.features = features

    def _raw(self, df):
        X = df[self.features].to_numpy(dtype=np.float64).copy()
        for j, f in enumerate(self.features):
            if f in LOG_FEATURES:
                X[:, j] = np.log(X[:, j])
            elif f == "EGARCH_Resid":
                X[:, j] = np.clip(X[:, j], -10, 10)
        return X

    def fit(self, df):
        X = self._raw(df)
        self.mean_, self.std_ = X.mean(0), X.std(0)
        return self

    def transform(self, df):
        return ((self._raw(df) - self.mean_) / self.std_).astype(np.float32)


def build_sequences(data: pd.DataFrame, features, scaler: Scaler, stock_index: dict,
                    lookback: int = C.LOOKBACK):
    """Sliding windows over each stock's full history.

    Every window ends on day t and only contains days <= t, so it can be used for any
    split; the split is decided afterwards by the date of the target (t + 1).
    """
    X, meta = [], []
    for sym, s in data.groupby("Symbol"):
        s = s.dropna(subset=features).sort_values("Date")
        if len(s) < lookback:
            continue
        Z = scaler.transform(s)
        win = np.lib.stride_tricks.sliding_window_view(Z, (lookback, Z.shape[1]))[:, 0]
        X.append(win)
        m = s.iloc[lookback - 1:][["Symbol", "Date", "Date_t1", C.TARGET]].copy()
        m["stock_id"] = stock_index[sym]
        meta.append(m)
    return np.concatenate(X).astype(np.float32), pd.concat(meta, ignore_index=True)


def _loader(X, meta, idx, shuffle):
    ds = TensorDataset(torch.from_numpy(X[idx]),
                       torch.from_numpy(meta[C.TARGET].to_numpy(np.float32)[idx]),
                       torch.from_numpy(meta["stock_id"].to_numpy(np.int64)[idx]))
    return DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def predict(model, X, meta, idx):
    model.eval()
    out = [model(xb, sb).numpy() for xb, _, sb in _loader(X, meta, idx, False)]
    return np.concatenate(out)


def train_one_seed(X, meta, train_idx, valid_idx, n_stocks, seed, verbose=True):
    """Train with early stopping on validation QLIKE; return model and history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    c = float(np.log(meta[C.TARGET].to_numpy()[train_idx]).mean())
    model = VolatilityLSTM(X.shape[2], n_stocks, c)
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=1e-6)
    loss_fn = QLIKELoss()
    train_dl = _loader(X, meta, train_idx, True)
    valid_dl = _loader(X, meta, valid_idx, False)

    best, best_state, wait, hist = np.inf, None, 0, []
    for epoch in range(1, C.MAX_EPOCHS + 1):
        model.train()
        tr = 0.0
        for xb, yb, sb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb, sb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * len(yb)
        tr /= len(train_idx)
        model.eval()
        with torch.no_grad():
            va = sum(loss_fn(model(xb, sb), yb).item() * len(yb) for xb, yb, sb in valid_dl)
        va /= len(valid_idx)
        sched.step(va)
        hist.append({"Seed": seed, "Epoch": epoch, "Train_QLIKE": tr, "Validation_QLIKE": va,
                     "Learning_Rate": opt.param_groups[0]["lr"]})
        if verbose:
            print(f"seed {seed} | epoch {epoch:2d} | train {tr:.4f} | valid {va:.4f}", flush=True)
        if va < best - 1e-5:
            best, best_state, wait = va, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= C.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)


def run_lstm(data: pd.DataFrame, name: str, symbols, seeds=C.SEEDS, verbose=True):
    """Train one model per seed; return validation/test forecasts (per seed and the
    seed-average ensemble), training history and the trained models."""
    features = FEATURE_SETS[name]
    data = data[data["Symbol"].isin(symbols)]
    stock_index = {s: i for i, s in enumerate(sorted(symbols))}
    d1 = data["Date_t1"]
    train_rows = data[(d1 >= C.TRAIN_START) & (d1 <= C.TRAIN_END)].dropna(subset=features)
    scaler = Scaler(features).fit(train_rows)
    X, meta = build_sequences(data, features, scaler, stock_index)
    t1 = meta["Date_t1"]
    ok = meta[C.TARGET].notna()
    idx = {
        "train": np.where(ok & (t1 >= C.TRAIN_START) & (t1 <= C.TRAIN_END))[0],
        "valid": np.where(ok & (t1 >= C.VALID_START) & (t1 <= C.VALID_END))[0],
        "test": np.where(ok & (t1 >= C.TEST_START) & (t1 <= C.TEST_END))[0],
    }
    preds = {"valid": [], "test": []}
    histories, models = [], {}
    for seed in seeds:
        model, hist = train_one_seed(X, meta, idx["train"], idx["valid"], len(stock_index),
                                     seed, verbose)
        histories.append(hist)
        models[seed] = model
        for split in ("valid", "test"):
            preds[split].append(predict(model, X, meta, idx[split]))

    out = {}
    for split in ("valid", "test"):
        f = meta.iloc[idx[split]][["Symbol", "Date", "Date_t1", C.TARGET]].reset_index(drop=True)
        for seed, p in zip(seeds, preds[split]):
            f[f"{name}_seed{seed}"] = p
        f[name] = np.mean(preds[split], axis=0)   # seed-average ensemble
        out[split] = f
    info = {"n_train": len(idx["train"]), "n_valid": len(idx["valid"]), "n_test": len(idx["test"]),
            "scaler_mean": scaler.mean_.tolist(), "scaler_std": scaler.std_.tolist()}
    return out, pd.concat(histories, ignore_index=True), models, info
