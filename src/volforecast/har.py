"""HAR-RV benchmarks (Corsi, 2009), one regression per stock.

HAR-RV   : RV_{t+1}     = b0 + bD RV_t     + bW RV5_t     + bM RV22_t     + e   (levels, OLS)
log-HAR  : log RV_{t+1} = b0 + bD log RV_t + bW log RV5_t + bM log RV22_t + e   (logs, OLS)
           forecast = exp(fitted + s^2 / 2)   (log-normal bias correction)

Validation forecasts use coefficients estimated on the training years.
Test forecasts use coefficients re-estimated on training + validation years.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import HAR_FEATURES, TARGET

FLOOR = 1e-8  # a level-HAR forecast can be negative; it is floored so every loss is defined


def _fit_predict(train: pd.DataFrame, apply: pd.DataFrame, log: bool):
    X = train[HAR_FEATURES].to_numpy()
    y = train[TARGET].to_numpy()
    Xa = apply[HAR_FEATURES].to_numpy()
    if log:
        X, y, Xa = np.log(X), np.log(y), np.log(Xa)
    fit = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
    pred = fit.predict(sm.add_constant(Xa, has_constant="add"))
    if log:
        pred = np.exp(pred + fit.mse_resid / 2)
    params = dict(zip(["const", "beta_D", "beta_W", "beta_M"], fit.params))
    return pred, params


def har_forecasts(train: pd.DataFrame, apply: pd.DataFrame, log: bool = False,
                  name: str = "HAR") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit per stock on `train`, forecast every row of `apply`.

    Returns (forecast rows, coefficient table). Negative level forecasts are counted
    and floored rather than silently dropped.
    """
    out, coefs = [], []
    for sym, tr in train.groupby("Symbol"):
        ap = apply[apply["Symbol"] == sym]
        if ap.empty or len(tr) < 100:
            continue
        pred, params = _fit_predict(tr, ap, log)
        f = ap[["Symbol", "Date", "Date_t1", TARGET]].copy()
        f[name] = pred
        out.append(f)
        coefs.append({"Symbol": sym, **params, "n_train": len(tr)})
    fc = pd.concat(out, ignore_index=True)
    fc[f"{name}_was_negative"] = fc[name] <= 0
    fc[name] = fc[name].clip(lower=FLOOR)
    return fc, pd.DataFrame(coefs)
