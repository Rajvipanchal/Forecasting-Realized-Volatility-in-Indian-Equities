"""Forecast evaluation: loss functions, tables and statistical tests.

Losses (lower is better), for forecast h and realized variance RV:
    MSE   = (RV - h)^2
    MAE   = |RV - h|
    QLIKE = RV/h - log(RV/h) - 1
QLIKE is the headline loss: it ranks volatility forecasts correctly even though RV is a
noisy proxy for true variance (Patton, 2011), and it is not dominated by a handful of
extreme days the way MSE is.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .config import TARGET


def losses(rv, h):
    rv = np.asarray(rv, float)
    h = np.asarray(h, float)
    ratio = rv / h
    return {"MSE": (rv - h) ** 2, "MAE": np.abs(rv - h), "QLIKE": ratio - np.log(ratio) - 1}


def metric_table(df, models, target=TARGET):
    rows = []
    for m in models:
        l = losses(df[target], df[m])
        rows.append({"Model": m, "Obs": len(df), **{k: v.mean() for k, v in l.items()}})
    return pd.DataFrame(rows).set_index("Model")


def grouped_metrics(df, models, by, target=TARGET):
    out = []
    for g, d in df.groupby(by, observed=True):
        t = metric_table(d, models, target).reset_index()
        t.insert(0, by, g)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def _daily_loss_diff(df, a, b, metric, target=TARGET):
    d = losses(df[target], df[a])[metric] - losses(df[target], df[b])[metric]
    return pd.Series(d, index=df["Date_t1"].to_numpy()).groupby(level=0).mean().sort_index()


def dm_test(df, a, b, metric="QLIKE", lags=5, target=TARGET):
    """Panel Diebold-Mariano test.

    Loss differences are averaged across stocks for each day, then the mean of that
    daily series is tested with Newey-West (HAC) standard errors. A positive mean
    difference means model b has the lower loss.
    """
    s = _daily_loss_diff(df, a, b, metric, target)
    fit = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"Metric": metric, "Model A": a, "Model B": b, "Mean diff (A-B)": s.mean(),
            "t-stat": fit.tvalues[0], "p-value": fit.pvalues[0], "Days": len(s)}


def dm_table(df, pairs, metrics=("QLIKE", "MSE", "MAE"), target=TARGET):
    t = pd.DataFrame([dm_test(df, a, b, m, target=target) for m in metrics for a, b in pairs])
    t["Holm p"] = np.nan
    for m in metrics:
        k = t["Metric"] == m
        t.loc[k, "Holm p"] = multipletests(t.loc[k, "p-value"], method="holm")[1]
    return t


def block_bootstrap(df, a, b, metric="QLIKE", block=5, n_boot=2000, seed=42, target=TARGET):
    """Moving-block bootstrap (blocks of trading days) for the mean daily loss difference."""
    v = _daily_loss_diff(df, a, b, metric, target).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(v)
    k = int(np.ceil(n / block))
    est = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=k)
        est[i] = np.concatenate([v[s:s + block] for s in starts])[:n].mean()
    return {"Model A": a, "Model B": b, "Mean diff (A-B)": v.mean(),
            "CI 2.5%": np.quantile(est, 0.025), "CI 97.5%": np.quantile(est, 0.975)}


def mincer_zarnowitz(df, models, target=TARGET, log=True):
    """Regress realized on forecast: RV = a + b*h. Unbiased & efficient forecast: a=0, b=1.
    Run in logs by default so a few extreme days do not dominate."""
    rows = []
    y = np.log(df[target]) if log else df[target]
    for m in models:
        x = np.log(df[m]) if log else df[m]
        fit = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(df["Date_t1"])[0]})
        rows.append({"Model": m, "alpha": fit.params[0], "beta": fit.params[1],
                     "se(beta)": fit.bse[1], "R2": fit.rsquared})
    return pd.DataFrame(rows).set_index("Model")


def stock_win_rates(df, models, benchmark, metric="QLIKE", target=TARGET):
    """Share of stocks on which each model has a lower average loss than the benchmark."""
    per = {m: df.groupby("Symbol").apply(lambda d: losses(d[target], d[m])[metric].mean())
           for m in models + [benchmark]}
    per = pd.DataFrame(per)
    return pd.Series({m: (per[m] < per[benchmark]).mean() for m in models}, name=f"share beating {benchmark}"), per
