"""EGARCH(1,1)-t features without look-ahead.

For each stock the EGARCH parameters are estimated on TRAINING-period daily returns
only. The fitted parameters are then held fixed and the model is run forward through
the validation and test years, so the conditional volatility on any day uses only
returns observed up to that day and parameters that never saw the evaluation data.

Features produced for day t (both known at the close of day t):
    EGARCH_Vol    conditional volatility sigma_t (%, daily)
    EGARCH_Resid  standardized shock z_t = (r_t - mu) / sigma_t
"""
import warnings
import numpy as np
import pandas as pd
from arch import arch_model

from . import config as C


def egarch_features(panel: pd.DataFrame, symbols, fit_end: str = C.TRAIN_END):
    feats, params = [], []
    for sym in symbols:
        s = panel[panel["Symbol"] == sym].sort_values("Date")
        r = s.set_index("Date")["ret_cc"].astype(float)
        r_fit = r[r.index <= fit_end]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(r_fit, mean="Constant", vol="EGARCH", p=1, o=1, q=1,
                             dist="t").fit(disp="off", show_warning=False)
            fixed = arch_model(r, mean="Constant", vol="EGARCH", p=1, o=1, q=1,
                               dist="t").fix(res.params)
        feats.append(pd.DataFrame({
            "Symbol": sym, "Date": r.index,
            "EGARCH_Vol": fixed.conditional_volatility.to_numpy(),
            "EGARCH_Resid": fixed.std_resid.to_numpy(),
        }))
        params.append({"Symbol": sym, **res.params.to_dict(), "n_fit": len(r_fit),
                       "converged": res.convergence_flag == 0})
    return pd.concat(feats, ignore_index=True), pd.DataFrame(params)
