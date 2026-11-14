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
