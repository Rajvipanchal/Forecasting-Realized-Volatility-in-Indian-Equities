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
