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
