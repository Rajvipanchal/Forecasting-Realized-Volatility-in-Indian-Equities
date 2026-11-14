def liquidity_groups(panel: pd.DataFrame, start=C.TRAIN_START, end=C.TRAIN_END) -> pd.DataFrame:
    """Rank stocks by 5-minute Amihud illiquidity measured in the TRAINING window only.

    ILLIQ = average over traded 5-minute bars of |return| / rupee volume  (Amihud, 2002).
    Lower ILLIQ = more liquid. Terciles: High / Medium / Low liquidity.
    Using training data only means the grouping contains no information from the
    validation or test years.
    """
    w = panel[(panel["Date"] >= start) & (panel["Date"] <= end)]
    agg = w.groupby("Symbol").agg(
        amihud_sum=("amihud_sum", "sum"), amihud_n=("amihud_n", "sum"),
        avg_daily_turnover_cr=("rupee_vol", lambda x: x.mean() / 1e7),
        train_days=("RV", "size"),
    )
    agg["ILLIQ"] = agg["amihud_sum"] / agg["amihud_n"] * 1e9   # per Rs 100 crore traded
    agg = agg.drop(columns=["amihud_sum", "amihud_n"]).sort_values("ILLIQ")
    agg["Liquidity_Rank"] = np.arange(1, len(agg) + 1)
    agg["Liquidity_Group"] = pd.qcut(
        agg["ILLIQ"], 3, labels=["High Liquidity", "Medium Liquidity", "Low Liquidity"]
    ).astype(str)
    return agg.reset_index()
