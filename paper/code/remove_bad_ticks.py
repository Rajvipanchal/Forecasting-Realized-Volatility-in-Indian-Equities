def _remove_bad_ticks(log_close: pd.Series, day: pd.Series) -> pd.Series:
    """Blank out isolated price spikes that are reversed in the next bar.

    A spike is a 5-minute move larger than BAD_TICK_MIN_MOVE that the next bar undoes
    almost completely (e.g. VBL on 14-Jun-2023: 322 -> 644 -> 322). Genuine large moves
    (e.g. Adani stocks in Feb 2023) are not reversed and are kept.
    """
    lc = log_close.copy()
    for _ in range(3):  # repeat: removing one spike can reveal another
        prev = lc.groupby(day).ffill().groupby(day).shift(1)
        nxt = lc.groupby(day).bfill().groupby(day).shift(-1)
        r_in = lc - prev
        r_out = nxt - lc
        spike = (
            (r_in.abs() > C.BAD_TICK_MIN_MOVE)
            & (np.sign(r_in) != np.sign(r_out))
            & ((r_in + r_out).abs() < C.BAD_TICK_REVERSAL * r_in.abs())
        )
        if not spike.any():
            break
        lc[spike] = np.nan
    return lc
