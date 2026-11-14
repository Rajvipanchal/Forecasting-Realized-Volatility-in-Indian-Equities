"""Clean the 5-minute bars and build the daily stock panel.

Output: one row per stock and trading day with
    RV          realized variance = sum of squared 5-minute log returns (09:15-15:30)
    n_returns   number of intraday returns used
    ret_oc      open-to-close log return (sum of intraday returns)
    ret_on      overnight log return (previous close -> today's open)
    ret_cc      close-to-close log return in %, used by EGARCH
    ca_flag     True if the overnight gap looks like an unadjusted split / bonus
    rupee_vol   traded value (sum of close x volume)
    amihud_sum, amihud_n   pieces of the 5-minute Amihud illiquidity ratio

Overnight returns are NOT part of RV. This keeps unadjusted splits and bonus issues,
which happen overnight, out of the volatility target.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C


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


def clean_intraday(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read one <SYMBOL>_5minute.csv file and return clean bars plus a cleaning log."""
    symbol = path.stem.replace("_5minute", "").upper()
    df = pd.read_csv(path, usecols=["date", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    log = {"Symbol": symbol, "raw_bars": len(df)}

    # 1. regular session only (drops special sessions such as Diwali muhurat trading)
    t = df["date"].dt.strftime("%H:%M")
    df = df[(t >= C.SESSION_FIRST_BAR) & (t <= C.SESSION_LAST_BAR)]
    df = df[df["date"] >= C.HISTORY_START]
    if symbol in C.LISTING_DATES:
        df = df[df["date"] >= C.LISTING_DATES[symbol]]
    log["session_bars"] = len(df)

    # 2. non-positive prices and zero-volume bars carry no valid price
    bad_price = (df["close"] <= 0) | (df["open"] <= 0) | df["close"].isna()
    no_trade = df["volume"] <= 0
    log["bad_price_bars"] = int(bad_price.sum())
    log["zero_volume_bars"] = int(no_trade.sum())
    df = df[~bad_price].copy()
    df["day"] = df["date"].dt.normalize()
    df["log_close"] = np.log(df["close"]).where(~no_trade[~bad_price])

    # 3. isolated spikes
    before = df["log_close"].notna().sum()
    df["log_close"] = _remove_bad_ticks(df["log_close"], df["day"])
    log["bad_ticks_removed"] = int(before - df["log_close"].notna().sum())

    # 4. days with too few traded bars (half days, pre-listing quotes, outages)
    valid_bars = df.groupby("day")["log_close"].transform("count")
    short_days = df.loc[valid_bars < C.MIN_VALID_BARS, "day"].nunique()
    df = df[valid_bars >= C.MIN_VALID_BARS].copy()
    log["short_days_dropped"] = int(short_days)
    df["Symbol"] = symbol
    return df, log


def daily_from_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clean bars to one row per day."""
    df = df.copy()
    lc = df["log_close"]
    # open of the first bar that actually traded (skips zero-volume pre-open quotes)
    df["open_traded"] = df["open"].where(lc.notna())
    first_open = np.log(df.groupby("day")["open_traded"].transform("first"))
    # previous valid close within the same day; the first traded bar is measured from the open
    prev = lc.groupby(df["day"]).ffill().groupby(df["day"]).shift(1)
    prev = prev.fillna(first_open)
    df["r"] = (lc - prev).where(lc.notna())
    df["rupee"] = df["close"] * df["volume"]
    traded = df["r"].notna() & (df["volume"] > 0)
    df["amihud"] = (df["r"].abs() / df["rupee"]).where(traded)
    df["r2"] = df["r"] ** 2

    g = df.groupby("day")
    daily = pd.DataFrame({
        "RV": g["r2"].sum(),
        "n_returns": g["r"].count(),
        "ret_oc": g["r"].sum(),
        "open": g["open_traded"].first(),
        "close": g["log_close"].last().pipe(np.exp),
        "rupee_vol": g["rupee"].sum(),
        "amihud_sum": g["amihud"].sum(),
        "amihud_n": g["amihud"].count(),
    })
    last_close = np.log(daily["close"]).shift(1)
    daily["ret_on"] = np.log(daily["open"]) - last_close
    daily["ca_flag"] = daily["ret_on"].abs() > C.CORPORATE_ACTION_GAP
    daily.loc[daily["ca_flag"], "ret_on"] = 0.0          # remove split / bonus gaps
    daily["ret_cc"] = 100 * (daily["ret_on"].fillna(0) + daily["ret_oc"])
    daily["Symbol"] = df["Symbol"].iloc[0]
    daily = daily.reset_index().rename(columns={"day": "Date"})
    return daily.drop(columns=["open", "close"])


def build_daily_panel(raw_dir: Path = C.RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean every CSV in raw_dir and stack the daily rows. Returns (panel, cleaning_log)."""
    files = sorted(Path(raw_dir).glob("*_5minute.csv"))
    if not files:
        raise FileNotFoundError(f"No *_5minute.csv files in {raw_dir}. Set VOL_DATA_DIR.")
    panels, logs = [], []
    for f in files:
        bars, log = clean_intraday(f)
        if bars.empty:
            logs.append(log)
            continue
        day = daily_from_bars(bars)
        log["days"] = len(day)
        log["corporate_action_days"] = int(day["ca_flag"].sum())
        panels.append(day)
        logs.append(log)
    panel = pd.concat(panels, ignore_index=True).sort_values(["Symbol", "Date"])
    logs = pd.DataFrame(logs)
    # Days with RV = 0 are stocks frozen at a price band all day (e.g. a 5% lower
    # circuit). Their RV is not a volatility measurement, so they are dropped.
    zero = panel["RV"] <= 0
    logs = logs.merge(panel[zero].groupby("Symbol").size().rename("zero_rv_days_dropped"),
                      left_on="Symbol", right_index=True, how="left").fillna({"zero_rv_days_dropped": 0})
    panel = panel[~zero]
    return panel.reset_index(drop=True), logs


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


def add_har_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily, weekly and monthly RV averages and the next-day target, per stock."""
    p = panel.sort_values(["Symbol", "Date"]).copy()
    g = p.groupby("Symbol")["RV"]
    p["RV_5"] = g.transform(lambda x: x.rolling(5).mean())
    p["RV_22"] = g.transform(lambda x: x.rolling(22).mean())
    p["RV_t1"] = g.shift(-1)
    p["Date_t1"] = p.groupby("Symbol")["Date"].shift(-1)
    return p


def split_by_target_date(p: pd.DataFrame):
    """Assign each row to a split by the date of its TARGET (t+1).

    This prevents the last training row from using a validation-day outcome.
    """
    d = p["Date_t1"]
    train = p[(d >= C.TRAIN_START) & (d <= C.TRAIN_END)].copy()
    valid = p[(d >= C.VALID_START) & (d <= C.VALID_END)].copy()
    test = p[(d >= C.TEST_START) & (d <= C.TEST_END)].copy()
    return train, valid, test
