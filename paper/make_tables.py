"""Write every LaTeX table in the paper from the CSV files in results/tables.

Run after notebook 06:  python make_tables.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)
MAIN = ["HAR-RV", "log-HAR", "HAR-RV-LSTM", "EGARCH-LSTM"]
ALL = MAIN + ["HAR-EGARCH-LSTM"]
GROUPS = ["High Liquidity", "Medium Liquidity", "Low Liquidity"]


ANNEX = {"topdays", "seeds", "winrates", "harcoef", "egarch", "perstock"}


def write(name, caption, label, header, rows, colspec, note=None, size="\\small"):
    lines = ["\\begin{table}[H]" if name in ANNEX else "\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
             "\\begin{adjustbox}{max width=\\textwidth}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    import re
    for r in rows:
        r = re.sub(r"(?<=[\s&(\[])-(?=\d)", "$-$", r)
        lines.append(r if r.startswith("\\") else r + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if note:
        lines.append(f"\\par\\smallskip\\parbox{{0.95\\textwidth}}{{\\footnotesize\\itshape {note}}}")
    lines.append("\\end{table}")
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:,.{d}f}".replace("-", "$-$")


def pct(x, d=1):
    return "--" if pd.isna(x) else (f"{x:+.{d}f}").replace("-", "$-$")


def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"{m}$\\times10^{{{int(e)}}}$"


def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def bold_min(vals, fmt):
    best = np.nanmin(vals)
    return [f"\\textbf{{{fmt(v)}}}" if v == best else fmt(v) for v in vals]


# ------------------------------------------------------------------ data tables
log = pd.read_csv(T / "data_cleaning_log.csv")
tot = log.sum(numeric_only=True)
rows = [
    f"Raw 5-minute bars, 100 stocks & -- & {tot.raw_bars:,.0f}",
    f"Regular session 09:15--15:30, from Nov 2015, after listing date & Keep & {tot.session_bars:,.0f}",
    f"Zero-volume bars (stale quote, price set to missing) & Drop price & {tot.zero_volume_bars:,.0f}",
    f"Isolated spikes ($>$5\\% move undone by next bar) & Drop price & {tot.bad_ticks_removed:,.0f}",
    f"Days with fewer than 60 traded bars & Drop day & {tot.short_days_dropped:,.0f}",
    f"Days with RV $=0$ (stock frozen at price band) & Drop day & {tot.zero_rv_days_dropped:,.0f}",
    f"Overnight gaps $>35\\%$ (split/bonus) in EGARCH returns & Set to 0 & {tot.corporate_action_days:,.0f}",
    f"Stock-days in the clean panel & -- & {tot.days - tot.zero_rv_days_dropped:,.0f}",
]
write("cleaning", "Data cleaning rules and their effect", "tab:cleaning",
      "Rule & Action & Count", rows, "lcr",
      note="Pre-listing rows removed for IRFC, SBILIFE, HDFCLIFE and MAZDOCK. RV never uses overnight returns.")

d = pd.read_csv(T / "descriptive_by_liquidity.csv").set_index("Liquidity_Group").loc[GROUPS]
rows = [f"{g.replace(' Liquidity', '')} & {r.Stocks:.0f} & {r.Stock_days:,.0f} & {r.Avg_turnover_cr:,.0f} & "
        f"{r.Mean_ann_vol:.1f} & {r.Median_ann_vol:.1f} & {r.P99_ann_vol:.1f} & {r.AR1_logRV:.2f}"
        for g, r in d.iterrows()]
write("descriptive", "The three liquidity groups, 2016--2026", "tab:desc",
      "Liquidity & Stocks & Stock-days & Turnover (Rs cr/day) & Mean vol (\\%) & Median vol (\\%) & 99th pct vol (\\%) & AR(1) log RV",
      rows, "lrrrrrrr",
      note="Vol = annualized intraday realized volatility $\\sqrt{252\\,RV}$. Turnover measured in the training years.")

# ------------------------------------------------------------------ overall results
te = pd.read_csv(T / "overall_test_metrics.csv").set_index("Model").loc[ALL]
va = pd.read_csv(T / "overall_validation_metrics.csv").set_index("Model").loc[ALL]
qb = bold_min(te.QLIKE.loc[MAIN].values, lambda v: f"{v:.4f}") + [f"{te.QLIKE.iloc[-1]:.4f}"]
mb = bold_min(te.MAE.loc[MAIN].values * 1e4, lambda v: f"{v:.3f}") + [f"{te.MAE.iloc[-1] * 1e4:.3f}"]
sb = bold_min(te.MSE.loc[MAIN].values * 1e8, lambda v: f"{v:.3f}") + [f"{te.MSE.iloc[-1] * 1e8:.3f}"]
rows = []
for i, m in enumerate(ALL):
    if m == "HAR-EGARCH-LSTM":
        rows.append("\\midrule")
    rows.append(f"{m} & {va.loc[m, 'QLIKE']:.4f} & {qb[i]} & {mb[i]} & {sb[i]} & "
                f"{'--' if m == 'HAR-RV' else pct(te.loc[m, 'QLIKE vs HAR-RV (%)'])} & {'--' if m == 'log-HAR' else pct(te.loc[m, 'QLIKE vs log-HAR (%)'])}")
write("overall", "Forecast accuracy, validation and test periods", "tab:overall",
      "Model & Valid QLIKE & Test QLIKE & Test MAE ($\\times10^{4}$) & Test MSE ($\\times10^{8}$) & vs HAR-RV (\\%) & vs log-HAR (\\%)",
      rows, "lrrrrrr",
      note=f"Test: {int(te.Obs.iloc[0]):,} stock-days, 95 stocks, 1 Jan 2025 -- 9 Apr 2026; validation: "
           f"{int(va.Obs.iloc[0]):,} stock-days, 2023--2024. Last two columns: change in test QLIKE (negative = better). "
           "Bold = lowest test loss among the four main models. HAR-EGARCH-LSTM is an ablation.")

s = pd.read_csv(T / "lstm_seed_stability.csv").set_index("Model")
rows = [f"{m} & {r['Valid QLIKE mean']:.4f} & {r['Valid QLIKE sd']:.4f} & {r['Test QLIKE mean']:.4f} & "
        f"{r['Test QLIKE sd']:.4f} & {r['Test min']:.4f}--{r['Test max']:.4f} & {r['Ensemble test QLIKE']:.4f}"
        for m, r in s.iterrows()]
write("seeds", "LSTM results across three random seeds", "tab:seeds",
      "Model & Valid mean & Valid s.d. & Test mean & Test s.d. & Test range & 3-seed average forecast",
      rows, "lrrrrcr", note="QLIKE of each seed's network and of the forecast averaged over the three networks.")

dm = pd.read_csv(T / "dm_tests_test.csv")
boot = pd.read_csv(T / "block_bootstrap_qlike.csv")
rows = []
for _, r in dm[dm.Metric == "QLIKE"].iterrows():
    b = boot[(boot["Model A"] == r["Model A"]) & (boot["Model B"] == r["Model B"])].iloc[0]
    mae = dm[(dm.Metric == "MAE") & (dm["Model A"] == r["Model A"]) & (dm["Model B"] == r["Model B"])].iloc[0]
    rows.append(f"{r['Model A']} vs {r['Model B']} & {r['Mean diff (A-B)']:.4f} & {r['t-stat']:.2f} & {pv(r['Holm p'])} & "
                f"[{b['CI 2.5%']:.4f}, {b['CI 97.5%']:.4f}] & {mae['t-stat']:.2f} & {pv(mae['Holm p'])}")
write("dm", "Diebold--Mariano tests on the test period (A vs B)", "tab:dm",
      "Comparison & QLIKE diff (A$-$B) & $t$ & Holm $p$ & 95\\% block-bootstrap CI & MAE $t$ & MAE Holm $p$",
      rows, "lrrrcrr",
      note="Positive difference / $t$ = model B has lower loss. Loss differences averaged across stocks per day "
           f"({int(dm.Days.iloc[0])} days), Newey--West (5 lags) standard errors; Holm adjustment over the seven comparisons per loss. "
           "Bootstrap: 5-day blocks, 2,000 draws.")

# ------------------------------------------------------------------ liquidity
lt = pd.read_csv(T / "liquidity_group_metrics.csv")
q = lt.pivot(index="Liquidity_Group", columns="Model", values="QLIKE").loc[GROUPS]
n = lt.groupby("Liquidity_Group").Obs.first().loc[GROUPS]
dmg = pd.read_csv(T / "liquidity_group_dm.csv")
rows = []
for g in GROUPS:
    v = bold_min(q.loc[g, MAIN].values, lambda x: f"{x:.4f}")
    t1 = dmg[(dmg.Liquidity_Group == g) & (dmg.Benchmark == "log-HAR")].iloc[0]
    gain = 100 * (1 - q.loc[g, "HAR-RV-LSTM"] / q.loc[g, "log-HAR"])
    gain_h = 100 * (1 - q.loc[g, "HAR-RV-LSTM"] / q.loc[g, "HAR-RV"])
    rows.append(f"{g.replace(' Liquidity', '')} & {n[g]:,} & " + " & ".join(v) +
                f" & {gain_h:.1f} & {gain:.1f} & {t1['t-stat']:.2f}")
write("liquidity", "Test QLIKE by liquidity group", "tab:liquidity",
      "Liquidity & Obs & HAR-RV & log-HAR & HAR-RV-LSTM & EGARCH-LSTM & Gain vs HAR (\\%) & Gain vs log-HAR (\\%) & DM $t$",
      rows, "lrrrrrrrr",
      note="Gain = percentage reduction in QLIKE of HAR-RV-LSTM. DM $t$: HAR-RV-LSTM against log-HAR within the group "
           "(positive = LSTM better). Bold = best model in the group.")

# ------------------------------------------------------------------ regimes, subperiods, robustness
ra = pd.read_csv(T / "regime_ex_ante_qlike.csv").set_index("Regime_ex_ante")
rp = pd.read_csv(T / "regime_ex_post_qlike.csv").set_index("Regime_ex_post")
rows = ["\\multicolumn{6}{l}{\\emph{Ex ante: market average RV$_{22}$ on day $t$, training-period terciles}}\\\\"]
for k, r in ra.iterrows():
    v = bold_min(r[MAIN].values.astype(float), lambda x: f"{x:.4f}")
    rows.append(f"{k} ({int(r.Days)} days) & " + " & ".join(v) + f" & {100 * (1 - r['HAR-RV-LSTM'] / r['log-HAR']):.1f}")
rows.append("\\midrule")
rows.append("\\multicolumn{6}{l}{\\emph{Ex post: realized RV of the target day, test terciles (descriptive)}}\\\\")
for k, r in rp.loc[["Low RV", "Medium RV", "High RV"]].iterrows():
    v = bold_min(r[MAIN].values.astype(float), lambda x: f"{x:.4f}")
    rows.append(f"{k} & " + " & ".join(v) + f" & {100 * (1 - r['HAR-RV-LSTM'] / r['log-HAR']):.1f}")
write("regimes", "Test QLIKE by market regime", "tab:regimes",
      "Regime & HAR-RV & log-HAR & HAR-RV-LSTM & EGARCH-LSTM & LSTM gain vs log-HAR (\\%)", rows, "lrrrrr")

sp = pd.read_csv(T / "subperiod_qlike.csv").set_index("Subperiod")
rows = []
for k, r in sp.iterrows():
    v = bold_min(r[MAIN].values.astype(float), lambda x: f"{x:.4f}")
    rows.append(f"{r.From} -- {r.To} & " + " & ".join(v))
tr = pd.read_csv(T / "trimmed_top1pct_metrics.csv").set_index("Model").loc[MAIN]
rows.append("\\midrule")
v = bold_min(tr.QLIKE.values, lambda x: f"{x:.4f}")
rows.append("Full test, top 1\\% RV days removed & " + " & ".join(v))
write("subperiods", "Test QLIKE in four consecutive sub-periods and without extreme days", "tab:subperiods",
      "Period & HAR-RV & log-HAR & HAR-RV-LSTM & EGARCH-LSTM", rows, "lrrrr")

mz = pd.read_csv(T / "mincer_zarnowitz_test.csv").set_index("Model").loc[MAIN]
rows = [f"{m} & {r.alpha:.3f} & {r.beta:.3f} & {r['se(beta)']:.3f} & {r.R2:.3f} & {r['Mean forecast / mean RV']:.3f}"
        for m, r in mz.iterrows()]
write("mz", "Mincer--Zarnowitz regressions, test period: $\\log RV_{t+1}=\\alpha+\\beta\\log\\hat h_{t+1}$",
      "tab:mz", "Model & $\\alpha$ & $\\beta$ & s.e.($\\beta$) & $R^2$ & Mean forecast / mean RV", rows, "lrrrrr",
      note="A well-calibrated forecast has $\\beta\\approx1$. Standard errors clustered by date.")

ps = pd.read_csv(T / "per_stock_test_qlike.csv")
k = json.load(open(T / "key_numbers.json"))
wr = k["win_rates"]
rows = [f"{m} & {100 * wr['share beating HAR-RV'].get(m, np.nan):.0f} & "
        + (f"{100 * wr['share beating log-HAR'][m]:.0f}" if m in wr['share beating log-HAR'] and not pd.isna(wr['share beating log-HAR'][m]) else "--")
        for m in ["log-HAR", "HAR-RV-LSTM", "EGARCH-LSTM"]]
write("winrates", "Share of the 95 stocks on which a model has lower test QLIKE than the benchmark (\\%)",
      "tab:winrates", "Model & Beats HAR-RV & Beats log-HAR", rows, "lrr")

# ------------------------------------------------------------------ annex tables
eg = pd.read_csv(T / "egarch_parameters.csv")
cols = ["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"]
names = ["$\\mu$", "$\\omega$", "$\\alpha$", "$\\gamma$", "$\\beta$", "$\\nu$"]
rows = [f"{nm} & {eg[c].mean():.3f} & {eg[c].median():.3f} & {eg[c].quantile(.1):.3f} & {eg[c].quantile(.9):.3f}"
        for c, nm in zip(cols, names)]
rows.append(f"Share with $\\gamma<0$ & \\multicolumn{{4}}{{l}}{{{100 * (eg['gamma[1]'] < 0).mean():.0f}\\%}}")
write("egarch", "EGARCH(1,1)-t parameters across the 95 stocks (estimated on 2016--2022)", "tab:egarch",
      "Parameter & Mean & Median & 10th pct & 90th pct", rows, "lrrrr")

rows = []
for mname in ["HAR-RV", "log-HAR"]:
    c = pd.read_csv(T / f"coefficients_{mname}.csv")
    scale = 1e4 if mname == "HAR-RV" else 1
    rows.append(f"{mname} & " + " & ".join(
        f"{c[x].mean() * (scale if x == 'const' else 1):.3f} ({c[x].std() * (scale if x == 'const' else 1):.3f})"
        for x in ["const", "beta_D", "beta_W", "beta_M"]) +
        f" & {(c.beta_D + c.beta_W + c.beta_M).mean():.3f}")
write("harcoef", "HAR coefficients, mean (s.d.) across stocks, estimated on 2016--2024", "tab:harcoef",
      "Model & $\\beta_0$ & $\\beta_D$ & $\\beta_W$ & $\\beta_M$ & $\\beta_D+\\beta_W+\\beta_M$", rows, "lrrrrr",
      note="HAR-RV constant shown $\\times10^{4}$.")

liq = pd.read_csv(T / "liquidity_groups.csv")
ps = ps.merge(liq[["Symbol", "ILLIQ", "avg_daily_turnover_cr"]], on="Symbol").sort_values("Liquidity_Rank")
half = (len(ps) + 1) // 2
rows = []
for i in range(half):
    def cell(r):
        return (f"{r.Symbol} & {r.Liquidity_Group[0]} & {r['log-HAR']:.3f} & {r['HAR-RV-LSTM']:.3f} & "
                f"{pct(r.gain_vs_logHAR_pct)}")
    a = cell(ps.iloc[i])
    b = cell(ps.iloc[i + half]) if i + half < len(ps) else " & & & & "
    rows.append(f"{i + 1} & {a} & {i + half + 1 if i + half < len(ps) else ''} & {b}")
write("perstock", "Test QLIKE for every stock, ordered by liquidity rank", "tab:perstock",
      "\\# & Stock & L & log-HAR & LSTM & Gain\\% & \\# & Stock & L & log-HAR & LSTM & Gain\\%",
      rows, "rlcrrr|rlcrrr", size="\\scriptsize",
      note="L = liquidity group (H/M/L). LSTM = HAR-RV-LSTM. Gain = \\% reduction in QLIKE against log-HAR.")
# top RV days (annex)
panel = pd.read_parquet(ROOT / "data_processed" / "daily_panel.parquet")
top = panel.nlargest(10, "RV")
ev = {"2023-02-02": "Adani sell-off after short-seller report", "2023-02-03": "Adani sell-off",
      "2023-02-01": "Adani sell-off; Union Budget", "2020-03-13": "COVID-19: market-wide circuit, halt and rebound",
      "2020-03-18": "COVID-19 crash", "2020-03-23": "COVID-19: national lockdown announced",
      "2020-03-24": "COVID-19 crash"}
rows = [f"{r.Symbol} & {r.Date.date()} & {r.RV:.4f} & {100 * (252 * r.RV) ** 0.5:.0f} & {r.ret_cc:.1f} & {ev.get(str(r.Date.date()), 'stock-specific')}"
        for _, r in top.iterrows()]
write("topdays", "The ten highest-RV stock-days after cleaning", "tab:topdays",
      "Stock & Date & RV & Ann. vol (\\%) & Return (\\%) & Event", rows, "llrrrl")

# first version vs corrected
old = pd.read_csv(T / "first_version_test_metrics.csv").set_index("Model")
lg = pd.read_csv(T / "liquidity_groups_first_vs_corrected.csv")
agree = 100 * (lg.Group_first_version == lg.Group_corrected).mean()
rows = []
for m in ["HAR-RV", "HAR-RV-LSTM", "EGARCH-LSTM"]:
    if m == "HAR-RV":
        rows.append(f"HAR-RV & {old.loc[m, 'QLIKE']:.4f} & {te.loc[m, 'QLIKE']:.4f} & -- & --")
        continue
    rows.append(f"{m} & {old.loc[m, 'QLIKE']:.4f} & {te.loc[m, 'QLIKE']:.4f} & "
                f"{100 * (old.loc[m, 'QLIKE'] / old.loc['HAR-RV', 'QLIKE'] - 1):+.1f} & "
                f"{100 * (te.loc[m, 'QLIKE'] / te.loc['HAR-RV', 'QLIKE'] - 1):+.1f}")
rows.append(f"log-HAR & -- & {te.loc['log-HAR', 'QLIKE']:.4f} & -- & {100 * (te.loc['log-HAR', 'QLIKE'] / te.loc['HAR-RV', 'QLIKE'] - 1):+.1f}")
write("versions", "Test QLIKE in the first version of the project and after the corrections", "tab:versions",
      "Model & First version & Corrected & vs HAR-RV, first (\\%) & vs HAR-RV, corrected (\\%)", rows, "lrrrr",
      note=f"First version: 29,830 stock-days, uncleaned RV with overnight gaps, EGARCH fitted on the full sample, "
           f"liquidity groups from the mis-scaled slope (only {agree:.0f}\\% of stocks keep the same group after the correction). "
           "The two columns are not on identical samples; compare the relative columns.")
print("tables written to", OUT)
