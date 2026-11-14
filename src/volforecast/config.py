"""Project-wide settings: paths, sample periods and cleaning thresholds.

Every notebook imports from here, so a change made once applies everywhere.
"""
from pathlib import Path
import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT is the folder that contains src/, notebooks/, results/ ...
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The raw 5-minute CSV files (about 1 GB) are kept OUTSIDE the project folder.
# Looked for in <project>/data, then in the folder next to the project.
# Override with the environment variable VOL_DATA_DIR if your data lives elsewhere.
_candidates = [PROJECT_ROOT / "data", PROJECT_ROOT.parent / "data"]
RAW_DATA_DIR = Path(os.environ.get(
    "VOL_DATA_DIR", next((c for c in _candidates if c.exists()), _candidates[0])))

PROCESSED_DIR = PROJECT_ROOT / "data_processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FORECAST_DIR = RESULTS_DIR / "forecasts"
TABLE_DIR = RESULTS_DIR / "tables"
MODEL_DIR = RESULTS_DIR / "models"
FIGURE_DIR = PROJECT_ROOT / "figures"

for _d in (PROCESSED_DIR, FORECAST_DIR, TABLE_DIR, MODEL_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Sample periods (strictly chronological, never shuffled across time)
# --------------------------------------------------------------------------
HISTORY_START = "2015-11-01"   # extra history so 22-day features exist on 1 Jan 2016
TRAIN_START, TRAIN_END = "2016-01-01", "2022-12-31"
VALID_START, VALID_END = "2023-01-01", "2024-12-31"
TEST_START, TEST_END = "2025-01-01", "2026-04-30"

# --------------------------------------------------------------------------
# Cleaning rules for the 5-minute bars (NSE cash market, 09:15-15:30 IST)
# --------------------------------------------------------------------------
SESSION_FIRST_BAR = "09:15"        # first 5-minute bar of the regular session
SESSION_LAST_BAR = "15:25"         # last 5-minute bar (covers 15:25-15:30)
MIN_VALID_BARS = 60                # a day needs >= 60 of 75 bars with trades
BAD_TICK_MIN_MOVE = 0.05           # |5-min log return| that triggers the spike check
BAD_TICK_REVERSAL = 0.25           # spike is "bad" if the next bar undoes >= 75% of it
CORPORATE_ACTION_GAP = 0.35        # |overnight log return| treated as split / bonus

# Listing dates for stocks whose files contain pre-listing rows
# (bond prices, placeholder prices or zero-volume quotes before the IPO).
LISTING_DATES = {
    "IRFC": "2021-01-29",
    "SBILIFE": "2017-10-03",
    "HDFCLIFE": "2017-11-17",
    "MAZDOCK": "2020-10-12",
}

# --------------------------------------------------------------------------
# Model settings
# --------------------------------------------------------------------------
HAR_FEATURES = ["RV", "RV_5", "RV_22"]
TARGET = "RV_t1"
LOOKBACK = 22
SEEDS = [11, 22, 33]
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
