"""Global configuration for the NIFTY backtesting framework."""

from __future__ import annotations

from pathlib import Path


# Free data source settings
USE_SYNTHETIC_INTRADAY = True
SYNTHETIC_SEED = 42
DATA_SOURCE_PRIORITY = ["jugaad", "bhav", "niftyindices"]
NSE_REQUEST_DELAY = 1.5

NSE_HOLIDAYS_2023 = [
    "2023-01-26",
    "2023-03-07",
    "2023-03-30",
    "2023-04-04",
    "2023-04-14",
    "2023-05-01",
    "2023-06-28",
    "2023-08-15",
    "2023-09-19",
    "2023-10-02",
    "2023-10-24",
    "2023-11-27",
]
NSE_HOLIDAYS_2024 = [
    "2024-01-22",
    "2024-01-26",
    "2024-03-25",
    "2024-04-09",
    "2024-04-14",
    "2024-04-17",
    "2024-05-23",
    "2024-06-17",
    "2024-07-17",
    "2024-08-15",
    "2024-10-02",
    "2024-10-14",
    "2024-11-01",
    "2024-11-15",
    "2024-11-20",
    "2024-12-25",
]

BACKTEST_START = "2024-01-01"
BACKTEST_END = "2024-12-31"
INTERVAL = "5minute"
ATM_ROUNDING = 50

BB_PERIOD = 20
BB_STD = 2

RSI_PERIOD = 9
RSI_MA_SHORT = 3
RSI_MA_LONG = 21

SR_CONFLUENCE_THRESHOLD = 15

ENTRY_START_TIME = "09:20"
ENTRY_CUTOFF_TIME = "13:00"
EOD_EXIT_TIME = "15:15"

CAPITAL = 500000
LOT_SIZE = 75
COOLDOWN_CANDLES = 2

CACHE_DIR = "nifty_backtest/data/cache/"

LOG_LEVEL = "INFO"
REPORT_DIR = "nifty_backtest/reports/"

SPOT_SYMBOL = "NIFTY"
SPOT_EXCHANGE = "NSE"
OPTION_SYMBOL = "NIFTY"
OPTION_EXCHANGE = "NFO"
VIX_SYMBOL = "INDIA VIX"
VIX_EXCHANGE = "NSE"

STRATEGY_CLASS_PATH = "strategy.atm_strategy:ATMStrategy"
PREMIUM_SL_TYPE = "conservative"
SR_TOLERANCE = 15
TARGET1_SPOT_MOVE = 65
TARGET2_SPOT_MOVE = 125
STOPLOSS_SPOT_MOVE = 27
VIX_LOW_THRESHOLD = 13
VIX_HIGH_THRESHOLD = 15
VIX_NORMAL_QTY_MULTIPLIER = 1.0
VIX_REDUCED_QTY_MULTIPLIER = 0.75
VIX_HALF_QTY_MULTIPLIER = 0.5
PREMIUM_SL_CONSERVATIVE_MULTIPLIER = 0.70
PREMIUM_SL_AGGRESSIVE_MULTIPLIER = 0.50

# EOD backtest settings
EOD_MAX_HOLD_DAYS = 4
EOD_ENTRY_ON_NEXT_OPEN = True


def resolve_cache_dir() -> Path:
    """Return the cache directory as an absolute path."""

    return Path(CACHE_DIR).resolve()


def resolve_report_dir() -> Path:
    """Return the report directory as an absolute path."""

    return Path(REPORT_DIR).resolve()
