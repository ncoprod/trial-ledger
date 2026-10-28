"""trial-ledger: a tamper-evident ledger of backtest trials with multiple-testing statistics."""

from .hooks import emit
from .ledger import Ledger, record_trial
from .report import build_report, format_report
from .stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    purged_kfold,
    sharpe_moments,
)

__version__ = "0.1.0"
__all__ = [
    "Ledger",
    "build_report",
    "deflated_sharpe_ratio",
    "emit",
    "expected_max_sharpe",
    "format_report",
    "pbo_cscv",
    "probabilistic_sharpe_ratio",
    "purged_kfold",
    "record_trial",
    "sharpe_moments",
]
