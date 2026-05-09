"""Budget gating: month-window cost aggregator + over-cap check.

A `BudgetGate` is a small per-process object that:
  - knows the configured monthly cap and warning threshold
  - computes month-to-date spend on demand from the usage table
  - exposes `can_spend(more_usd)` for the worker to consult before paid calls
  - logs a one-time warning when crossing the threshold
"""
from __future__ import annotations
import calendar
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def month_start_unix(now_unix: int | None = None) -> int:
    """Unix timestamp for 00:00:00 on the 1st of the current calendar month, UTC."""
    now = datetime.fromtimestamp(now_unix or time.time(), tz=timezone.utc)
    start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    return calendar.timegm(start.timetuple())


@dataclass
class BudgetStatus:
    spent_this_month_usd: float
    limit_usd: float
    percent_used: float       # 0..1+ (>1 means over cap)
    blocked: bool             # True iff a paid embed call would now be refused


class BudgetGate:
    def __init__(self, conn: sqlite3.Connection, monthly_limit_usd: float = 0.0,
                 warning_threshold: float = 0.8):
        self.conn = conn
        self.monthly_limit_usd = float(monthly_limit_usd)
        self.warning_threshold = float(warning_threshold)
        self._warned_threshold = False  # one-shot per process

    @property
    def is_active(self) -> bool:
        """A monthly_limit_usd of 0 disables the gate entirely (unlimited)."""
        return self.monthly_limit_usd > 0.0

    def spent_this_month(self) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM usage WHERE timestamp >= ?",
            (month_start_unix(),),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def status(self) -> BudgetStatus:
        spent = self.spent_this_month()
        if not self.is_active:
            return BudgetStatus(
                spent_this_month_usd=round(spent, 6),
                limit_usd=0.0,
                percent_used=0.0,
                blocked=False,
            )
        pct = spent / self.monthly_limit_usd if self.monthly_limit_usd > 0 else 0.0
        return BudgetStatus(
            spent_this_month_usd=round(spent, 6),
            limit_usd=self.monthly_limit_usd,
            percent_used=round(pct, 4),
            blocked=spent >= self.monthly_limit_usd,
        )

    def can_spend(self, additional_usd: float) -> bool:
        """Returns True if `additional_usd` MORE spent this month would still
        be within cap. Free calls (additional_usd == 0) always pass."""
        if not self.is_active:
            return True
        if additional_usd <= 0.0:
            return True
        spent = self.spent_this_month()
        new_total = spent + additional_usd
        if not self._warned_threshold and self.is_active and \
                new_total >= self.monthly_limit_usd * self.warning_threshold:
            log.warning(
                "budget: $%.4f / $%.2f used (%.0f%%) — approaching cap",
                new_total, self.monthly_limit_usd,
                100.0 * new_total / self.monthly_limit_usd,
            )
            self._warned_threshold = True
        if new_total > self.monthly_limit_usd:
            log.warning(
                "budget exceeded: would spend $%.4f, cap $%.2f",
                new_total, self.monthly_limit_usd,
            )
            return False
        return True
