"""Cost tracking + budget enforcement.

The DB schema's `usage` table records one row per embedder call. This
package owns all reads and writes to that table.
"""
from semanticsd.usage.recorder import UsageEvent, record_usage, compute_cost
from semanticsd.usage.budget import BudgetGate, BudgetStatus, month_start_unix
from semanticsd.usage.reports import (
    UsageRow,
    UsageTotals,
    aggregate_by_provider,
    totals,
)

__all__ = [
    "UsageEvent",
    "record_usage",
    "compute_cost",
    "BudgetGate",
    "BudgetStatus",
    "month_start_unix",
    "UsageRow",
    "UsageTotals",
    "aggregate_by_provider",
    "totals",
]
