"""Usage recorder + budget gate + reports."""
import time
from semanticsd.db import connection, migrations
from semanticsd.usage.recorder import UsageEvent, record_usage, compute_cost
from semanticsd.usage.budget import BudgetGate, month_start_unix
from semanticsd.usage.reports import aggregate_by_provider, totals


def _fresh_conn(tmp_path):
    db = tmp_path / "u.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    return conn


def _insert_usage(conn, *, provider_id, model_id, op, tokens, cost, when=None):
    when = when if when is not None else int(time.time())
    conn.execute(
        "INSERT INTO usage(timestamp, provider_id, model_id, operation, "
        "input_tokens, cost_usd, chunk_count, duration_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, 1, 50)",
        (when, provider_id, model_id, op, tokens, cost),
    )


# --- recorder ---

def test_compute_cost_zero_rate_is_free():
    assert compute_cost(1_000_000, 0.0) == 0.0
    assert compute_cost(1_000_000, 0.0) == 0.0


def test_compute_cost_basic():
    # 1M tokens at $0.15/M = $0.15
    assert abs(compute_cost(1_000_000, 0.15) - 0.15) < 1e-9
    # 500k tokens at $0.30/M = $0.15
    assert abs(compute_cost(500_000, 0.30) - 0.15) < 1e-9


def test_record_usage_inserts_row(tmp_path):
    conn = _fresh_conn(tmp_path)
    record_usage(conn, UsageEvent(
        provider_id="gemini", model_id="gemini-embedding-2",
        operation="text_embed", input_tokens=1234,
        chunk_count=4, duration_ms=120, cost_usd=0.000185,
    ))
    rows = conn.execute("SELECT * FROM usage").fetchall()
    assert len(rows) == 1
    cols = {d[0] for d in conn.execute("SELECT * FROM usage LIMIT 1").description}
    assert {"provider_id", "model_id", "input_tokens", "cost_usd"} <= cols


# --- budget gate ---

def test_budget_inactive_when_limit_zero(tmp_path):
    conn = _fresh_conn(tmp_path)
    g = BudgetGate(conn, monthly_limit_usd=0.0)
    assert g.is_active is False
    assert g.can_spend(100.0) is True
    s = g.status()
    assert s.blocked is False
    assert s.limit_usd == 0.0


def test_budget_can_spend_within_cap(tmp_path):
    conn = _fresh_conn(tmp_path)
    g = BudgetGate(conn, monthly_limit_usd=5.0, warning_threshold=0.8)
    _insert_usage(conn, provider_id="gemini", model_id="g", op="text_embed",
                  tokens=1, cost=2.0)
    assert g.spent_this_month() == 2.0
    assert g.can_spend(1.0) is True   # 2.0 + 1.0 = 3.0 < 5.0


def test_budget_blocks_when_exceeded(tmp_path):
    conn = _fresh_conn(tmp_path)
    g = BudgetGate(conn, monthly_limit_usd=5.0)
    _insert_usage(conn, provider_id="gemini", model_id="g", op="text_embed",
                  tokens=1, cost=4.5)
    assert g.can_spend(1.0) is False  # 4.5 + 1.0 > 5.0


def test_budget_status_payload(tmp_path):
    conn = _fresh_conn(tmp_path)
    g = BudgetGate(conn, monthly_limit_usd=10.0)
    _insert_usage(conn, provider_id="gemini", model_id="g", op="text_embed",
                  tokens=1, cost=2.5)
    s = g.status()
    assert s.spent_this_month_usd == 2.5
    assert s.limit_usd == 10.0
    assert abs(s.percent_used - 0.25) < 1e-6
    assert s.blocked is False


def test_budget_excludes_previous_month(tmp_path):
    """A month-old usage row should not count toward this month's spend."""
    conn = _fresh_conn(tmp_path)
    g = BudgetGate(conn, monthly_limit_usd=5.0)
    # 40 days ago — definitely last month
    _insert_usage(conn, provider_id="gemini", model_id="g", op="text_embed",
                  tokens=1, cost=10.0,
                  when=int(time.time()) - 40 * 86400)
    assert g.spent_this_month() == 0.0
    assert g.can_spend(2.0) is True


def test_budget_free_calls_always_allowed(tmp_path):
    """additional_usd=0 (local provider) bypasses the gate even when over."""
    conn = _fresh_conn(tmp_path)
    g = BudgetGate(conn, monthly_limit_usd=1.0)
    _insert_usage(conn, provider_id="gemini", model_id="g", op="text_embed",
                  tokens=1, cost=2.0)  # already over
    assert g.can_spend(0.0) is True


# --- reports ---

def test_totals_groups_by_provider_model_op(tmp_path):
    conn = _fresh_conn(tmp_path)
    _insert_usage(conn, provider_id="gemini", model_id="g2", op="text_embed",
                  tokens=100, cost=0.1)
    _insert_usage(conn, provider_id="gemini", model_id="g2", op="text_embed",
                  tokens=200, cost=0.2)
    _insert_usage(conn, provider_id="ollama", model_id="emb", op="text_embed",
                  tokens=500, cost=0.0)

    t = totals(conn)
    assert t.calls == 3
    assert t.input_tokens == 800
    assert abs(t.cost_usd - 0.3) < 1e-9
    # gemini row aggregated, ollama separate
    by_p = {r.provider_id for r in t.by_provider}
    assert by_p == {"gemini", "ollama"}
    gemini_row = next(r for r in t.by_provider if r.provider_id == "gemini")
    assert gemini_row.calls == 2
    assert abs(gemini_row.cost_usd - 0.3) < 1e-9


def test_aggregate_filters_by_window(tmp_path):
    conn = _fresh_conn(tmp_path)
    now = int(time.time())
    _insert_usage(conn, provider_id="x", model_id="m", op="text_embed",
                  tokens=10, cost=1.0, when=now - 100)  # in window
    _insert_usage(conn, provider_id="x", model_id="m", op="text_embed",
                  tokens=20, cost=2.0, when=now - 1000)  # out of window
    rows = aggregate_by_provider(conn, since_unix=now - 200)
    assert len(rows) == 1
    assert rows[0].calls == 1


def test_aggregate_filters_by_provider(tmp_path):
    conn = _fresh_conn(tmp_path)
    _insert_usage(conn, provider_id="a", model_id="m", op="text_embed",
                  tokens=10, cost=1.0)
    _insert_usage(conn, provider_id="b", model_id="m", op="text_embed",
                  tokens=20, cost=2.0)
    rows = aggregate_by_provider(conn, provider="a")
    assert [r.provider_id for r in rows] == ["a"]


def test_month_start_is_stable():
    """month_start_unix should be idempotent for the same wall-clock instant."""
    t = int(time.time())
    a = month_start_unix(t)
    b = month_start_unix(t)
    assert a == b
    # Within the current month:
    assert a <= t
