"""CLI entry points: `semanticsd` (admin) and `ssearch` (client)."""
from __future__ import annotations
import json
import sys
import typer
import httpx
import uvicorn
from semanticsd import config, keychain, logging_setup, paths
from semanticsd.admin import install as admin_install


semanticsd_app = typer.Typer(no_args_is_help=True, help="SemanticsD daemon admin")
ssearch_app = typer.Typer(no_args_is_help=True, help="SemanticsD client CLI")
watch_app = typer.Typer(no_args_is_help=False, help="Inspect / control the FSEvents watcher")
power_app = typer.Typer(no_args_is_help=False, help="Inspect / switch power mode (active|saver)")
ssearch_app.add_typer(watch_app, name="watch")
ssearch_app.add_typer(power_app, name="power")


# ---- semanticsd admin ----

@semanticsd_app.command()
def serve():
    """Run the daemon (normally invoked by launchd)."""
    from contextlib import asynccontextmanager
    from semanticsd.db import connection, migrations
    from semanticsd.embedders import get_router
    from semanticsd.pipeline.indexer import Indexer
    from semanticsd.pipeline.worker import Worker
    from semanticsd.server.app import create_app
    from semanticsd.watcher.power import PowerController

    cfg = config.load()
    logging_setup.configure(level=cfg.daemon.log_level, to_file=True)
    paths.ensure_dirs()

    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)
    router = get_router()
    indexer = Indexer(
        conn=conn,
        max_file_size_mb=cfg.watch.max_file_size_mb,
        ignore_patterns=cfg.watch.ignore_patterns,
    )
    worker = Worker(conn=conn, router=router,
                    batch_size=cfg.embedding.text.batch_size if cfg.embedding.text else 128)
    power = PowerController(cfg, indexer, worker)

    @asynccontextmanager
    async def lifespan(_app):
        await power.startup()
        yield
        await power.shutdown()

    app = create_app(power_controller=power)
    app.router.lifespan_context = lifespan
    uvicorn.run(
        app,
        host=cfg.daemon.http_host,
        port=cfg.daemon.http_port,
        log_level=cfg.daemon.log_level,
        access_log=False,
    )


@semanticsd_app.command()
def install():
    """Install launchd plist + config + token. Idempotent."""
    result = admin_install.install()
    typer.echo("SemanticsD installed.")
    for a in result["actions"]:
        typer.echo(f"  - {a}")
    typer.echo(f"\nToken: {result['token_hint']}")
    typer.echo(f"Plist: {result['plist']}")


@semanticsd_app.command()
def uninstall():
    """Stop and remove launchd agent. Does NOT delete index/config."""
    result = admin_install.uninstall()
    typer.echo("SemanticsD uninstalled.")
    for a in result["actions"]:
        typer.echo(f"  - {a}")


token_app = typer.Typer(help="Auth-token management")
semanticsd_app.add_typer(token_app, name="token")


@token_app.command("print")
def token_print():
    """Print the current API auth token."""
    typer.echo(admin_install.print_token())


# ---- ssearch client ----

def _client() -> httpx.Client:
    cfg = config.load()
    tok = keychain.get_auth_token()
    if not tok:
        typer.echo("ERROR: no auth token in Keychain. Run `semanticsd install` first.", err=True)
        raise typer.Exit(2)
    return httpx.Client(
        base_url=f"http://{cfg.daemon.http_host}:{cfg.daemon.http_port}",
        headers={"X-Auth-Token": tok},
        timeout=10.0,
    )


@ssearch_app.callback(invoke_without_command=True)
def ssearch_root(
    ctx: typer.Context,
    query: list[str] = typer.Argument(None, help="Search query (positional, multi-word ok)"),
    mode: str = typer.Option("hybrid", "--mode", help="hybrid|semantic|filename|grep"),
    semantic: bool = typer.Option(False, "--semantic", help="Shortcut for --mode=semantic"),
    filename: bool = typer.Option(False, "--filename", help="Shortcut for --mode=filename"),
    grep: bool = typer.Option(False, "--grep", help="Shortcut for --mode=grep"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    all_scope: bool = typer.Option(False, "--all", help="Search whole corpus, not just CWD"),
    no_vision: bool = typer.Option(False, "--no-vision", help="Disable cross-modal vision search"),
    status: bool = typer.Option(False, "--status", help="Show daemon status."),
    presets: bool = typer.Option(False, "--presets", help="List available embedder presets."),
    test_embedder: str = typer.Option(
        "", "--test-embedder", metavar="PRESET",
        help="Round-trip test the embedder for the given preset.",
    ),
    index_path: str = typer.Option(
        "", "--index", metavar="PATH",
        help="Index a file or directory; runs the worker once when done.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
):
    if status:
        try:
            with _client() as c:
                r = c.get("/v1/health")
                r.raise_for_status()
                body = r.json()
        except httpx.HTTPError as e:
            typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
            raise typer.Exit(3)
        if json_output:
            typer.echo(json.dumps(body, indent=2))
        else:
            typer.echo(f"status:    {body['status']}")
            typer.echo(f"version:   {body['version']}")
            typer.echo(f"doc_count: {body['doc_count']}")
            emb = body.get("embedder", {})
            typer.echo(f"embedder:  {emb.get('message','')}")
            if emb.get("provider_id"):
                typer.echo(f"  provider: {emb['provider_id']}")
                typer.echo(f"  model:    {emb['model_id']}")
                typer.echo(f"  dim:      {emb['dim']}")
        return

    if presets:
        try:
            with _client() as c:
                r = c.get("/v1/presets")
                r.raise_for_status()
                body = r.json()
        except httpx.HTTPError as e:
            typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
            raise typer.Exit(3)
        if json_output:
            typer.echo(json.dumps(body, indent=2))
        else:
            for preset_id, info in body["presets"].items():
                key_flag = "(needs API key)" if info.get("needs_api_key") else ""
                url_flag = "(needs base URL)" if info.get("needs_base_url") else ""
                model = info.get("default_model") or "<user-pick>"
                typer.echo(f"  {preset_id:<20} model={model} {key_flag} {url_flag}".rstrip())
        return

    if test_embedder:
        body_req = {"preset": test_embedder}
        try:
            with _client() as c:
                r = c.post("/v1/embedder/test", json=body_req)
                r.raise_for_status()
                body = r.json()
        except httpx.HTTPError as e:
            typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
            raise typer.Exit(3)
        if json_output:
            typer.echo(json.dumps(body, indent=2))
        else:
            if body["ok"]:
                typer.echo(f"OK  preset={test_embedder}")
                typer.echo(f"  provider: {body['provider_id']}")
                typer.echo(f"  model:    {body['model_id']}")
                typer.echo(f"  dim:      {body['dim']}")
                typer.echo(f"  latency:  {body['latency_ms']}ms")
            else:
                typer.echo(f"FAIL preset={test_embedder}: {body.get('error','unknown error')}")
                raise typer.Exit(4)
        return

    if index_path:
        try:
            with _client() as c:
                r = c.post(
                    "/v1/index",
                    json={"path": index_path, "drain": True},
                    timeout=600.0,
                )
                r.raise_for_status()
                body = r.json()
        except httpx.HTTPError as e:
            typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
            raise typer.Exit(3)
        if json_output:
            typer.echo(json.dumps(body, indent=2))
        else:
            typer.echo(f"indexed:   {body.get('files_indexed', 0)} files")
            typer.echo(f"chunks:    {body.get('chunks_created', 0)}")
            typer.echo(f"queued:    {body.get('jobs_queued', 0)} jobs")
            typer.echo(f"drained:   {body.get('drained', 0)} jobs in this run")
            if body.get("files_skipped_unsupported"):
                typer.echo(f"skipped:   {body['files_skipped_unsupported']} unsupported files")
            if body.get("files_skipped_unchanged"):
                typer.echo(f"unchanged: {body['files_skipped_unchanged']} files")
        return

    if query:
        q = " ".join(query)
        if semantic:
            mode = "semantic"
        elif filename:
            mode = "filename"
        elif grep:
            mode = "grep"
        if mode not in ("hybrid", "semantic", "filename", "grep"):
            typer.echo(f"ERROR: unknown mode {mode!r}", err=True)
            raise typer.Exit(2)

        from pathlib import Path as _P
        params = {
            "q": q,
            "mode": mode,
            "limit": limit,
            "all": "true" if all_scope else "false",
            "vision": "false" if no_vision else "true",
        }
        if not all_scope:
            params["cwd"] = str(_P.cwd().resolve())

        try:
            with _client() as c:
                r = c.get("/v1/search", params=params, timeout=60.0)
                r.raise_for_status()
                body = r.json()
        except httpx.HTTPError as e:
            typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
            raise typer.Exit(3)

        if json_output:
            typer.echo(json.dumps(body, indent=2))
            return

        results = body.get("results", [])
        if not results:
            typer.echo(f"No matches for {q!r} (mode={mode}, took {body.get('took_ms', 0)}ms).")
            return
        for r in results:
            modes = r.get("metadata", {}).get("contributing_modes")
            mode_label = "+".join(sorted(set(modes))) if modes else r.get("mode", "?")
            line = f"{r['path']}  ({mode_label}, {r['modality']}, score={r['score']:.3f})"
            typer.echo(line)
            if r.get("snippet"):
                snip = r["snippet"]
                # Indent + dim-ish prefix for the snippet
                for sl in snip.splitlines() or [snip]:
                    typer.echo(f"  {sl}")
            typer.echo("")
        typer.echo(f"-- {len(results)} results in {body.get('took_ms', 0)}ms --")
        return

    if ctx.invoked_subcommand is None:
        typer.echo("Usage: ssearch [QUERY] | --status | --presets | --test-embedder PRESET | --index PATH")
        raise typer.Exit(0)


# ---- ssearch watch ----

@watch_app.callback(invoke_without_command=True)
def watch_status(
    ctx: typer.Context,
    sweep: bool = typer.Option(False, "--sweep", help="Force a full re-walk now."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
):
    """Show watcher status, or trigger a full sweep with --sweep."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        with _client() as c:
            if sweep:
                r = c.post("/v1/watch/sweep", timeout=600.0)
            else:
                r = c.get("/v1/watch")
            r.raise_for_status()
            body = r.json()
    except httpx.HTTPError as e:
        typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
        raise typer.Exit(3)

    if json_output:
        typer.echo(json.dumps(body, indent=2))
        return

    if sweep:
        s = body.get("stats", {})
        typer.echo(f"sweep complete in {s.get('elapsed_s','?')}s — "
                   f"{s.get('files_indexed', 0)} files indexed, "
                   f"{s.get('chunks_created', 0)} chunks, "
                   f"{s.get('jobs_queued', 0)} jobs queued.")
        return

    typer.echo(f"mode:               {body.get('mode')}")
    typer.echo(f"watcher_running:    {body.get('watcher_running')}")
    typer.echo(f"power_source:       {body.get('power_source')}")
    typer.echo(f"auto_saver:         {body.get('auto_saver_on_battery')}")
    typer.echo(f"saver_interval_s:   {body.get('saver_interval_s')}")
    typer.echo(f"dirty_pending:      {body.get('dirty_pending')}")
    typer.echo(f"last_sweep_at:      {body.get('last_sweep_at')}")
    dirs = body.get("directories", []) or []
    if dirs:
        typer.echo("directories:")
        for d in dirs:
            typer.echo(f"  {d}")
    else:
        typer.echo("directories:        (none configured — add to [watch].directories)")


# ---- ssearch power ----

@power_app.callback(invoke_without_command=True)
def power_status(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
):
    """Show current power mode."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        with _client() as c:
            r = c.get("/v1/power")
            r.raise_for_status()
            body = r.json()
    except httpx.HTTPError as e:
        typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
        raise typer.Exit(3)
    if json_output:
        typer.echo(json.dumps(body, indent=2))
        return
    typer.echo(f"mode:                  {body.get('mode')}")
    typer.echo(f"power_source:          {body.get('power_source')}")
    typer.echo(f"auto_saver_on_battery: {body.get('auto_saver_on_battery')}")


@power_app.command("active")
def power_active():
    """Switch to active mode (FSEvents watcher running)."""
    _power_set("active")


@power_app.command("saver")
def power_saver():
    """Switch to saver mode (watcher off, periodic sweep)."""
    _power_set("saver")


def _power_set(mode: str) -> None:
    try:
        with _client() as c:
            r = c.post("/v1/power", json={"mode": mode})
            r.raise_for_status()
            body = r.json()
    except httpx.HTTPError as e:
        typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
        raise typer.Exit(3)
    typer.echo(f"power mode -> {body.get('mode')}")


# ---- ssearch usage ----

usage_app = typer.Typer(no_args_is_help=False, help="Show embedder spend + volume")
ssearch_app.add_typer(usage_app, name="usage")


@usage_app.callback(invoke_without_command=True)
def usage_show(
    ctx: typer.Context,
    today: bool = typer.Option(False, "--today", help="Today only"),
    month: bool = typer.Option(False, "--this-month", help="Calendar month (default)"),
    all_time: bool = typer.Option(False, "--all", help="All recorded usage"),
    provider: str = typer.Option("", "--provider", help="Filter by provider_id"),
    since: str = typer.Option("", "--since", help="YYYY-MM-DD"),
    until: str = typer.Option("", "--until", help="YYYY-MM-DD"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    csv_output: bool = typer.Option(False, "--csv", help="CSV output (one row per provider)"),
):
    if ctx.invoked_subcommand is not None:
        return
    params: dict[str, str] = {}
    if since:
        params["since"] = since
    elif today:
        from datetime import datetime, timezone
        params["since"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    elif all_time:
        params["since"] = "1970-01-01"
    # else default = start of month (server-side default when 'since' omitted)
    if until:
        params["until"] = until
    if provider:
        params["provider"] = provider

    try:
        with _client() as c:
            r = c.get("/v1/usage", params=params, timeout=30.0)
            r.raise_for_status()
            body = r.json()
    except httpx.HTTPError as e:
        typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
        raise typer.Exit(3)

    if json_output:
        typer.echo(json.dumps(body, indent=2))
        return

    if csv_output:
        import csv as _csv
        from io import StringIO
        buf = StringIO()
        w = _csv.writer(buf)
        w.writerow(["provider_id", "model_id", "operation", "calls", "chunks",
                    "input_tokens", "cost_usd", "duration_ms"])
        for r in body.get("by_provider", []):
            w.writerow([r["provider_id"], r["model_id"], r["operation"],
                        r["calls"], r["chunks"], r["input_tokens"],
                        f"{r['cost_usd']:.6f}", r["duration_ms"]])
        typer.echo(buf.getvalue().rstrip())
        return

    typer.echo(f"calls:        {body.get('calls', 0)}")
    typer.echo(f"chunks:       {body.get('chunks', 0)}")
    typer.echo(f"tokens:       {body.get('input_tokens', 0):,}")
    typer.echo(f"total spend:  ${body.get('cost_usd', 0):.4f}")
    typer.echo("")
    rows = body.get("by_provider", [])
    if rows:
        typer.echo(f"  {'provider':<10} {'model':<30} {'op':<14} {'calls':>6} "
                   f"{'chunks':>6} {'tokens':>10} {'cost':>10}")
        for r in rows:
            typer.echo(
                f"  {r['provider_id']:<10} {r['model_id']:<30} "
                f"{r['operation']:<14} {r['calls']:>6} {r['chunks']:>6} "
                f"{r['input_tokens']:>10,} ${r['cost_usd']:>8.4f}"
            )


# ---- ssearch reembed ----

reembed_app = typer.Typer(no_args_is_help=False,
                          help="Queue re-embed jobs after switching embedder providers")
ssearch_app.add_typer(reembed_app, name="reembed")


def _reembed_call(modality: str) -> None:
    try:
        with _client() as c:
            r = c.post("/v1/reembed", json={"modality": modality}, timeout=60.0)
            r.raise_for_status()
            body = r.json()
    except httpx.HTTPError as e:
        typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
        raise typer.Exit(3)
    typer.echo(f"queued: text={body['queued']['text']}, vision={body['queued']['vision']} "
               f"(total {body['total']})")


@reembed_app.callback(invoke_without_command=True)
def reembed_root(ctx: typer.Context):
    """Queue both text and vision re-embed jobs (default)."""
    if ctx.invoked_subcommand is not None:
        return
    _reembed_call("all")


@reembed_app.command("text")
def reembed_text():
    """Queue text re-embed jobs only."""
    _reembed_call("text")


@reembed_app.command("vision")
def reembed_vision():
    """Queue vision re-embed jobs only."""
    _reembed_call("vision")
