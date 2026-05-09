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


# ---- semanticsd admin ----

@semanticsd_app.command()
def serve():
    """Run the daemon (normally invoked by launchd)."""
    cfg = config.load()
    logging_setup.configure(level=cfg.daemon.log_level, to_file=True)
    from semanticsd.server.app import create_app
    app = create_app()
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
