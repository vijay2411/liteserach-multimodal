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
    status: bool = typer.Option(False, "--status", help="Show daemon status."),
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
            typer.echo(f"embedder:  {body['embedder']['message']}")
        return

    if ctx.invoked_subcommand is None:
        typer.echo("Usage: ssearch [QUERY] | --status | <subcommand>")
        typer.echo("Search subcommands land in Plan 5.")
        raise typer.Exit(0)
