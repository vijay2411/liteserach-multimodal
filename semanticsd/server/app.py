"""FastAPI app factory."""
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from semanticsd import __version__
from semanticsd.server.routes import (
    health, presets, embedder_test,
    index as index_route, search as search_route,
    watch as watch_route, power as power_route,
    usage as usage_route,
    reembed as reembed_route,
    image as image_route,
)

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(power_controller=None) -> FastAPI:
    """Build the FastAPI app.

    `power_controller` is the runtime PowerController instance. The /v1/watch
    and /v1/power endpoints reach into it via app.state. When None (e.g.
    tests not exercising the watcher), those endpoints respond with 503.
    """
    app = FastAPI(
        title="SemanticsD",
        version=__version__,
        description="Local semantic search daemon for macOS.",
    )
    app.state.power_controller = power_controller
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix="/v1")
    app.include_router(presets.router, prefix="/v1")
    app.include_router(embedder_test.router, prefix="/v1")
    app.include_router(index_route.router, prefix="/v1")
    app.include_router(search_route.router, prefix="/v1")
    app.include_router(watch_route.router, prefix="/v1")
    app.include_router(power_route.router, prefix="/v1")
    app.include_router(usage_route.router, prefix="/v1")
    app.include_router(reembed_route.router, prefix="/v1")
    app.include_router(image_route.router, prefix="/v1")

    # Serve the bundled web UI at /. The single-page app uses the same /v1
    # endpoints over fetch(); auth token is collected once via a setup modal
    # and stashed in localStorage. No build step or external assets.
    index_html = _STATIC_DIR / "index.html"
    if index_html.exists():
        @app.get("/", include_in_schema=False)
        def _ui_root():
            return FileResponse(index_html)

    return app
