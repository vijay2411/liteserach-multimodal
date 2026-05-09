"""FastAPI app factory."""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from semanticsd import __version__
from semanticsd.server.routes import health, presets, embedder_test, index as index_route


def create_app() -> FastAPI:
    app = FastAPI(
        title="SemanticsD",
        version=__version__,
        description="Local semantic search daemon for macOS.",
    )
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
    return app
