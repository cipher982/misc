from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path


def mount_frontend(app: FastAPI, dist_dir: str | None = None) -> None:
    # Enable CORS for local dev UI
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if dist_dir is None:
        dist_dir = str(Path(__file__).resolve().parent.parent / "web" / "dist")
    path = Path(dist_dir)
    if path.exists():
        assets = path / "assets"
        if assets.exists():
            # Serve built assets at /assets
            app.mount("/assets", StaticFiles(directory=str(assets), html=False), name="assets")

        index_file = path / "index.html"
        if index_file.exists():
            @app.get("/ui")
            async def ui_index() -> FileResponse:  # type: ignore[override]
                return FileResponse(str(index_file))

            @app.get("/ui/{path:path}")
            async def ui_catch_all(path: str) -> FileResponse:  # type: ignore[override]
                # SPA fallback
                return FileResponse(str(index_file))
