from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
import uvicorn

# Run from either project root or src/.
THIS_FILE = Path(__file__).resolve()
CWD = Path.cwd().resolve()
if (CWD / "src").exists():
    sys.path.insert(0, str(CWD / "src"))
else:
    sys.path.insert(0, str(CWD))

HOST = os.environ.get("RETRIEVAL_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("RETRIEVAL_SERVER_PORT", "8765"))

app = FastAPI(title="JOKER retrieval permanent server", version="2.0")

print("Loading retrieval module...")
import retrieval  # noqa: E402

print("Loading RetrievalPipeline once. This is the only expensive startup step.")
_t0 = time.time()
ASSETS: dict[str, Any] = {
    "_retrieval_pipeline": retrieval.RetrievalPipeline(),
    "started_at": time.time(),
}
print(f"RetrievalPipeline loaded in {ASSETS['started_at'] - _t0:.1f}s")


def _logic():
    """Hot-reload retrieval_refactored.py without rebuilding ASSETS."""
    global retrieval
    retrieval = importlib.reload(retrieval)
    return retrieval


@app.get("/status")
def status() -> dict[str, Any]:
    pipe = ASSETS.get("_retrieval_pipeline")
    return {
        "ok": True,
        "assets_loaded": pipe is not None,
        "pipeline_class": type(pipe).__name__ if pipe is not None else None,
        "started_at": ASSETS.get("started_at"),
        "server_pid": os.getpid(),
    }


@app.post("/debug_row")
def debug_row(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        mod = _logic()
        return mod.debug_row(ASSETS, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/retrieve")
def retrieve_rows(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        mod = _logic()

        # Never let a stale active-job lock block development runs.
        # The server owns loaded models/indexes only; retrieval_refactored.py owns execution.
        ASSETS.pop("_active_job", None)
        ASSETS.pop("active_job", None)
        ASSETS["cancel_requested"] = False

        return mod.retrieve(ASSETS, payload)
    except Exception as e:
        ASSETS.pop("_active_job", None)
        ASSETS.pop("active_job", None)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/eval")
def eval_rows(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        mod = _logic()
        return mod.eval_rows(ASSETS, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run("retrieval_server:app", host=HOST, port=PORT, reload=False)
