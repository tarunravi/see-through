from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse


APP_VERSION = "0.1.0"
JOBS_ROOT = Path(os.environ.get("SEE_THROUGH_JOBS_ROOT", "/workspace/jobs"))
SEE_THROUGH_SCRIPT = Path(
    os.environ.get(
        "SEE_THROUGH_SCRIPT",
        "/workspace/see-through/inference/scripts/inference_psd.py",
    )
)
PYTHON_BIN = os.environ.get("SEE_THROUGH_PYTHON", sys.executable)

app = FastAPI(title="Avatar See-through Decomposition API", version=APP_VERSION)
_job_lock = threading.Lock()


def _job_dir(job_id: str) -> Path:
    if "/" in job_id or ".." in job_id:
        raise HTTPException(status_code=400, detail="invalid job id")
    return JOBS_ROOT / job_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_job(job_id: str) -> dict[str, Any]:
    path = _job_dir(job_id) / "job.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _list_files(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                }
            )
    return files


def _run_job(job_id: str) -> None:
    job_root = _job_dir(job_id)
    input_path = job_root / "input" / "source.png"
    output_dir = job_root / "output"
    log_path = job_root / "run.log"
    job_path = job_root / "job.json"
    job = _read_job(job_id)

    with _job_lock:
        start = time.time()
        job.update({"status": "running", "started_at": start})
        _write_json(job_path, job)

        args = job["request"]
        cmd = [
            PYTHON_BIN,
            str(SEE_THROUGH_SCRIPT),
            "--srcp",
            str(input_path),
            "--save_dir",
            str(output_dir),
            "--resolution",
            str(args["resolution"]),
            "--resolution_depth",
            str(args["resolution_depth"]),
            "--inference_steps",
            str(args["inference_steps"]),
        ]
        if args["save_to_psd"]:
            cmd.append("--save_to_psd")
        if args["tblr_split"]:
            cmd.append("--tblr_split")
        if args["group_offload"]:
            cmd.append("--group_offload")

        output_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log:
            process = subprocess.run(
                cmd,
                cwd=SEE_THROUGH_SCRIPT.parents[2],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        duration = time.time() - start
        job.update(
            {
                "status": "succeeded" if process.returncode == 0 else "failed",
                "finished_at": time.time(),
                "duration_seconds": round(duration, 3),
                "returncode": process.returncode,
                "files": _list_files(output_dir),
                "log_path": "run.log",
            }
        )
        _write_json(job_path, job)


@app.get("/health")
def health() -> dict[str, Any]:
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
        }
    except Exception as exc:  # pragma: no cover - health diagnostic only
        cuda = {"available": False, "error": str(exc)}

    return {
        "service": "avatar-see-through",
        "version": APP_VERSION,
        "status": "ok",
        "jobs_root": str(JOBS_ROOT),
        "see_through_script": str(SEE_THROUGH_SCRIPT),
        "cuda": cuda,
    }


@app.post("/v1/decompositions", status_code=202)
async def create_decomposition(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    resolution: int = Form(1024),
    resolution_depth: int = Form(768),
    inference_steps: int = Form(20),
    group_offload: bool = Form(True),
    save_to_psd: bool = Form(True),
    tblr_split: bool = Form(True),
) -> dict[str, Any]:
    if resolution < 512 or resolution > 1600:
        raise HTTPException(status_code=400, detail="resolution must be 512..1600")
    if resolution_depth < 256 or resolution_depth > 1280:
        raise HTTPException(status_code=400, detail="resolution_depth must be 256..1280")
    if inference_steps < 1 or inference_steps > 60:
        raise HTTPException(status_code=400, detail="inference_steps must be 1..60")

    job_id = uuid.uuid4().hex
    job_root = _job_dir(job_id)
    input_dir = job_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / "source.png"
    input_path.write_bytes(await image.read())

    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "input_path": "input/source.png",
        "request": {
            "filename": image.filename,
            "resolution": resolution,
            "resolution_depth": resolution_depth,
            "inference_steps": inference_steps,
            "group_offload": group_offload,
            "save_to_psd": save_to_psd,
            "tblr_split": tblr_split,
        },
        "files": [],
    }
    _write_json(job_root / "job.json", job)
    background_tasks.add_task(_run_job, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/v1/decompositions/{job_id}",
    }


@app.get("/v1/decompositions/{job_id}")
def get_decomposition(job_id: str) -> dict[str, Any]:
    return _read_job(job_id)


@app.get("/v1/decompositions/{job_id}/artifacts/{artifact_path:path}")
def get_artifact(job_id: str, artifact_path: str) -> FileResponse:
    root = _job_dir(job_id)
    path = (root / artifact_path).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise HTTPException(status_code=400, detail="invalid artifact path")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
