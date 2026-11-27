import asyncio
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile

from doc_auth.deepfake.exceptions import (
    InvalidConfigError,
    ModelUnavailableError,
    VideoValidationError,
)
from doc_auth.deepfake.pipeline import DeepfakePipeline
from doc_auth.deepfake.visual_detector import VisualDetector

DEFAULT_DEEPFAKE_MODEL_PATH = "models/deepfake_detector.npz"
DEFAULT_DEEPFAKE_SHA256 = "040323df408ed884a2325f3161b314562a79988b9489d077ecc8472afab86844"


async def _read_upload_with_limit(
    file: UploadFile,
    max_upload_bytes: int = 100 * 1024 * 1024,
    chunk_size: int = 65536,
) -> bytes:
    total = 0
    chunks: list[bytes] = []
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Y?kl?n?n fayl maksimum limiti ({max_upload_bytes} bayt) a??r.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _resolve_model_config(
    model_path: str | Path | None = None,
    expected_sha256: str | None = None,
    threshold: float | None = None,
) -> tuple[str, str | None, float]:
    resolved_path = (
        str(model_path)
        if model_path is not None
        else os.getenv("DEEPFAKE_MODEL_PATH", DEFAULT_DEEPFAKE_MODEL_PATH)
    )
    if expected_sha256 is not None:
        resolved_sha = expected_sha256.strip() or None
    elif "DEEPFAKE_MODEL_SHA256" in os.environ:
        env_sha = os.environ["DEEPFAKE_MODEL_SHA256"].strip()
        resolved_sha = env_sha or None
    else:
        resolved_sha = DEFAULT_DEEPFAKE_SHA256

    raw_threshold = (
        threshold
        if threshold is not None
        else float(os.getenv("DEEPFAKE_THRESHOLD", "0.5"))
    )
    return resolved_path, resolved_sha, float(raw_threshold)


def build_deepfake_pipeline_from_config(
    model_path: str | Path | None = None,
    expected_sha256: str | None = None,
    threshold: float | None = None,
) -> tuple[DeepfakePipeline, dict[str, Any]]:
    resolved_path, resolved_sha, resolved_threshold = _resolve_model_config(
        model_path=model_path,
        expected_sha256=expected_sha256,
        threshold=threshold,
    )
    detector = VisualDetector(
        model_path=resolved_path,
        threshold=resolved_threshold,
        expected_sha256=resolved_sha,
    )
    model_info = detector.load_model()
    pipeline = DeepfakePipeline(visual_detector=detector)
    model_info["sha256"] = VisualDetector.compute_file_sha256(Path(resolved_path))
    model_info["threshold"] = resolved_threshold
    return pipeline, model_info


def create_deepfake_router(
    pipeline_runner: Callable[[str | Path], dict[str, Any]] | None = None,
    max_upload_bytes: int = 100 * 1024 * 1024,
    prefix: str = "/deepfake",
    model_path: str | Path | None = None,
    expected_sha256: str | None = None,
    threshold: float | None = None,
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["deepfake"])

    state: dict[str, Any] = {
        "ready": False,
        "error": None,
        "model_info": None,
        "pipeline": None,
    }

    if pipeline_runner is not None:
        runner = pipeline_runner
        state["ready"] = True
        state["model_info"] = {"status": "custom_runner"}
    else:
        try:
            built_pipeline, model_info = build_deepfake_pipeline_from_config(
                model_path=model_path,
                expected_sha256=expected_sha256,
                threshold=threshold,
            )
            state["pipeline"] = built_pipeline
            state["model_info"] = model_info
            state["ready"] = True
            state["error"] = None
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, ImportError) as exc:
            state["ready"] = False
            state["error"] = str(exc)
            state["model_info"] = None

        def _default_runner(path: str | Path) -> dict[str, Any]:
            active_pipeline = state.get("pipeline")
            if active_pipeline is None or not state.get("ready"):
                raise ModelUnavailableError(
                    state.get("error") or "Deepfake modeli haz?r deyil v? ya y?kl?n? bilm?di."
                )
            return active_pipeline.run(path)

        runner = _default_runner

    router.state = state  # type: ignore[attr-defined]

    @router.get("/readiness")
    async def deepfake_readiness() -> dict[str, Any]:
        if not state.get("ready"):
            raise HTTPException(
                status_code=503,
                detail={
                    "ready": False,
                    "status": "model_unavailable",
                    "error": state.get("error") or "Deepfake modeli y?kl?nm?yib.",
                },
            )
        return {
            "ready": True,
            "status": "ready",
            "model": state.get("model_info"),
        }

    @router.post("/analyze")
    async def analyze_deepfake_endpoint(
        file: Annotated[UploadFile, File(...)],
    ) -> dict[str, Any]:
        if not state.get("ready") and pipeline_runner is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model ?l?atan deyil: {state.get('error') or 'Model y?kl?nm?yib.'}",
            )
        if not file.filename:
            raise HTTPException(status_code=400, detail="Fayl ad? t?l?b olunur.")

        content = await _read_upload_with_limit(file, max_upload_bytes=max_upload_bytes)
        if not content:
            raise HTTPException(status_code=400, detail="Bo? fayl g?nd?rilib.")

        suffix = Path(file.filename).suffix or ".mp4"
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            result = await asyncio.to_thread(runner, tmp_path)
            if isinstance(result, dict):
                if result.get("status") == "model_unavailable":
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model ?l?atan deyil: {result.get('error') or 'Model y?kl?nm?yib.'}",
                    )
                return result
            return {"status": "success", "result": result}
        except (VideoValidationError, InvalidConfigError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=f"Model ?l?atan deyil: {exc}") from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Daxili analiz x?tas?: {exc}") from exc
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    return router


def attach_deepfake_router(
    app: FastAPI,
    pipeline_runner: Callable[[str | Path], dict[str, Any]] | None = None,
    max_upload_bytes: int = 100 * 1024 * 1024,
    prefix: str = "/deepfake",
    model_path: str | Path | None = None,
    expected_sha256: str | None = None,
    threshold: float | None = None,
) -> APIRouter:
    router = create_deepfake_router(
        pipeline_runner=pipeline_runner,
        max_upload_bytes=max_upload_bytes,
        prefix=prefix,
        model_path=model_path,
        expected_sha256=expected_sha256,
        threshold=threshold,
    )
    app.include_router(router)
    if not hasattr(app.state, "deepfake_states"):
        app.state.deepfake_states = []
    app.state.deepfake_states.append(router.state)  # type: ignore[attr-defined]
    return router
