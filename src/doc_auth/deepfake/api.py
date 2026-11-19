import asyncio
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile

from doc_auth.deepfake.exceptions import InvalidConfigError, ModelUnavailableError, VideoValidationError
from doc_auth.deepfake.pipeline import analyze_video_deepfake


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
                detail=f"Yüklənən fayl maksimum limiti ({max_upload_bytes} bayt) aşır.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def create_deepfake_router(
    pipeline_runner: Optional[Callable[[str | Path], Dict[str, Any]]] = None,
    max_upload_bytes: int = 100 * 1024 * 1024,
    prefix: str = "/deepfake",
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["deepfake"])
    runner = pipeline_runner or (lambda path: analyze_video_deepfake(path))

    @router.post("/analyze")
    async def analyze_deepfake_endpoint(
        file: UploadFile = File(...),
    ) -> Dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Fayl adı tələb olunur.")

        content = await _read_upload_with_limit(file, max_upload_bytes=max_upload_bytes)
        if not content:
            raise HTTPException(status_code=400, detail="Boş fayl göndərilib.")

        suffix = Path(file.filename).suffix or ".mp4"
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            result = await asyncio.to_thread(runner, tmp_path)
            if isinstance(result, dict):
                return result
            return {"status": "success", "result": result}
        except (VideoValidationError, InvalidConfigError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=f"Model əlçatan deyil: {exc}") from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Daxili analiz xətası: {exc}") from exc
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    return router


def attach_deepfake_router(
    app: FastAPI,
    pipeline_runner: Optional[Callable[[str | Path], Dict[str, Any]]] = None,
    max_upload_bytes: int = 100 * 1024 * 1024,
    prefix: str = "/deepfake",
) -> APIRouter:
    router = create_deepfake_router(
        pipeline_runner=pipeline_runner,
        max_upload_bytes=max_upload_bytes,
        prefix=prefix,
    )
    app.include_router(router)
    return router