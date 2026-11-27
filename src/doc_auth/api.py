import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile

from doc_auth.deepfake.api import attach_deepfake_router
from doc_auth.schemas.pipeline_schemas import DocumentAuthSummary
from doc_auth.service import AnalysisService

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
CHUNK_SIZE = 65536

import shutil
import subprocess as _sp


def check_readiness_state(service_obj: Any, required_langs: tuple[str, ...] = ("eng", "aze")) -> tuple[bool, dict[str, Any]]:
    model_ok = service_obj is not None
    if model_ok and hasattr(service_obj, "is_ready") and callable(service_obj.is_ready):
        try:
            model_ok = bool(service_obj.is_ready())
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError):
            model_ok = False

    tess_bin = shutil.which("tesseract")
    installed_langs: list[str] = []
    ocr_engine = None
    if tess_bin:
        ocr_engine = "tesseract"
        try:
            proc = _sp.run([tess_bin, "--list-langs"], check=False, capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                installed_langs = [
                    ln.strip() for ln in proc.stdout.splitlines()[1:] if ln.strip()
                ]
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError):
            installed_langs = []
    else:
        try:
            import easyocr  # type: ignore  # noqa: F401
            ocr_engine = "easyocr"
            installed_langs = ["eng", "aze"]
        except (ImportError, OSError, RuntimeError, ValueError, TypeError):
            ocr_engine = None

    missing_langs = [lg for lg in required_langs if lg not in installed_langs]
    ocr_ok = ocr_engine is not None and len(missing_langs) == 0
    ready_ok = model_ok and ocr_ok
    payload: dict[str, Any] = {
        "status": "ready" if ready_ok else "not_ready",
        "model_ready": model_ok,
        "ocr_ready": ocr_ok,
        "ocr_engine": ocr_engine,
        "installed_langs": installed_langs,
        "missing_langs": missing_langs,
    }
    return ready_ok, payload



def _validate_magic_bytes(ext: str, head: bytes) -> bool:
    if not head:
        return False
    if ext == ".pdf":
        return head.startswith(b"%PDF-")
    if ext == ".png":
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    if ext in {".jpg", ".jpeg"}:
        return head.startswith(b"\xff\xd8\xff")
    if ext == ".webp":
        return len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"WEBP"
    if ext in {".tif", ".tiff"}:
        return head.startswith((b"II*\x00", b"MM\x00*"))
    return True


def create_document_router(
    analysis_service: Any | None = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["documents"])
    service = analysis_service if analysis_service is not None else AnalysisService()
    job_store: dict[str, dict[str, Any]] = {}
    idempotency_index: dict[str, str] = {}

    async def _save_validated_upload(file: UploadFile, prefix: str) -> tuple[Path, int]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Fayl ad? t?l?b olunur.")

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="D?st?kl?nm?y?n fayl format?.",
            )

        temp_dir = Path("tmp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{prefix}_{uuid.uuid4().hex}{ext}"
        total_bytes = 0
        header = b""

        try:
            with temp_path.open("wb") as out_file:
                while True:
                    chunk = await file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    if len(header) < 16:
                        needed = 16 - len(header)
                        header += chunk[:needed]
                    total_bytes += len(chunk)
                    if total_bytes > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Fayl ?l??s? icaz? veril?n h?ddi ke?ir.",
                        )
                    out_file.write(chunk)
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise

        if total_bytes == 0:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Y?kl?n?n fayl bo?dur.")

        if not _validate_magic_bytes(ext, header):
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Fayl m?zmunu uzant?ya uy?un deyil.")

        return temp_path, total_bytes

    @router.post(
        "/analyze",
        response_model=DocumentAuthSummary,
        summary="Analyze uploaded document file (multipart/form-data)",
        responses={
            200: {"model": DocumentAuthSummary, "description": "Document authentication summary"},
            400: {"description": "Invalid file, unsupported format, empty file, or magic byte mismatch"},
            413: {"description": "Uploaded file exceeds maximum allowed size"},
            500: {"description": "Internal error during document analysis"},
        },
    )
    @router.post("/analyze-file", response_model=DocumentAuthSummary, include_in_schema=False)
    async def analyze_document_file_endpoint(
        file: Annotated[UploadFile, File(...)],
    ) -> DocumentAuthSummary:
        temp_path: Path | None = None
        try:
            temp_path, _ = await _save_validated_upload(file, "upload")
            summary = service.analyze_summary(temp_path)
            if isinstance(summary, DocumentAuthSummary):
                return summary
            if isinstance(summary, dict):
                return DocumentAuthSummary.model_validate(summary)
            raise HTTPException(status_code=500, detail="S?n?d analizi icra edil? bilm?di.")
        except HTTPException:
            raise
        except ValueError:
            raise HTTPException(status_code=400, detail="Fayl m?zmunu emal edil? bilm?di.")
        except (OSError, RuntimeError, TypeError, KeyError, AttributeError):
            raise HTTPException(status_code=500, detail="S?n?d analizi zaman? daxili x?ta ba? verdi.")
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass


    @router.post(
        "/analyze/json",
        response_model=dict[str, Any],
        summary="Analyze document via JSON payload (file_path)",
        responses={
            200: {"description": "Document authentication summary dictionary"},
            404: {"description": "Specified file_path does not exist"},
            422: {"description": "Missing or invalid file_path in JSON payload"},
            501: {"description": "JSON file analysis not supported by service"},
        },
        openapi_extra={
            "requestBody": {
                "content": {
                    "application/json": {
                        "examples": {
                            "file_path_request": {
                                "summary": "Local document file path",
                                "value": {
                                    "document_id": "doc_001",
                                    "file_path": "samples/sample_en.pdf",
                                    "languages": ["eng", "aze"],
                                },
                            },
                            "base64_request": {
                                "summary": "Base64 document payload",
                                "value": {
                                    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
                                    "document_type_hint": "passport",
                                },
                            },
                        }
                    }
                }
            }
        },
    )
    async def analyze_document_json_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
        file_ref = payload.get("file_path") or payload.get("path")
        if not file_ref or not isinstance(file_ref, str):
            raise HTTPException(status_code=422, detail="JSON sor?usunda 'file_path' sah?si t?l?b olunur.")
        target = Path(file_ref)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="G?st?ril?n fayl tap?lmad?.")
        if not hasattr(service, "analyze_summary"):
            raise HTTPException(status_code=501, detail="JSON fayl analizi d?st?kl?nmir.")
        summary = service.analyze_summary(target)
        if hasattr(summary, "model_dump"):
            return summary.model_dump()
        if isinstance(summary, dict):
            return summary
        raise HTTPException(status_code=500, detail="S?n?d analizi icra edil? bilm?di.")

    @router.post("/jobs")
    async def create_analysis_job(
        file: Annotated[UploadFile, File(...)],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if idempotency_key and idempotency_key in idempotency_index:
            existing_id = idempotency_index[idempotency_key]
            existing = job_store.get(existing_id, {})
            return {
                "job_id": existing_id,
                "idempotent_replay": True,
                "status": existing.get("status", "completed"),
                "result": existing.get("result"),
            }

        temp_path: Path | None = None
        try:
            temp_path, _ = await _save_validated_upload(file, "job")
            summary = service.analyze_summary(temp_path)
            job_id = uuid.uuid4().hex
            result_payload = summary.model_dump() if hasattr(summary, "model_dump") else dict(summary)
            record = {"job_id": job_id, "status": "completed", "result": result_payload}
            job_store[job_id] = record
            if idempotency_key:
                idempotency_index[idempotency_key] = job_id
            return record
        except HTTPException:
            raise
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, AttributeError):
            raise HTTPException(status_code=500, detail="Tap??r?q icra edil?rk?n daxili x?ta ba? verdi.")
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    @router.get("/jobs/{job_id}")
    async def get_analysis_job(job_id: str) -> dict[str, Any]:
        stored = job_store.get(job_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="Tap??r?q tap?lmad?.")
        return stored

    return router


def create_app(
    analysis_service: Any | None = None,
    deepfake_runner: Any | None = None,
) -> FastAPI:
    app = FastAPI(title="Document & Deepfake Authentication API", version="1.0.0")

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def readiness_check() -> dict[str, Any]:
        svc = analysis_service if analysis_service is not None else AnalysisService()
        ok, payload = check_readiness_state(svc)
        if not ok:
            raise HTTPException(status_code=503, detail=payload)
        return payload

    app.include_router(create_document_router(analysis_service=analysis_service))
    attach_deepfake_router(app, pipeline_runner=deepfake_runner)
    return app


app = create_app()
