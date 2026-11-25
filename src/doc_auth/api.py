from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile

from doc_auth.deepfake.api import attach_deepfake_router
from doc_auth.schemas.pipeline_schemas import DocumentAuthSummary
from doc_auth.service import AnalysisService

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
CHUNK_SIZE = 65536


def create_document_router(
    analysis_service: Optional[Any] = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["documents"])
    service = analysis_service if analysis_service is not None else AnalysisService()

    @router.post("/analyze", response_model=DocumentAuthSummary)
    async def analyze_document_endpoint(file: UploadFile = File(...)) -> DocumentAuthSummary:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Fayl ad? t?l?b olunur.")

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"D?st?kl?nm?y?n fayl format?: {ext or 'nam?lum'}.",
            )

        temp_dir = Path("tmp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"upload_{uuid.uuid4().hex}{ext}"
        temp_path = temp_dir / unique_name
        total_bytes = 0

        try:
            with temp_path.open("wb") as out_file:
                while True:
                    chunk = await file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Fayl ?l??s? icaz? veril?n h?ddi ke?ir.",
                        )
                    out_file.write(chunk)

            if total_bytes == 0:
                raise HTTPException(status_code=400, detail="Y?kl?n?n fayl bo?dur.")

            summary = service.analyze_summary(temp_path)
            if isinstance(summary, DocumentAuthSummary):
                return summary
            if isinstance(summary, dict):
                return DocumentAuthSummary.model_validate(summary)
            raise HTTPException(
                status_code=500,
                detail="Analiz xidm?tind?n g?zl?nilm?y?n n?tic? qaytar?ld?.",
            )
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"S?n?d analizi zaman? x?ta ba? verdi: {exc}",
            ) from exc
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass


    idempotency_map: Dict[str, str] = {}

    @router.post("/jobs")
    async def create_analysis_job(
        file: UploadFile = File(...),
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if idempotency_key and idempotency_key in idempotency_map:
            existing_id = idempotency_map[idempotency_key]
            return {"job_id": existing_id, "idempotent_replay": True, "status": "deduplicated"}

        if not file.filename:
            raise HTTPException(status_code=400, detail="Fayl ad? t?l?b olunur.")

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"D?st?kl?nm?y?n fayl format?: {ext or 'nam?lum'}.",
            )

        temp_dir = Path("tmp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"job_{uuid.uuid4().hex}{ext}"
        temp_path = temp_dir / unique_name
        total_bytes = 0

        try:
            with temp_path.open("wb") as out_file:
                while True:
                    chunk = await file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Fayl ?l??s? icaz? veril?n h?ddi ke?ir.",
                        )
                    out_file.write(chunk)

            if total_bytes == 0:
                raise HTTPException(status_code=400, detail="Y?kl?n?n fayl bo?dur.")

            summary = service.analyze_summary(temp_path)
            job_id = uuid.uuid4().hex
            result_payload = summary.model_dump() if hasattr(summary, "model_dump") else dict(summary)
            if idempotency_key:
                idempotency_map[idempotency_key] = job_id
            idempotency_map[f"__job__:{job_id}"] = result_payload  # type: ignore[assignment]
            return {"job_id": job_id, "status": "completed", "result": result_payload}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Job icra x?tas?: {exc}") from exc
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    @router.get("/jobs/{job_id}")
    async def get_analysis_job(job_id: str) -> Dict[str, Any]:
        stored = idempotency_map.get(f"__job__:{job_id}")
        if stored is None:
            raise HTTPException(status_code=404, detail="Job tap?lmad?.")
        return {"job_id": job_id, "status": "completed", "result": stored}

    return router


def create_app(
    analysis_service: Optional[Any] = None,
    deepfake_runner: Optional[Any] = None,
) -> FastAPI:
    app = FastAPI(title="Document & Deepfake Authentication API", version="1.0.0")

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    app.include_router(create_document_router(analysis_service=analysis_service))
    attach_deepfake_router(app, pipeline_runner=deepfake_runner)
    return app


app = create_app()
