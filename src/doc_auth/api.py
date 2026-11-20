from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile

from doc_auth.deepfake.api import attach_deepfake_router
from doc_auth.schemas.pipeline_schemas import DocumentAuthSummary
from doc_auth.service import AnalysisService

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


def create_document_router(
    analysis_service: Optional[Any] = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["documents"])
    service = analysis_service if analysis_service is not None else AnalysisService()

    @router.post("/analyze", response_model=DocumentAuthSummary)
    async def analyze_document_endpoint(file: UploadFile = File(...)) -> DocumentAuthSummary:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Fayl adı tələb olunur.")

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Dəstəklənməyən fayl formatı: {ext or 'naməlum'}.",
            )

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Yüklənən fayl boşdur.")

        if len(content) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail="Fayl ölçüsü icazə verilən həddi keçir.",
            )

        temp_dir = Path("tmp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"upload_{Path(file.filename).name}"
        temp_path = temp_dir / safe_name

        try:
            temp_path.write_bytes(content)
            summary = service.analyze_summary(temp_path)
            if isinstance(summary, DocumentAuthSummary):
                return summary
            if isinstance(summary, dict):
                return DocumentAuthSummary.model_validate(summary)
            raise HTTPException(
                status_code=500,
                detail="Analiz xidmətindən gözlənilməyən nəticə qaytarıldı.",
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
                detail=f"Sənəd analizi zamanı xəta baş verdi: {exc}",
            ) from exc
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

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