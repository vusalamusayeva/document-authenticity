from fastapi import APIRouter, UploadFile, File, HTTPException, status
from doc_auth.deepfake.pipeline import analyze_video_deepfake

router = APIRouter(prefix="/api/v1/deepfake", tags=["Deepfake Analysis"])

MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB

@router.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    """
    Video faylını qəbul edir, ölçü limitini yoxlayır və deepfake analizi icra edir.
    Daxili filesystem yolları response-a çıxarılmır.
    """
    # Fayl ölçüsü yoxlaması
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Fayl ölçüsü icazə verilən 500MB limitini keçir."
        )

    try:
        result = analyze_video_deepfake(contents)
        if result.get("status") == "failed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Analiz zamanı xəta baş verdi.")
            )
        return result
    except HTTPException as he:
        raise he
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gözlənilməz daxili xəta baş verdi."
        )