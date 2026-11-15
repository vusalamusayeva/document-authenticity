from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import tempfile
import os
from src.doc_auth.deepfake.pipeline import DeepfakePipeline
from src.doc_auth.deepfake.schemas import DeepfakeConfig, DeepfakeAnalysisResult

app = FastAPI(
    title="Deepfake Video Analysis Engine API",
    description="Sənəd və video forensika üçün deepfake aşkarlama xidməti",
    version="1.0.0"
)

pipeline = DeepfakePipeline()

@app.post("/api/v1/deepfake/analyze", response_model=DeepfakeAnalysisResult)
async def analyze_video_endpoint(file: UploadFile = File(...)):
    """
    Yüklənən video faylını qəbul edir, müvəqqəti saxlamaqla DeepfakePipeline üzərindən 
    analiz edir və nəticəni (risk skorları, temporal uyğunsuzluq, audio statusu) qaytarır.
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Dəstəklənməyən video formatı.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    try:
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()

        result = pipeline.run(temp_file.name)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz zamanı xəta baş verdi: {str(e)}")
    finally:
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception:
                pass