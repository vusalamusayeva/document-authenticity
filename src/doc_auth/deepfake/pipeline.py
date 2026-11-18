import os
import tempfile
from typing import Dict, Any, Optional
from doc_auth.deepfake.video_loader import validate_video_metadata
from doc_auth.deepfake.frame_sampler import sample_frames
from doc_auth.deepfake.face_detector import detect_faces
from doc_auth.deepfake.visual_detector import analyze_visual_artifacts
from doc_auth.deepfake.temporal_analysis import analyze_temporal
from doc_auth.deepfake.audio_analysis import analyze_audio
from doc_auth.deepfake.risk_scoring import calculate_risk_score
from doc_auth.deepfake.exceptions import DocAuthError

def analyze_video_deepfake(video_bytes: bytes, config: Optional[Dict[str, Any]] = None, model_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """
    Video bayitlərini qəbul edir, müvəqqəti fayla yazır, bütün mərhələləri (loader, 
    sampler, face detection, visual, temporal, audio, risk scoring) birləşdirir 
    və finally blokunda müvəqqəti faylı təmizləyir.
    """
    config = config or {}
    temp_file_path = None

    try:
        # Müvəqqəti faylın yaradılması
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            temp_file_path = tmp.name

        # 1. Video metadata və loader yoxlaması
        metadata = validate_video_metadata(temp_file_path, config)

        # 2. Frame sampling
        sample_rate = config.get("sample_rate", 5)
        max_frames = config.get("max_frames", 20)
        
        frames_predictions = []
        frame_iter = sample_frames(temp_file_path, sample_rate=sample_rate, max_frames=max_frames)

        for item in frame_iter:
            frame = item["frame"]
            # 3. Face detection
            faces = detect_faces(frame)
            for face in faces:
                # 4. Visual analysis
                visual_res = analyze_visual_artifacts(face["crop"], model_checkpoint=model_checkpoint)
                frames_predictions.append({
                    "frame_index": item["frame_index"],
                    "timestamp_ms": item["timestamp_ms"],
                    "score": visual_res.get("score"),
                    "is_deepfake": visual_res.get("is_deepfake")
                })

        # 5. Temporal analysis
        temporal_res = analyze_temporal(frames_predictions)

        # 6. Audio analysis
        audio_res = analyze_audio(temp_file_path)

        # Analiz nəticələrinin yığılması
        analyses = {
            "visual": {"status": "success", "score": frames_predictions[-1]["score"] if frames_predictions else 0.0},
            "temporal": temporal_res,
            "audio": audio_res
        }

        # 7. Risk scoring
        risk_result = calculate_risk_score(analyses, config.get("weights"))

        return {
            "status": "success",
            "metadata": metadata,
            "deepfake_result": {
                "is_deepfake": risk_result.get("decision") == "fake",
                "risk_score": risk_result.get("risk_score"),
                "decision": risk_result.get("decision"),
                "frames_analyzed": len(frames_predictions)
            }
        }

    except DocAuthError as e:
        return {
            "status": "failed",
            "error_type": type(e).__name__,
            "message": str(e)
        }
    except Exception as e:
        return {
            "status": "failed",
            "error_type": "InternalError",
            "message": "Daxili sistem xətası baş verdi."
        }
    finally:
        # Müvəqqəti faylın həmişə təmizlənməsi (cleanup)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass  # Cleanup xətası əsas nəticəni əvəz etmir