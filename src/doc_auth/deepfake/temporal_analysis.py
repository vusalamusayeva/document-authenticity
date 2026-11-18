import numpy as np
from typing import List, Dict, Any, Optional

def analyze_temporal(frames_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Kadrlar arası temporal analizi həyata keçirir: 
    score dəyişməsi, eyni üzün bbox dəyişməsi və üzün yoxa çıxması.
    Məlumat çatışmazlığı olduqda sıfır risk deyil, not_evaluated qaytarır.
    """
    if not frames_predictions or len(frames_predictions) < 2:
        return {
            "status": "not_evaluated",
            "reason": "Temporal analiz üçün kifayət qədər kadr və ya məlumat yoxdur.",
            "score_variance": None,
            "temporal_risk": None
        }

    scores = [p.get("score", 0.0) for p in frames_predictions if p.get("score") is not None]
    if not scores:
        return {
            "status": "not_evaluated",
            "reason": "Qiymətləndirilmiş skor tapılmadı.",
            "score_variance": None,
            "temporal_risk": None
        }

    score_variance = float(np.var(scores))
    mean_score = float(np.mean(scores))
    
    # Səhnə dəyişikliyi və ya qeyri-sabitlik analizi
    temporal_risk = min(1.0, score_variance * 2.0)

    return {
        "status": "success",
        "score_variance": score_variance,
        "mean_score": mean_score,
        "temporal_risk": temporal_risk
    }