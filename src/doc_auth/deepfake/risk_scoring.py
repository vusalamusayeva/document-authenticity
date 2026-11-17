from typing import Dict, Any

class RiskScorer:
    def __init__(self, visual_weight: float = 0.5, temporal_weight: float = 0.3, audio_weight: float = 0.2):
        total = visual_weight + temporal_weight + audio_weight
        self.w_visual = visual_weight / total
        self.w_temporal = temporal_weight / total
        self.w_audio = audio_weight / total

    def compute_risk(self, max_visual_score: float, temporal_score: float, audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        audio_risk = 0.0
        if not audio_analysis.get("has_audio", True):
            audio_risk = 0.3
        elif audio_analysis.get("audio_status") == "audio_video_desync":
            audio_risk = 0.6

        raw_risk = (
            (max_visual_score * self.w_visual) +
            (temporal_score * self.w_temporal) +
            (audio_risk * self.w_audio)
        )

        overall_risk = float(min(max(raw_risk, 0.0), 1.0))
        is_suspicious = overall_risk > 0.5

        return {
            "overall_risk_score": round(overall_risk, 4),
            "is_suspicious": is_suspicious
        }