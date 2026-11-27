import math
from typing import Any

from doc_auth.deepfake.exceptions import InvalidConfigError


def validate_score_value(value: Any, name: str = "score") -> float:
    """Skor dəyərinin `[0.0, 1.0]` aralığında etibarlı rəqəm olduğunu yoxlayır."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidConfigError(f"{name} rəqəm olmalıdır, alındı: {type(value).__name__}")
    score = float(value)
    if math.isnan(score) or math.isinf(score) or not (0.0 <= score <= 1.0):
        raise InvalidConfigError(f"{name} [0.0, 1.0] aralığında olmalıdır, alındı: {value}")
    return score


def _validate_weight(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidConfigError(f"{name} rəqəm olmalıdır, alındı: {type(value).__name__}")
    w = float(value)
    if math.isnan(w) or math.isinf(w) or w < 0.0:
        raise InvalidConfigError(f"{name} mənfi olmamalıdır, alındı: {value}")
    return w


class DeepfakeRiskScorer:
    """
    Visual, temporal və audio/lip-sync siqnallarından yekun deepfake risk balı və
    qərarı hesablayır. Siqnallar natamam olduqda false positive/negative əvəzinə
    `inconclusive` qaytarır.
    """

    def __init__(
        self,
        visual_weight: float = 0.65,
        temporal_weight: float = 0.20,
        audio_weight: float = 0.15,
        suspicious_threshold: float = 0.50,
    ) -> None:
        vw = _validate_weight(visual_weight, "visual_weight")
        tw = _validate_weight(temporal_weight, "temporal_weight")
        aw = _validate_weight(audio_weight, "audio_weight")
        total_w = vw + tw + aw
        if total_w <= 0.0:
            raise InvalidConfigError("Çəkilərin cəmi müsbət olmalıdır.")
        validate_score_value(suspicious_threshold, name="suspicious_threshold")

        self.visual_weight = vw
        self.temporal_weight = tw
        self.audio_weight = aw
        self.suspicious_threshold = float(suspicious_threshold)

    def score(self, signals: dict[str, Any] | None = None) -> dict[str, Any]:
        data = signals or {}
        raw_visual = data.get("visual")
        visual: dict[str, Any] = raw_visual if isinstance(raw_visual, dict) else {}

        raw_temporal = data.get("temporal")
        temporal: dict[str, Any] = raw_temporal if isinstance(raw_temporal, dict) else {}

        raw_audio = data.get("audio")
        audio: dict[str, Any] = raw_audio if isinstance(raw_audio, dict) else {}

        weighted_sum = 0.0
        active_weight = 0.0
        used_signals = []

        v_score = visual.get("score")
        if visual.get("status") == "success" and v_score is not None:
            validated_v = validate_score_value(v_score, name="visual_score")
            weighted_sum += self.visual_weight * validated_v
            active_weight += self.visual_weight
            used_signals.append("visual")

        t_score = temporal.get("temporal_score")
        if t_score is None:
            t_score = temporal.get("temporal_risk")
        if temporal.get("status") == "success" and t_score is not None:
            validated_t = validate_score_value(t_score, name="temporal_score")
            weighted_sum += self.temporal_weight * validated_t
            active_weight += self.temporal_weight
            used_signals.append("temporal")

        a_score = audio.get("lip_sync_score")
        if audio.get("status") == "success" and a_score is not None:
            validated_a = validate_score_value(a_score, name="lip_sync_score")
            weighted_sum += self.audio_weight * validated_a
            active_weight += self.audio_weight
            used_signals.append("audio")

        if active_weight <= 0.0 or "visual" not in used_signals:
            return {
                "status": "inconclusive",
                "decision": "inconclusive",
                "risk_score": None,
                "overall_risk_score": None,
                "is_suspicious": None,
                "used_signals": used_signals,
            }

        risk_score = round(min(1.0, max(0.0, weighted_sum / active_weight)), 4)
        is_suspicious = bool(risk_score >= self.suspicious_threshold)

        return {
            "status": "success",
            "decision": "suspicious" if is_suspicious else "low_risk",
            "risk_score": risk_score,
            "overall_risk_score": risk_score,
            "is_suspicious": is_suspicious,
            "threshold": self.suspicious_threshold,
            "used_signals": used_signals,
        }

    def compute_risk(
        self,
        max_visual_score: float | None = None,
        temporal_score: float | None = None,
        audio_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        visual_payload: dict[str, Any] = (
            {"status": "success", "score": max_visual_score}
            if max_visual_score is not None
            else {"status": "unavailable", "score": None}
        )
        temporal_payload: dict[str, Any] = (
            {"status": "success", "temporal_score": temporal_score}
            if temporal_score is not None
            else {"status": "unavailable", "temporal_score": None}
        )

        audio_payload: dict[str, Any] = {"status": "unavailable", "lip_sync_score": None}
        if isinstance(audio_analysis, dict):
            if "lip_sync_score" in audio_analysis and audio_analysis["lip_sync_score"] is not None:
                audio_payload = {"status": "success", "lip_sync_score": audio_analysis["lip_sync_score"]}
            elif audio_analysis.get("has_audio") is True:
                status_val = str(audio_analysis.get("audio_status", "")).lower()
                inferred_score = 0.5 if status_val in {"suspicious", "mismatch", "anomalous"} else 0.1
                audio_payload = {"status": "success", "lip_sync_score": inferred_score}

        return self.score({
            "visual": visual_payload,
            "temporal": temporal_payload,
            "audio": audio_payload,
        })

    def __call__(self, signals: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.score(signals)


RiskScorer = DeepfakeRiskScorer


def calculate_risk_score(
    signals: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    visual_weight: float = 0.65,
    temporal_weight: float = 0.20,
    audio_weight: float = 0.15,
    suspicious_threshold: float = 0.50,
) -> dict[str, Any]:
    if isinstance(weights, dict):
        vw = weights.get("visual", weights.get("visual_weight", 0.0))
        tw = weights.get("temporal", weights.get("temporal_weight", 0.0))
        aw = weights.get("audio", weights.get("audio_weight", 0.0))
    else:
        vw = visual_weight
        tw = temporal_weight
        aw = audio_weight

    scorer = DeepfakeRiskScorer(
        visual_weight=vw,
        temporal_weight=tw,
        audio_weight=aw,
        suspicious_threshold=suspicious_threshold,
    )
    return scorer.score(signals)