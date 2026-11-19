from typing import Any, Dict, Optional


class DecisionModel:
    """
    Ingestion, metadata, OCR, classification və forensics siqnallarını birləşdirərək
    yekun sənəd autentifikasiya qərarı və risk balı çıxarır.
    """

    def __init__(self, suspicious_threshold: float = 0.6) -> None:
        if not (0.0 <= suspicious_threshold <= 1.0):
            raise ValueError("suspicious_threshold [0.0, 1.0] aralığında olmalıdır.")
        self.suspicious_threshold = suspicious_threshold

    def decide(self, signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = signals or {}
        forensics = data.get("forensics") if isinstance(data.get("forensics"), dict) else {}
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        ocr = data.get("ocr") if isinstance(data.get("ocr"), dict) else {}

        risk_points = 0.0
        weight_sum = 0.0

        tampering_score = forensics.get("tampering_score")
        if isinstance(tampering_score, (int, float)) and not isinstance(tampering_score, bool):
            risk_points += 0.6 * float(tampering_score)
            weight_sum += 0.6
        elif forensics.get("tampered") is True:
            risk_points += 0.6
            weight_sum += 0.6

        if metadata.get("suspicious") is True:
            risk_points += 0.2
            weight_sum += 0.2
        elif metadata.get("status") == "success":
            weight_sum += 0.2

        if ocr.get("suspicious") is True:
            risk_points += 0.2
            weight_sum += 0.2
        elif ocr.get("status") == "success":
            weight_sum += 0.2

        if weight_sum <= 0.0:
            return {
                "status": "inconclusive",
                "is_authentic": None,
                "is_suspicious": None,
                "risk_score": None,
                "reason": "Qərar vermək üçün kifayət qədər etibarlı siqnal yoxdur.",
            }

        risk_score = round(max(0.0, min(1.0, risk_points / weight_sum)), 4)
        is_suspicious = bool(risk_score >= self.suspicious_threshold)

        return {
            "status": "success",
            "is_authentic": not is_suspicious,
            "is_suspicious": is_suspicious,
            "risk_score": risk_score,
            "threshold": self.suspicious_threshold,
        }

    def fuse(self, signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.decide(signals)

    def predict(self, signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.decide(signals)

    def analyze(self, signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.decide(signals)

    def __call__(self, signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.decide(signals)