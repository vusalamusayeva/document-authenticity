from typing import Any


class DecisionModel:
    """
    Ingestion, metadata, OCR, classification v? forensics siqnallar?n? birl??dir?r?k
    yekun s?n?d autentifikasiya q?rar?, risk bal? v? ??ki ?sasland?rmas? ??xar?r.
    """

    SCHEMA_VERSION = "1.1.0"

    def __init__(self, suspicious_threshold: float = 0.6) -> None:
        if not (0.0 <= suspicious_threshold <= 1.0):
            raise ValueError("suspicious_threshold [0.0, 1.0] aral???nda olmal?d?r.")
        self.suspicious_threshold = suspicious_threshold

    def decide(self, signals: dict[str, Any] | None = None) -> dict[str, Any]:
        data: dict[str, Any] = signals if isinstance(signals, dict) else {}

        raw_forensics = data.get("forensics")
        forensics: dict[str, Any] = raw_forensics if isinstance(raw_forensics, dict) else {}

        raw_metadata = data.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}

        raw_ocr = data.get("ocr")
        ocr: dict[str, Any] = raw_ocr if isinstance(raw_ocr, dict) else {}

        raw_classification = data.get("classification")
        classification: dict[str, Any] = (
            raw_classification if isinstance(raw_classification, dict) else {}
        )

        risk_points = 0.0
        weight_sum = 0.0
        contributions: list[dict[str, Any]] = []

        tampering_score = forensics.get("tampering_score")
        if isinstance(tampering_score, (int, float)) and not isinstance(tampering_score, bool):
            val = float(max(0.0, min(1.0, tampering_score)))
            w = 0.50
            risk_points += w * val
            weight_sum += w
            contributions.append({"signal": "forensics.tampering_score", "weight": w, "value": val, "weighted_risk": round(w * val, 4)})
        elif forensics.get("tampered") is True:
            w = 0.50
            risk_points += w
            weight_sum += w
            contributions.append({"signal": "forensics.tampered", "weight": w, "value": 1.0, "weighted_risk": w})

        if metadata.get("suspicious") is True:
            w = 0.20
            risk_points += w
            weight_sum += w
            contributions.append({"signal": "metadata.suspicious", "weight": w, "value": 1.0, "weighted_risk": w})
        elif metadata.get("status") == "success":
            w = 0.20
            weight_sum += w
            contributions.append({"signal": "metadata.suspicious", "weight": w, "value": 0.0, "weighted_risk": 0.0})

        if ocr.get("suspicious") is True:
            w = 0.15
            risk_points += w
            weight_sum += w
            contributions.append({"signal": "ocr.suspicious", "weight": w, "value": 1.0, "weighted_risk": w})
        elif ocr.get("status") == "success":
            w = 0.15
            weight_sum += w
            contributions.append({"signal": "ocr.suspicious", "weight": w, "value": 0.0, "weighted_risk": 0.0})

        cls_status = classification.get("status")
        if cls_status == "inconclusive":
            w = 0.15
            risk_points += w * 0.5
            weight_sum += w
            contributions.append({"signal": "classification.inconclusive", "weight": w, "value": 0.5, "weighted_risk": round(w * 0.5, 4)})
        elif cls_status == "success":
            w = 0.15
            is_unknown = 1.0 if classification.get("document_type") == "unknown" else 0.0
            risk_points += w * is_unknown
            weight_sum += w
            contributions.append({"signal": "classification.document_type", "weight": w, "value": is_unknown, "weighted_risk": round(w * is_unknown, 4)})

        if weight_sum <= 0.0:
            return {
                "schema_version": self.SCHEMA_VERSION,
                "status": "inconclusive",
                "decision": "inconclusive",
                "is_authentic": None,
                "is_suspicious": None,
                "risk_score": None,
                "contributions": [],
                "reason": "Q?rar verm?k ???n kifay?t q?d?r etibarl? siqnal yoxdur.",
            }

        risk_score = round(max(0.0, min(1.0, risk_points / weight_sum)), 4)
        is_suspicious = bool(risk_score >= self.suspicious_threshold)
        decision_label = "suspicious" if is_suspicious else "authentic"

        return {
            "schema_version": self.SCHEMA_VERSION,
            "status": "success",
            "decision": decision_label,
            "is_authentic": not is_suspicious,
            "is_suspicious": is_suspicious,
            "risk_score": risk_score,
            "threshold": self.suspicious_threshold,
            "contributions": contributions,
        }

    def fuse(self, signals: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.decide(signals)

    def predict(self, signals: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.decide(signals)

    def analyze(self, signals: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.decide(signals)

    def __call__(self, signals: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.decide(signals)
