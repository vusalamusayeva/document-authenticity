from typing import Any


class Explainer:
    """Q?rar, ??kill?r v? siqnallar ?sas?nda strukturla?d?r?lm?? izah v? faktor siyah?s? qaytar?r."""

    def explain(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data: dict[str, Any] = payload if isinstance(payload, dict) else {}

        raw_decision = data.get("decision")
        decision: dict[str, Any] = raw_decision if isinstance(raw_decision, dict) else {}

        raw_signals = data.get("signals")
        signals: dict[str, Any] = raw_signals if isinstance(raw_signals, dict) else data

        raw_contributions = decision.get("contributions")
        contributions: list[Any] = raw_contributions if isinstance(raw_contributions, list) else []
        factors: list[dict[str, Any]] = []
        for item in contributions:
            if isinstance(item, dict):
                factors.append({
                    "signal": item.get("signal"),
                    "weight": item.get("weight"),
                    "value": item.get("value"),
                    "weighted_risk": item.get("weighted_risk"),
                })

        evidence_highlights: list[str] = []

        raw_forensics = signals.get("forensics")
        forensics: dict[str, Any] = raw_forensics if isinstance(raw_forensics, dict) else {}

        raw_metadata = signals.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}

        raw_ocr = signals.get("ocr")
        ocr: dict[str, Any] = raw_ocr if isinstance(raw_ocr, dict) else {}

        raw_classification = signals.get("classification")
        classification: dict[str, Any] = (
            raw_classification if isinstance(raw_classification, dict) else {}
        )

        if forensics.get("tampering_score") is not None:
            evidence_highlights.append(
                f"Forensics ELA skoru: {forensics.get('tampering_score')} (tampered={forensics.get('tampered')})"
            )

        raw_suspicious = metadata.get("suspicious_signals")
        suspicious_signals: list[Any] = raw_suspicious if isinstance(raw_suspicious, list) else []
        for sig in suspicious_signals:
            if isinstance(sig, dict) and sig.get("reason"):
                evidence_highlights.append(f"Metadata [{sig.get('field')}]: {sig.get('reason')}")

        if ocr.get("status"):
            evidence_highlights.append(f"OCR status: {ocr.get('status')} (suspicious={ocr.get('suspicious')})")

        if classification.get("status"):
            evidence_highlights.append(
                f"Classification: status={classification.get('status')}, type={classification.get('document_type')}"
            )

        decision_label = decision.get("decision") or decision.get("status") or "evaluated"
        risk_score = decision.get("risk_score")
        base_reason = decision.get("summary") or decision.get("reason")
        if base_reason:
            summary = str(base_reason)
        elif risk_score is not None:
            summary = f"Yekun q?rar: {decision_label} (risk_score={risk_score})."
        else:
            summary = f"Yekun q?rar: {decision_label}."

        return {
            "status": "success",
            "decision": decision_label,
            "risk_score": risk_score,
            "summary": summary,
            "signals_count": len(signals),
            "factors": factors,
            "evidence_highlights": evidence_highlights,
        }

    def analyze(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.explain(payload)

    def __call__(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.explain(payload)
