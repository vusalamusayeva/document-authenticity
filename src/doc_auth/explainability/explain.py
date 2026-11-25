from typing import Any, Dict, List, Optional


class Explainer:
    """Q?rar, ??kil?r v? siqnallar ?sas?nda strukturla?d?r?lm?? izah v? faktor siyah?s? qaytar?r."""

    def explain(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = payload or {}
        decision = data.get("decision") if isinstance(data.get("decision"), dict) else {}
        signals = data.get("signals") if isinstance(data.get("signals"), dict) else data

        contributions = decision.get("contributions") if isinstance(decision.get("contributions"), list) else []
        factors: List[Dict[str, Any]] = []
        for item in contributions:
            if isinstance(item, dict):
                factors.append({
                    "signal": item.get("signal"),
                    "weight": item.get("weight"),
                    "value": item.get("value"),
                    "weighted_risk": item.get("weighted_risk"),
                })

        evidence_highlights: List[str] = []
        forensics = signals.get("forensics") if isinstance(signals.get("forensics"), dict) else {}
        metadata = signals.get("metadata") if isinstance(signals.get("metadata"), dict) else {}
        ocr = signals.get("ocr") if isinstance(signals.get("ocr"), dict) else {}
        classification = signals.get("classification") if isinstance(signals.get("classification"), dict) else {}

        if forensics.get("tampering_score") is not None:
            evidence_highlights.append(
                f"Forensics ELA skoru: {forensics.get('tampering_score')} (tampered={forensics.get('tampered')})"
            )

        for sig in metadata.get("suspicious_signals") or []:
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
            "signals_count": len(signals) if isinstance(signals, dict) else 0,
            "factors": factors,
            "evidence_highlights": evidence_highlights,
        }

    def analyze(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.explain(payload)

    def __call__(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.explain(payload)
