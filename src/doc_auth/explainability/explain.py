from typing import Any, Dict, Optional


class Explainer:
    """Qərar və siqnallar əsasında izahlı xülasə formalaşdırır."""

    def explain(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = payload or {}
        decision = data.get("decision") if isinstance(data.get("decision"), dict) else {}
        signals = data.get("signals") if isinstance(data.get("signals"), dict) else data

        summary = decision.get("summary") or decision.get("reason")
        if not summary:
            status = decision.get("status", "evaluated")
            summary = f"Analiz yekunlaşdı (status={status})."

        return {
            "status": "success",
            "summary": str(summary),
            "signals_count": len(signals) if isinstance(signals, dict) else 0,
        }

    def analyze(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.explain(payload)

    def __call__(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.explain(payload)