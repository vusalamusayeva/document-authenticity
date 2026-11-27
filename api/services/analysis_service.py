"""Inference pipeline-? ?a??ran servis qat?."""

from __future__ import annotations

import base64
import binascii
import tempfile
import uuid
from pathlib import Path
from typing import Any

from api.schemas.document import DocumentAnalysisRequest, DocumentAnalysisResponse, RegionScore
from doc_auth.service import AnalysisService as CoreDocumentAnalysisService


def _decode_base64_image(raw_b64: str) -> bytes:
    payload = raw_b64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Yanl?? base64 ??kli g?nd?rilib.") from exc
    if not decoded:
        raise ValueError("Bo? base64 ??kli g?nd?rilib.")
    return decoded


def _map_label(raw_status: str | None, overall_score: float | None) -> str:
    status = (raw_status or "").lower()
    if status in {"forged", "fake", "tampered", "high_risk"}:
        return "forged"
    if status in {"suspicious", "review", "medium_risk", "needs_review"}:
        return "suspicious"
    if overall_score is not None:
        if overall_score >= 0.75:
            return "forged"
        if overall_score >= 0.40:
            return "suspicious"
    return "authentic"


class AnalysisService:
    def __init__(self, pipeline: Any = None, core_service: CoreDocumentAnalysisService | None = None):
        self.pipeline = pipeline
        self.core_service = core_service or CoreDocumentAnalysisService()

    def analyze(self, request: DocumentAnalysisRequest) -> DocumentAnalysisResponse:
        request_id = str(uuid.uuid4())
        image_bytes = _decode_base64_image(request.image_base64)

        suffix = ".png"
        if image_bytes[:2] == b"\xff\xd8":
            suffix = ".jpg"
        elif image_bytes[:4] == b"%PDF":
            suffix = ".pdf"

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_bytes)
                tmp_path = Path(tmp.name)

            if self.pipeline is not None and hasattr(self.pipeline, "run"):
                raw_result = self.pipeline.run(str(tmp_path))
            else:
                raw_result = self.core_service.analyze(str(tmp_path))
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        data = raw_result if isinstance(raw_result, dict) else {}
        overall_score = data.get("overall_risk_score", data.get("risk_score", 0.0))
        try:
            score_val = float(overall_score) if overall_score is not None else 0.0
        except (TypeError, ValueError):
            score_val = 0.0

        label = _map_label(
            str(data.get("status") or data.get("verdict") or ""),
            score_val,
        )
        confidence = max(0.0, min(1.0, round(1.0 - score_val if label == "authentic" else score_val, 4)))

        doc_type = request.document_type_hint
        classification = data.get("classification")
        if isinstance(classification, dict):
            doc_type = doc_type or classification.get("document_type") or classification.get("label")

        regions: list[RegionScore] = []
        raw_regions = data.get("suspicious_regions") or []
        if isinstance(raw_regions, list):
            for item in raw_regions:
                if isinstance(item, dict) and isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4:
                    regions.append(
                        RegionScore(
                            bbox=[float(x) for x in item["bbox"]],
                            tamper_type=str(item.get("tamper_type", "anomaly")),
                            score=float(item.get("score", score_val)),
                        )
                    )

        return DocumentAnalysisResponse(
            label=label,
            confidence=confidence,
            document_type=doc_type,
            suspicious_regions=regions,
            explanation_mask_base64=None,
            request_id=request_id,
        )
