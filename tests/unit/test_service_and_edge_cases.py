from pathlib import Path

import cv2
import numpy as np
import pytest

from doc_auth.deepfake.risk_scoring import DeepfakeRiskScorer
from doc_auth.service import AnalysisService


def test_analysis_service_end_to_end_image(tmp_path: Path) -> None:
    doc_img = tmp_path / "sample_doc.png"
    canvas = np.ones((320, 240, 3), dtype=np.uint8) * 255
    cv2.rectangle(canvas, (30, 40), (210, 280), (20, 20, 20), 2)
    cv2.putText(canvas, "DOC AUTH", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(doc_img), canvas)

    service = AnalysisService()
    result = service.analyze(doc_img)
    assert result["document_path"] == str(doc_img)
    assert "decision" in result
    assert "explanation" in result

    summary = service.analyze_summary(doc_img)
    assert summary.request_id == "sample_doc.png"
    assert len(summary.checks) == 6


def test_analysis_service_missing_file() -> None:
    service = AnalysisService()
    with pytest.raises(FileNotFoundError):
        service.analyze("nonexistent_document_12345.png")


def test_inconclusive_risk_when_visual_missing() -> None:
    scorer = DeepfakeRiskScorer()
    res = scorer.score({
        "visual": {"status": "no_faces", "score": None},
        "temporal": {"status": "success", "temporal_score": 0.2},
    })
    assert res["status"] == "inconclusive"
    assert res["risk_score"] is None
    assert res["overall_risk_score"] is None
    assert res["is_suspicious"] is None