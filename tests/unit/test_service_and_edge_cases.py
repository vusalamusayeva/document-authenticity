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

def test_document_classifier_save_load_and_auto_service(tmp_path: Path) -> None:
    from doc_auth.classification.document_classifier import DocumentClassifier

    base = np.ones((160, 240, 3), dtype=np.uint8) * 240
    cv2.rectangle(base, (20, 20), (220, 140), (30, 30, 30), 2)
    cv2.putText(base, "SAMPLE DOC", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

    clf = DocumentClassifier()
    clf.fit(
        [
            base,
            cv2.convertScaleAbs(base, alpha=1.1, beta=8),
            base[15:145, 20:220],
            cv2.flip(base, 1),
        ],
        ["full_scan", "full_scan", "cropped_scan", "mirrored_scan"],
    )
    pkl_path = tmp_path / "classifier.pkl"
    clf.save(str(pkl_path))
    assert pkl_path.exists()

    loaded = DocumentClassifier.load(str(pkl_path))
    assert getattr(loaded, "_is_fitted", False) is True

    default_model = Path("models/document_classifier.pkl")
    if default_model.exists():
        svc = AnalysisService()
        assert getattr(svc.classifier, "_is_fitted", False) is True
from pathlib import Path
import pytest

from doc_auth.service import AnalysisService


def test_service_init_missing_model_and_classifier_edge_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing_model = tmp_path / "nonexistent_classifier.pkl"
    monkeypatch.setenv("DOC_AUTH_CLASSIFIER_PATH", str(missing_model))

    svc = AnalysisService()
    assert svc.classifier is not None

    class _ListClassifier:
        def predict(self, _payload: object) -> list[str]:
            return ["invoice"]

    svc._run_classifier.__self__.classifier = _ListClassifier()
    res_list = svc._run_classifier({}, {}, document_path=tmp_path / "doc.pdf")
    assert res_list == {"status": "success", "predictions": ["invoice"]}

    class _NoMethodClassifier:
        pass

    svc.classifier = _NoMethodClassifier()
    res_unavail = svc._run_classifier({}, {}, document_path=tmp_path / "doc.pdf")
    assert res_unavail["status"] == "unavailable"
