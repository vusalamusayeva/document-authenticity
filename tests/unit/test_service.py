from pathlib import Path
import types

import numpy as np
import pytest

from doc_auth import service as svc_mod
from doc_auth.schemas.pipeline_schemas import DocumentAuthSummary


def test_helpers_and_default_explainer(monkeypatch: pytest.MonkeyPatch) -> None:
    arr = np.zeros((2, 3), dtype=np.uint8)
    sanitized = svc_mod._sanitize_for_json(
        {
            "arr": arr,
            "scalar": np.float32(0.5),
            "path": Path("a/b.txt"),
            "items": [np.int64(2), ("x", 1)],
        }
    )
    assert sanitized["arr"] == {"ndarray_shape": [2, 3], "ndarray_dtype": "uint8"}
    assert sanitized["scalar"] == pytest.approx(0.5, abs=1e-5)
    assert sanitized["path"] == str(Path("a/b.txt"))
    assert sanitized["items"] == [2, ["x", 1]]

    # _default_explainer fallback to public callable when Explainer absent
    fake_mod = types.SimpleNamespace(build_explainer=lambda: "custom-explainer")
    monkeypatch.setattr(svc_mod, "explain_module", fake_mod)
    assert svc_mod._default_explainer() == "custom-explainer"

    # _default_explainer raises RuntimeError when nothing callable exists
    empty_mod = types.SimpleNamespace(_private=lambda: None, constant=123)
    monkeypatch.setattr(svc_mod, "explain_module", empty_mod)
    with pytest.raises(RuntimeError):
        svc_mod._default_explainer()


def test_classifier_load_fallback_and_invoke_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_file = tmp_path / "doc_classifier.pkl"
    model_file.write_bytes(b"corrupt")
    monkeypatch.setenv("DOC_AUTH_CLASSIFIER_PATH", str(model_file))

    def _boom_load(_: str):
        raise ValueError("cannot unpickle")

    monkeypatch.setattr(svc_mod.DocumentClassifier, "load", staticmethod(_boom_load))
    svc = svc_mod.AnalysisService(explainer=lambda payload: {"summary": "ok"})
    assert isinstance(svc.classifier, svc_mod.DocumentClassifier)

    # Non-dict return wrapped in {"status": "success", "result": ...}
    comp_val = types.SimpleNamespace(run=lambda x: 42)
    assert svc_mod.AnalysisService._invoke_module(comp_val, ("run",), {"k": 1}) == {
        "status": "success",
        "result": 42,
    }

    # Callable component with dict return
    assert svc_mod.AnalysisService._invoke_module(lambda x: {"ok": True}, ("missing",), {}) == {
        "ok": True
    }

    # Callable component with non-dict return
    assert svc_mod.AnalysisService._invoke_module(lambda x: "done", ("missing",), {}) == {
        "status": "success",
        "result": "done",
    }

    # No matching method and not callable -> RuntimeError
    with pytest.raises(RuntimeError):
        svc_mod.AnalysisService._invoke_module(object(), ("run",), {})


def test_run_classifier_analyze_and_summary(tmp_path: Path) -> None:
    doc_file = tmp_path / "sample.png"
    doc_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)

    seen_payloads = []

    class _RecordingClassifier:
        def predict(self, payload):
            seen_payloads.append(payload)
            return {"status": "ok", "label": "invoice", "score": 0.2, "is_suspicious": False}

    loader = types.SimpleNamespace(
        load=lambda p: {"pages": [{"image": np.ones((4, 4, 3), dtype=np.uint8)}]}
    )
    detector = types.SimpleNamespace(detect=lambda d: {"detected": True, "area_ratio": 0.9})
    meta = types.SimpleNamespace(analyze=lambda p: {"author": "test", "status": "ok"})
    ocr = types.SimpleNamespace(check=lambda d: {"text_density": 0.4, "char_count": 120, "status": "ok"})
    tamper = types.SimpleNamespace(detect=lambda d: {"tampered": False, "score": 0.05, "status": "ok"})
    decision = types.SimpleNamespace(
        decide=lambda f: {"status": "completed", "decision": "authentic", "risk_score": 0.1}
    )
    explainer = types.SimpleNamespace(explain=lambda ctx: {"reasons": ["clean"]})

    svc = svc_mod.AnalysisService(
        loader=loader,
        detector=detector,
        metadata_analyzer=meta,
        ocr_checker=ocr,
        classifier=_RecordingClassifier(),
        tampering_detector=tamper,
        decision_model=decision,
        explainer=explainer,
    )

    summary = svc.analyze_summary(doc_file)
    assert isinstance(summary, DocumentAuthSummary)
    assert summary.decision == "authentic"
    assert summary.overall_risk_score == pytest.approx(0.1)
    assert len(checks := summary.checks) == 6
    assert isinstance(seen_payloads[0], np.ndarray)
    assert seen_payloads[0].shape == (4, 4, 3)

    # Classifier raising an exception returns status="unavailable"
    class _FailingClassifier:
        def predict(self, payload):
            raise RuntimeError("model offline")

    svc_fail = svc_mod.AnalysisService(
        loader=loader,
        detector=detector,
        metadata_analyzer=meta,
        ocr_checker=ocr,
        classifier=_FailingClassifier(),
        tampering_detector=tamper,
        decision_model=decision,
        explainer=explainer,
    )
    res_fail = svc_fail.analyze(doc_file)
    assert res_fail["classification"]["status"] == "unavailable"
    assert "model offline" in res_fail["classification"]["error"]

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        svc.analyze(tmp_path / "does_not_exist.png")
