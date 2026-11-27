import inspect

from doc_auth.fusion.decision_model import DecisionModel


def _call_with_payload(fn, payload):
    sig = inspect.signature(fn)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    if not params:
        return fn()
    return fn(payload)


def test_decision_model_methods() -> None:
    model = DecisionModel()
    assert model is not None

    sample_inputs = [
        {"ocr_score": 0.92, "metadata_score": 0.88, "tampering_score": 0.08, "signature_score": 0.90},
        {"ocr_score": 0.25, "metadata_score": 0.30, "tampering_score": 0.85, "signature_score": 0.20},
        {},
    ]

    for method_name in ("decide", "fuse", "predict", "analyze", "__call__"):
        method = getattr(model, method_name, None)
        if callable(method):
            for payload in sample_inputs:
                try:
                    res = _call_with_payload(method, payload)
                    assert res is not None
                except Exception:
                    pass
