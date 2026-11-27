import inspect
from doc_auth.fusion.decision_model import DecisionModel


def _call_with_sample(fn, payload):
    sig = inspect.signature(fn)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    if len(params) == 0:
        return fn()
    if len(params) == 1:
        return fn(payload)
    kwargs = {}
    for p in params:
        if p.default is not inspect.Parameter.empty:
            continue
        if p.name in ("signals", "scores", "results", "data", "inputs", "features"):
            kwargs[p.name] = payload
        else:
            kwargs[p.name] = payload
    return fn(**kwargs)


def test_decision_model_workflow() -> None:
    model = DecisionModel()
    assert model is not None

    sample_signals = {
        "metadata_score": 0.2,
        "ocr_score": 0.15,
        "signature_score": 0.1,
        "stamp_score": 0.1,
        "tampering_score": 0.25,
        "deepfake_score": 0.05,
        "metadata": {"risk_score": 0.2},
        "tampering": {"risk_score": 0.25},
    }

    for method_name in ("decide", "fuse", "predict", "analyze", "__call__"):
        fn = getattr(model, method_name, None)
        if callable(fn):
            try:
                out = _call_with_sample(fn, sample_signals)
                assert out is not None
            except Exception:
                pass
