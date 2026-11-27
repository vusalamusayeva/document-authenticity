import pytest
from pathlib import Path
import importlib
import inspect
import numpy as np


def test_deepfake_visual_detector_import_and_callable() -> None:
    mod = importlib.import_module("doc_auth.deepfake.visual_detector")
    assert mod is not None

    frame = np.ones((224, 224, 3), dtype=np.uint8) * 128

    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if isinstance(obj, type) and obj.__module__ == mod.__name__:
            try:
                inst = obj()
            except TypeError:
                continue
            for m_name in ("predict", "detect", "analyze", "score", "forward", "run"):
                fn = getattr(inst, m_name, None)
                if callable(fn):
                    sig = inspect.signature(fn)
                    req = [
                        p for p in sig.parameters.values()
                        if p.default is inspect.Parameter.empty
                        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                    ]
                    if len(req) == 1:
                        try:
                            out = fn(frame)
                        except Exception:
                            out = fn([frame])
                        assert out is not None
                        return

def test_load_model_raises_when_torch_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import doc_auth.deepfake.visual_detector as vd_mod
    from doc_auth.deepfake.exceptions import ModelUnavailableError

    ckpt = tmp_path / "dummy_weights.pt"
    ckpt.write_bytes(b"placeholder")

    monkeypatch.setattr(vd_mod, "torch", None)
    detector = vd_mod.VisualDetector(checkpoint_path=str(ckpt))

    with pytest.raises(ModelUnavailableError):
        detector._load_model()
