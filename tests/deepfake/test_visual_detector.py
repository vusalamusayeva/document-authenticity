from pathlib import Path

import numpy as np
import pytest

from doc_auth.deepfake.exceptions import ModelUnavailableError
from doc_auth.deepfake.visual_detector import VisualDetector


def test_visual_detector_loads_torch_checkpoint_and_infers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    import doc_auth.deepfake.visual_detector as vd_mod

    fake_nn = types.SimpleNamespace(Module=type("_FakeModule", (), {}))
    fake_torch = types.SimpleNamespace(load=lambda *a, **kw: {"epoch": 1, "state_dict": {"weight": [0.1]}})
    monkeypatch.setattr(vd_mod, "nn", fake_nn, raising=False)
    monkeypatch.setattr(vd_mod, "torch", fake_torch, raising=False)

    ckpt_path = tmp_path / "dummy_model.pt"
    ckpt_path.write_bytes(b"pt-checkpoint")

    detector = VisualDetector(checkpoint_path=ckpt_path, threshold=0.5)
    info = detector.load_model()
    assert info["status"] == "loaded"

    fake_face_img = np.random.default_rng(42).integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    faces = [{"bbox": [16, 16, 112, 112], "score": 0.99}]
    with pytest.raises(ModelUnavailableError):
        detector.analyze_frame(fake_face_img, faces=faces)


def test_visual_detector_rejects_unsupported_extension(tmp_path: Path) -> None:
    bad_ckpt = tmp_path / "weights.txt"
    bad_ckpt.write_text("not a checkpoint", encoding="utf-8")

    detector = VisualDetector(checkpoint_path=bad_ckpt)
    with pytest.raises(ModelUnavailableError):
        detector.verify_model_weights()


def test_visual_detector_rejects_unsupported_extension(tmp_path: Path) -> None:
    bad_ckpt = tmp_path / "weights.txt"
    bad_ckpt.write_text("not a checkpoint", encoding="utf-8")

    detector = VisualDetector(checkpoint_path=bad_ckpt)
    with pytest.raises(ModelUnavailableError):
        detector.verify_model_weights()


def test_visual_detector_torch_module_and_state_dict_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    import doc_auth.deepfake.visual_detector as vd_mod

    class _FakeModule:
        def __init__(self) -> None:
            self.moved_to = None
            self.eval_called = False

        def to(self, device: str):
            self.moved_to = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    fake_nn = types.SimpleNamespace(Module=_FakeModule)
    monkeypatch.setattr(vd_mod, "nn", fake_nn, raising=False)

    mod_ckpt = tmp_path / "fake_module.pt"
    mod_ckpt.write_bytes(b"pt-module")

    mod_instance = _FakeModule()
    fake_torch_mod = types.SimpleNamespace(load=lambda *a, **kw: mod_instance)
    monkeypatch.setattr(vd_mod, "torch", fake_torch_mod, raising=False)

    det_mod = VisualDetector(checkpoint_path=mod_ckpt, device="cpu")
    res_mod = det_mod.load_model()
    assert res_mod["backend"] == "torch_checkpoint"
    assert res_mod["status"] == "loaded"
    assert mod_instance.moved_to == "cpu"
    assert mod_instance.eval_called is True
    assert det_mod.load_model()["status"] == "loaded"

    patch = np.full((16, 16, 3), 128, dtype=np.uint8)
    with pytest.raises(ModelUnavailableError):
        det_mod._compute_region_score(patch)

    sd_ckpt = tmp_path / "fake_sd.pth"
    sd_ckpt.write_bytes(b"pt-sd")
    fake_torch_sd = types.SimpleNamespace(
        load=lambda *a, **kw: {"state_dict": {"module.conv.weight": [1.0], "bias": [0.0]}}
    )
    monkeypatch.setattr(vd_mod, "torch", fake_torch_sd, raising=False)
    det_sd = VisualDetector(checkpoint_path=sd_ckpt)
    res_sd = det_sd.load_model()
    assert res_sd["backend"] == "torch_checkpoint"
    assert "conv.weight" in det_sd._model
    assert "module.conv.weight" not in det_sd._model

    msd_ckpt = tmp_path / "fake_msd.pt"
    msd_ckpt.write_bytes(b"pt-msd")
    fake_torch_msd = types.SimpleNamespace(
        load=lambda *a, **kw: {"model_state_dict": {"head.weight": [2.0]}}
    )
    monkeypatch.setattr(vd_mod, "torch", fake_torch_msd, raising=False)
    det_msd = VisualDetector(checkpoint_path=msd_ckpt)
    assert det_msd.load_model()["backend"] == "torch_checkpoint"
    assert "head.weight" in det_msd._model

    raw_ckpt = tmp_path / "fake_raw.pt"
    raw_ckpt.write_bytes(b"pt-raw")
    fake_torch_raw = types.SimpleNamespace(load=lambda *a, **kw: {"w": [3.0]})
    monkeypatch.setattr(vd_mod, "torch", fake_torch_raw, raising=False)
    det_raw = VisualDetector(checkpoint_path=raw_ckpt)
    assert det_raw.load_model()["backend"] == "torch_checkpoint"
    assert det_raw._model == {"w": [3.0]}

    empty_ckpt = tmp_path / "fake_empty.pt"
    empty_ckpt.write_bytes(b"pt-empty")
    fake_torch_empty = types.SimpleNamespace(load=lambda *a, **kw: {"state_dict": {}})
    monkeypatch.setattr(vd_mod, "torch", fake_torch_empty, raising=False)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=empty_ckpt).load_model()

    bad_obj_ckpt = tmp_path / "fake_bad_obj.pt"
    bad_obj_ckpt.write_bytes(b"pt-bad")
    fake_torch_bad_obj = types.SimpleNamespace(load=lambda *a, **kw: ["invalid"])
    monkeypatch.setattr(vd_mod, "torch", fake_torch_bad_obj, raising=False)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=bad_obj_ckpt).load_model()

    err_ckpt = tmp_path / "fake_err.pt"
    err_ckpt.write_bytes(b"pt-err")
    def _boom(*a, **kw):
        raise RuntimeError("corrupt archive")
    monkeypatch.setattr(vd_mod, "torch", types.SimpleNamespace(load=_boom), raising=False)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=err_ckpt).load_model()


def test_visual_detector_load_model_fallback_when_torch_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import doc_auth.deepfake.visual_detector as vd_mod

    monkeypatch.setattr(vd_mod, "torch", None, raising=False)
    ckpt = tmp_path / "fallback.pt"
    ckpt.write_bytes(b"pt-data")

    detector = VisualDetector(checkpoint_path=ckpt)
    with pytest.raises(ModelUnavailableError):
        detector.load_model()


def test_visual_detector_load_model_unsupported_suffix(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.onnx"
    ckpt.write_bytes(b"onnx-weights")

    detector = VisualDetector(checkpoint_path=ckpt)
    with pytest.raises(ModelUnavailableError):
        detector.load_model()

def test_visual_detector_nn_and_unavailable_checkpoint(monkeypatch, tmp_path):
    import pytest
    from doc_auth.deepfake import visual_detector as vd_mod
    from doc_auth.deepfake.exceptions import ModelUnavailableError

    class _DummyTorch:
        @staticmethod
        def load(_path, map_location=None):
            return {}

    class _DummyNN:
        Module = object

    monkeypatch.setattr(vd_mod, "torch", _DummyTorch())
    monkeypatch.setattr(vd_mod, "nn", _DummyNN())
    assert vd_mod.nn is not None
    missing_ckpt = tmp_path / "nonexistent_weights.pt"
    det = vd_mod.VisualDetector(checkpoint_path=str(missing_ckpt))
    with pytest.raises(ModelUnavailableError):
        det.load()
