from pathlib import Path

import numpy as np
import pytest

from doc_auth.deepfake.exceptions import ModelUnavailableError
from doc_auth.deepfake.visual_detector import VisualDetector

torch = pytest.importorskip("torch")


def test_visual_detector_loads_torch_checkpoint_and_infers(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "dummy_model.pt"
    torch.save({"epoch": 1, "state_dict": {}}, ckpt_path)

    detector = VisualDetector(checkpoint_path=ckpt_path, threshold=0.5)
    info = detector.load_model()
    assert info["status"] == "loaded"

    fake_face_img = np.random.default_rng(42).integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    faces = [{"bbox": [16, 16, 112, 112], "score": 0.99}]
    out = detector.analyze_frame(fake_face_img, faces=faces)

    assert out["status"] == "success"
    assert out["face_detected"] is True
    assert len(out["face_predictions"]) == 1
    assert 0.0 <= out["score"] <= 1.0


def test_visual_detector_rejects_unsupported_extension(tmp_path: Path) -> None:
    bad_ckpt = tmp_path / "weights.txt"
    bad_ckpt.write_text("not a checkpoint", encoding="utf-8")

    detector = VisualDetector(checkpoint_path=bad_ckpt)
    with pytest.raises(ModelUnavailableError):
        detector.verify_model_weights()


from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from doc_auth.deepfake.exceptions import ModelUnavailableError
from doc_auth.deepfake.visual_detector import VisualDetector


class _TinyTorchNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)


def test_load_model_torch_module_and_state_dicts(tmp_path: Path) -> None:
    mod_file = tmp_path / "model.pt"
    torch.save(_TinyTorchNet(), mod_file)
    det_mod = VisualDetector(checkpoint_path=mod_file)
    info_mod = det_mod.load_model()
    assert info_mod["backend"] == "torch_checkpoint"
    assert info_mod["status"] == "loaded"
    assert isinstance(det_mod._model, nn.Module)
    # Second call hits cached loaded branch
    assert det_mod.load_model()["status"] == "loaded"

    sd_file = tmp_path / "prefixed_sd.pth"
    torch.save({"state_dict": {"module.weight": torch.ones(1, 2)}}, sd_file)
    det_sd = VisualDetector(checkpoint_path=sd_file)
    info_sd = det_sd.load_model()
    assert info_sd["backend"] == "torch_checkpoint"
    assert "weight" in det_sd._model
    assert "module.weight" not in det_sd._model

    msd_file = tmp_path / "model_state_dict.pt"
    torch.save({"model_state_dict": {"fc.bias": torch.zeros(1)}}, msd_file)
    det_msd = VisualDetector(checkpoint_path=msd_file)
    assert det_msd.load_model()["backend"] == "torch_checkpoint"
    assert "fc.bias" in det_msd._model

    raw_dict_file = tmp_path / "raw_dict.pt"
    torch.save({"layer.weight": torch.ones(2, 2)}, raw_dict_file)
    det_raw = VisualDetector(checkpoint_path=raw_dict_file)
    assert det_raw.load_model()["backend"] == "torch_checkpoint"
    assert "layer.weight" in det_raw._model


def test_load_model_torch_invalid_objects_and_score_without_weights(tmp_path: Path) -> None:
    empty_sd = tmp_path / "empty_sd.pt"
    torch.save({"state_dict": {}}, empty_sd)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=empty_sd).load_model()

    bad_obj = tmp_path / "bad_obj.pt"
    torch.save(["not", "a", "module", "or", "dict"], bad_obj)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=bad_obj).load_model()

    mod_file = tmp_path / "model_for_score.pt"
    torch.save(_TinyTorchNet(), mod_file)
    det = VisualDetector(checkpoint_path=mod_file)
    det.load_model()
    patch = np.full((16, 16, 3), 128, dtype=np.uint8)
    with pytest.raises(ModelUnavailableError):
        det._compute_region_score(patch)


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
    # Cached load path
    assert det_mod.load_model()["status"] == "loaded"

    # Line 220: _weights_data is None while _model is set
    patch = np.full((16, 16, 3), 128, dtype=np.uint8)
    with pytest.raises(ModelUnavailableError):
        det_mod._compute_region_score(patch)

    # Prefixed state_dict
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

    # model_state_dict fallback
    msd_ckpt = tmp_path / "fake_msd.pt"
    msd_ckpt.write_bytes(b"pt-msd")
    fake_torch_msd = types.SimpleNamespace(
        load=lambda *a, **kw: {"model_state_dict": {"head.weight": [2.0]}}
    )
    monkeypatch.setattr(vd_mod, "torch", fake_torch_msd, raising=False)
    det_msd = VisualDetector(checkpoint_path=msd_ckpt)
    assert det_msd.load_model()["backend"] == "torch_checkpoint"
    assert "head.weight" in det_msd._model

    # Direct dict state_dict
    raw_ckpt = tmp_path / "fake_raw.pt"
    raw_ckpt.write_bytes(b"pt-raw")
    fake_torch_raw = types.SimpleNamespace(load=lambda *a, **kw: {"w": [3.0]})
    monkeypatch.setattr(vd_mod, "torch", fake_torch_raw, raising=False)
    det_raw = VisualDetector(checkpoint_path=raw_ckpt)
    assert det_raw.load_model()["backend"] == "torch_checkpoint"
    assert det_raw._model == {"w": [3.0]}

    # Empty state_dict -> ModelUnavailableError
    empty_ckpt = tmp_path / "fake_empty.pt"
    empty_ckpt.write_bytes(b"pt-empty")
    fake_torch_empty = types.SimpleNamespace(load=lambda *a, **kw: {"state_dict": {}})
    monkeypatch.setattr(vd_mod, "torch", fake_torch_empty, raising=False)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=empty_ckpt).load_model()

    # Non-dict, non-module object -> ModelUnavailableError
    bad_obj_ckpt = tmp_path / "fake_bad_obj.pt"
    bad_obj_ckpt.write_bytes(b"pt-bad")
    fake_torch_bad_obj = types.SimpleNamespace(load=lambda *a, **kw: ["invalid"])
    monkeypatch.setattr(vd_mod, "torch", fake_torch_bad_obj, raising=False)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=bad_obj_ckpt).load_model()

    # torch.load raises -> ModelUnavailableError
    err_ckpt = tmp_path / "fake_err.pt"
    err_ckpt.write_bytes(b"pt-err")
    def _boom(*a, **kw):
        raise RuntimeError("corrupt archive")
    monkeypatch.setattr(vd_mod, "torch", types.SimpleNamespace(load=_boom), raising=False)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=err_ckpt).load_model()
