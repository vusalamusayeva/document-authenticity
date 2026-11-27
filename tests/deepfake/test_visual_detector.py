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
