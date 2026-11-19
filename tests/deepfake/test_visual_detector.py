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