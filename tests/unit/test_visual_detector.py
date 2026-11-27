import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from doc_auth.deepfake import visual_detector as vd_mod
from doc_auth.deepfake.exceptions import InvalidConfigError, ModelUnavailableError
from doc_auth.deepfake.visual_detector import VisualDetector


class _DummyFaceDetector:
    def __init__(self, faces: list[Any] | None = None) -> None:
        self._faces = faces if faces is not None else []

    def detect_faces(self, frame: np.ndarray) -> list[Any]:
        return self._faces


def _write_valid_npz(path: Path, input_size: tuple[int, int] = (32, 32)) -> None:
    np.savez(
        path,
        weights=np.array([0.1, 0.2, -0.05, 0.3], dtype=np.float32),
        bias=np.array([0.05], dtype=np.float32),
        laplacian_scale=np.array([1.0], dtype=np.float32),
        input_size=np.array(list(input_size), dtype=np.int32),
        class_mapping=np.array([json.dumps({"0": "real", "1": "deepfake"})]),
    )


def test_visual_detector_config_and_npz_lifecycle(tmp_path: Path) -> None:
    with pytest.raises(InvalidConfigError):
        VisualDetector(threshold=1.5)
    with pytest.raises(InvalidConfigError):
        VisualDetector(threshold=True)

    npz_path = tmp_path / "model.npz"
    _write_valid_npz(npz_path, input_size=(32, 32))
    digest = VisualDetector.compute_file_sha256(npz_path)

    det = VisualDetector(
        model_path=npz_path,
        threshold=0.5,
        face_detector=_DummyFaceDetector(faces=[{"bbox": [2, 2, 20, 20]}, [0, 0, 0, 0]]),  # type: ignore[arg-type]
        input_size=(32, 32),
        expected_sha256=digest,
    )

    info1 = det.load_model()
    assert info1["status"] == "loaded"
    assert info1["backend"] == "numpy_linear_model"
    # Second load hits cached loaded branch
    info2 = det.load_model()
    assert info2["status"] == "loaded"

    frame = np.full((40, 40, 3), 120, dtype=np.uint8)
    res = det.analyze_frame(frame)
    assert res["status"] == "success"
    assert res["face_detected"] is True
    assert len(res["face_predictions"]) == 2
    assert res["max_deepfake_score"] == max(p["score"] for p in res["face_predictions"])

    assert det.analyze_frame(np.empty((0, 0, 3), dtype=np.uint8))["status"] == "invalid_frame"

    no_face_det = VisualDetector(
        model_path=npz_path,
        face_detector=_DummyFaceDetector(faces=[]),  # type: ignore[arg-type]
        input_size=(32, 32),
    )
    assert no_face_det.analyze_frame(frame)["status"] == "no_face"
    assert no_face_det._compute_region_score(np.array([], dtype=np.uint8)) == 0.0


def test_visual_detector_torch_and_error_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pt_path = tmp_path / "ckpt.pt"
    pt_path.write_bytes(b"pt-bytes")

    if vd_mod.torch is not None and vd_mod.nn is not None:
        mod = vd_mod.nn.Linear(4, 2)
        monkeypatch.setattr(vd_mod.torch, "load", lambda *args, **kwargs: mod)
        det_mod = VisualDetector(checkpoint_path=pt_path)
        assert det_mod.load_model()["backend"] == "torch_checkpoint"

        det_no_weights = VisualDetector(checkpoint_path=pt_path)
        det_no_weights._model = mod
        with pytest.raises(ModelUnavailableError):
            det_no_weights._compute_region_score(np.ones((10, 10, 3), dtype=np.uint8))

        monkeypatch.setattr(vd_mod.torch, "load", lambda *args, **kwargs: {"state_dict": {"module.w": 1}})
        det_sd = VisualDetector(checkpoint_path=pt_path)
        assert det_sd.load_model()["backend"] == "torch_checkpoint"
        assert "w" in det_sd._model

        monkeypatch.setattr(vd_mod.torch, "load", lambda *args, **kwargs: "bad-obj")
        det_bad = VisualDetector(checkpoint_path=pt_path)
        with pytest.raises(ModelUnavailableError):
            det_bad.load_model()

    # SHA-256 mismatch error
    npz_path = tmp_path / "bad_hash.npz"
    _write_valid_npz(npz_path, input_size=(32, 32))
    det_hash = VisualDetector(model_path=npz_path, input_size=(32, 32), expected_sha256="0" * 64)
    with pytest.raises(ModelUnavailableError):
        det_hash.load_model()
