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


def _write_npz(path: Path, **arrays: Any) -> None:
    np.savez(path, **arrays)


def _valid_npz_kwargs(input_size: tuple[int, int] = (32, 32)) -> dict[str, Any]:
    return {
        "weights": np.array([0.1, 0.2, -0.05, 0.3], dtype=np.float32),
        "bias": np.array([0.05], dtype=np.float32),
        "laplacian_scale": np.array([1.0], dtype=np.float32),
        "input_size": np.array(list(input_size), dtype=np.int32),
        "class_mapping": np.array([json.dumps({"0": "real", "1": "deepfake"})]),
    }


def test_visual_detector_weight_validation_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(InvalidConfigError):
        VisualDetector(threshold=1.5)
    with pytest.raises(InvalidConfigError):
        VisualDetector(threshold=True)

    with pytest.raises(ModelUnavailableError):
        VisualDetector().verify_model_weights()

    missing = tmp_path / "nope.npz"
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=missing).verify_model_weights()

    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=empty).verify_model_weights()

    bad_ext = tmp_path / "model.bin"
    bad_ext.write_bytes(b"data")
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=bad_ext).verify_model_weights()

    pt_file = tmp_path / "model.pt"
    pt_file.write_bytes(b"pt")
    monkeypatch.setattr(vd_mod, "torch", None)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=pt_file).verify_model_weights()


def test_visual_detector_npz_validation_and_frame_analysis(tmp_path: Path) -> None:
    # Missing required keys
    p_missing = tmp_path / "missing_keys.npz"
    _write_npz(p_missing, weights=np.ones(4, dtype=np.float32))
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_missing).load_model()

    # Wrong shapes
    p_shape = tmp_path / "bad_shape.npz"
    kw_shape = _valid_npz_kwargs((32, 32))
    kw_shape["weights"] = np.ones((2, 2), dtype=np.float32)
    _write_npz(p_shape, **kw_shape)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_shape, input_size=(32, 32)).load_model()

    # Non-finite values
    p_nan = tmp_path / "nan.npz"
    kw_nan = _valid_npz_kwargs((32, 32))
    kw_nan["bias"] = np.array([np.nan], dtype=np.float32)
    _write_npz(p_nan, **kw_nan)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_nan, input_size=(32, 32)).load_model()

    # Mismatched input_size
    p_size = tmp_path / "bad_size.npz"
    _write_npz(p_size, **_valid_npz_kwargs((16, 16)))
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_size, input_size=(32, 32)).load_model()

    # Invalid class_mapping
    p_map = tmp_path / "bad_map.npz"
    kw_map = _valid_npz_kwargs((32, 32))
    kw_map["class_mapping"] = np.array([json.dumps({"0": "cat", "1": "dog"})])
    _write_npz(p_map, **kw_map)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_map, input_size=(32, 32)).load_model()

    # Corrupted npz file
    p_corrupt = tmp_path / "corrupt.npz"
    p_corrupt.write_bytes(b"not-a-zip")
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_corrupt).load_model()

    # Valid npz + SHA-256 + cached reload + face without bbox fallback
    p_ok = tmp_path / "ok.npz"
    _write_npz(p_ok, **_valid_npz_kwargs((32, 32)))
    sha = VisualDetector.compute_file_sha256(p_ok)
    det = VisualDetector(
        model_path=p_ok,
        threshold=0.0,
        face_detector=_DummyFaceDetector(faces=[{"bbox": [2, 2, 18, 18]}, {}]),  # type: ignore[arg-type]
        input_size=(32, 32),
        expected_sha256=sha,
    )
    assert det.load_model()["backend"] == "numpy_linear_model"
    assert det.load_model()["status"] == "loaded"

    frame = np.full((36, 36, 3), 128, dtype=np.uint8)
    out = det.analyze_frame(frame)
    assert out["status"] == "success"
    assert len(out["face_predictions"]) == 2
    assert out["face_predictions"][1]["bbox"] == [0, 0, 36, 36]
    assert out["face_predictions"][0]["is_deepfake"] is True

    assert det.analyze_frame(np.empty((0, 0, 3), dtype=np.uint8))["status"] == "invalid_frame"
    no_face = VisualDetector(model_path=p_ok, face_detector=_DummyFaceDetector([]), input_size=(32, 32))  # type: ignore[arg-type]
    assert no_face.analyze_frame(frame)["status"] == "no_face"
    assert no_face._compute_region_score(np.array([], dtype=np.uint8)) == 0.0


def test_visual_detector_torch_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if vd_mod.torch is None or vd_mod.nn is None:
        pytest.skip("torch not installed")

    pt_path = tmp_path / "ckpt.pt"
    pt_path.write_bytes(b"pt-bytes")

    # torch.load raises
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("corrupt pt")

    monkeypatch.setattr(vd_mod.torch, "load", _raise)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=pt_path).load_model()

    # nn.Module branch + missing weights error in _compute_region_score
    mod = vd_mod.nn.Linear(4, 2)
    monkeypatch.setattr(vd_mod.torch, "load", lambda *args, **kwargs: mod)
    det_mod = VisualDetector(checkpoint_path=pt_path)
    assert det_mod.load_model()["backend"] == "torch_checkpoint"
    with pytest.raises(ModelUnavailableError):
        det_mod._compute_region_score(np.ones((12, 12, 3), dtype=np.uint8))

    # dict with state_dict and module. prefix
    monkeypatch.setattr(vd_mod.torch, "load", lambda *args, **kwargs: {"state_dict": {"module.fc.weight": 1}})
    det_sd = VisualDetector(checkpoint_path=pt_path)
    assert det_sd.load_model()["backend"] == "torch_checkpoint"
    assert det_sd._model == {"fc.weight": 1}

    # empty dict -> invalid state_dict
    monkeypatch.setattr(vd_mod.torch, "load", lambda *args, **kwargs: {"state_dict": {}})
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=pt_path).load_model()

    # non-module, non-dict object
    monkeypatch.setattr(vd_mod.torch, "load", lambda *args, **kwargs: 12345)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=pt_path).load_model()
