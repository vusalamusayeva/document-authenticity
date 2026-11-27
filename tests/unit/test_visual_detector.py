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
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from doc_auth.deepfake import visual_detector as vd_mod
from doc_auth.deepfake.exceptions import ModelUnavailableError
from doc_auth.deepfake.visual_detector import VisualDetector


def _save_valid_npz(path: Path, input_size: tuple[int, int] = (224, 224)) -> None:
    np.savez(
        path,
        weights=np.array([0.2, -0.1, 0.05, 0.3], dtype=np.float32),
        bias=np.array([0.0], dtype=np.float32),
        laplacian_scale=np.array([1.0], dtype=np.float32),
        input_size=np.array(list(input_size), dtype=np.int32),
        class_mapping=np.array(['{"0": "real", "1": "deepfake"}']),
    )


def test_visual_detector_sha256_mismatch_and_autoload(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.npz"
    _save_valid_npz(ckpt)

    bad_sha = VisualDetector(checkpoint_path=ckpt, expected_sha256="0" * 64)
    with pytest.raises(ModelUnavailableError):
        bad_sha.verify_model_weights()

    det = VisualDetector(checkpoint_path=ckpt)
    patch = np.full((32, 32, 3), 120, dtype=np.uint8)
    score = det._compute_region_score(patch)
    assert 0.0 <= score <= 1.0
    # Second load hits cached loaded branch
    cached = det.load_model()
    assert cached["status"] == "loaded"
    assert cached["backend"] == "numpy_linear_model"


def test_visual_detector_torch_checkpoint_and_missing_weights_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if vd_mod.torch is None or vd_mod.nn is None:
        pytest.skip("torch is not installed")

    pt_path = tmp_path / "weights.pt"
    pt_path.write_bytes(b"fake-pt-bytes")

    class _DummyNet(vd_mod.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = vd_mod.nn.Linear(4, 1)

    dummy_net = _DummyNet()
    monkeypatch.setattr(vd_mod.torch, "load", lambda *args, **kwargs: dummy_net)

    det = VisualDetector(checkpoint_path=pt_path)
    info = det.load_model()
    assert info["backend"] == "torch_checkpoint"
    assert det.load_model()["status"] == "loaded"

    # _compute_region_score raises ModelUnavailableError when torch model has no _weights_data
    with pytest.raises(ModelUnavailableError):
        det._compute_region_score(np.ones((16, 16, 3), dtype=np.uint8))

    # state_dict with module. prefix
    monkeypatch.setattr(
        vd_mod.torch,
        "load",
        lambda *args, **kwargs: {"state_dict": {"module.weight": np.array([1.0])}},
    )
    det2 = VisualDetector(checkpoint_path=pt_path)
    info2 = det2.load_model()
    assert info2["backend"] == "torch_checkpoint"
    assert isinstance(det2._model, dict) and "weight" in det2._model

    # Empty state_dict raises ModelUnavailableError
    monkeypatch.setattr(vd_mod.torch, "load", lambda *args, **kwargs: {"model_state_dict": {}})
    det3 = VisualDetector(checkpoint_path=pt_path)
    with pytest.raises(ModelUnavailableError):
        det3.load_model()

    # Unsupported object type raises ModelUnavailableError
    monkeypatch.setattr(vd_mod.torch, "load", lambda *args, **kwargs: ["invalid"])
    det4 = VisualDetector(checkpoint_path=pt_path)
    with pytest.raises(ModelUnavailableError):
        det4.load_model()

    # torch.load failure raises ModelUnavailableError
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr(vd_mod.torch, "load", _boom)
    det5 = VisualDetector(checkpoint_path=pt_path)
    with pytest.raises(ModelUnavailableError):
        det5.load_model()
from pathlib import Path
import numpy as np
import pytest

from doc_auth.deepfake import visual_detector as vd_mod
from doc_auth.deepfake.exceptions import ModelUnavailableError
from doc_auth.deepfake.visual_detector import VisualDetector


@pytest.mark.skipif(vd_mod.torch is None or vd_mod.nn is None, reason="PyTorch required")
def test_visual_detector_torch_module_and_state_dict(tmp_path: Path) -> None:
    torch = vd_mod.torch
    nn = vd_mod.nn

    class _TinyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 1)

        def forward(self, x):
            return self.fc(x)

    pt_file = tmp_path / "model.pt"
    torch.save(_TinyNet(), pt_file)

    det = VisualDetector(checkpoint_path=pt_file)
    info = det.load_model()
    assert info["status"] == "loaded"
    assert info["backend"] == "torch_checkpoint"
    # Second call hits cached _model branch
    info_cached = det.load_model()
    assert info_cached["backend"] == "torch_checkpoint"

    # _compute_region_score raises ModelUnavailableError when only _model is set
    with pytest.raises(ModelUnavailableError):
        det._compute_region_score(np.zeros((16, 16, 3), dtype=np.uint8))

    # state_dict with "module." prefix
    pth_file = tmp_path / "weights.pth"
    torch.save({"state_dict": {"module.fc.weight": torch.ones((1, 4))}}, pth_file)
    det_sd = VisualDetector(checkpoint_path=pth_file)
    sd_info = det_sd.load_model()
    assert sd_info["backend"] == "torch_checkpoint"
    assert "fc.weight" in det_sd._model

    # Invalid state_dict (empty dict)
    bad_sd_file = tmp_path / "empty_sd.pth"
    torch.save({"state_dict": {}}, bad_sd_file)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=bad_sd_file).load_model()

    # Unsupported object inside checkpoint (e.g., plain list)
    bad_obj_file = tmp_path / "list_ckpt.pth"
    torch.save([1, 2, 3], bad_obj_file)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=bad_obj_file).load_model()

    # Corrupted torch file raises ModelUnavailableError during torch.load
    corrupt_file = tmp_path / "corrupt.pt"
    corrupt_file.write_bytes(b"not-a-valid-pytorch-archive")
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=corrupt_file).load_model()
