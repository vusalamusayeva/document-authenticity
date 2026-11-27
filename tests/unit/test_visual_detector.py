import json
from pathlib import Path

import numpy as np
import pytest

from doc_auth.deepfake.exceptions import InvalidConfigError, ModelUnavailableError
from doc_auth.deepfake.visual_detector import VisualDetector


def _make_valid_npz(path: Path, input_size=(224, 224), with_mapping: bool = True) -> None:
    payload: dict[str, np.ndarray] = {
        "weights": np.array([0.2, -0.1, 0.3, 0.05], dtype=np.float32),
        "bias": np.array([0.1], dtype=np.float32),
        "laplacian_scale": np.array([1.0], dtype=np.float32),
        "input_size": np.array(list(input_size), dtype=np.int32),
    }
    if with_mapping:
        payload["class_mapping"] = np.array([json.dumps({"0": "real", "1": "deepfake"})])
    np.savez(path, **payload)


def test_visual_detector_threshold_and_weight_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(InvalidConfigError):
        VisualDetector(threshold=True)
    with pytest.raises(InvalidConfigError):
        VisualDetector(threshold=1.5)

    with pytest.raises(ModelUnavailableError):
        VisualDetector().verify_model_weights()

    missing = tmp_path / "missing.npz"
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=missing).verify_model_weights()

    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=empty).verify_model_weights()

    bad_ext = tmp_path / "model.bin"
    bad_ext.write_bytes(b"123")
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=bad_ext).verify_model_weights()

    pt_file = tmp_path / "model.pt"
    pt_file.write_bytes(b"dummy")
    monkeypatch.setattr("doc_auth.deepfake.visual_detector.torch", None)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=pt_file).verify_model_weights()
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=pt_file).load_model()

    npz_file = tmp_path / "valid.npz"
    _make_valid_npz(npz_file)
    real_sha = VisualDetector.compute_file_sha256(npz_file)

    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=npz_file, expected_sha256="0" * 64).verify_model_weights()

    det = VisualDetector(model_path=npz_file, expected_sha256=real_sha)
    assert det.verify_model_weights() is True


def test_visual_detector_npz_loading_and_validation_errors(tmp_path: Path) -> None:
    p_missing = tmp_path / "missing_keys.npz"
    np.savez(p_missing, weights=np.zeros(4, dtype=np.float32))
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_missing).load_model()

    p_shape = tmp_path / "bad_shape.npz"
    np.savez(
        p_shape,
        weights=np.zeros(2, dtype=np.float32),
        bias=np.zeros(1, dtype=np.float32),
        laplacian_scale=np.ones(1, dtype=np.float32),
    )
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_shape).load_model()

    p_nan = tmp_path / "nan.npz"
    np.savez(
        p_nan,
        weights=np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32),
        bias=np.zeros(1, dtype=np.float32),
        laplacian_scale=np.ones(1, dtype=np.float32),
    )
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_nan).load_model()

    p_size = tmp_path / "bad_size.npz"
    _make_valid_npz(p_size, input_size=(128, 128))
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_size, input_size=(224, 224)).load_model()

    p_map = tmp_path / "bad_map.npz"
    np.savez(
        p_map,
        weights=np.ones(4, dtype=np.float32),
        bias=np.zeros(1, dtype=np.float32),
        laplacian_scale=np.ones(1, dtype=np.float32),
        class_mapping=np.array([json.dumps({"0": "cat", "1": "dog"})]),
    )
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_map).load_model()

    p_corrupt = tmp_path / "corrupt.npz"
    p_corrupt.write_bytes(b"not-a-zip-archive")
    with pytest.raises(ModelUnavailableError):
        VisualDetector(model_path=p_corrupt).load_model()

    p_ok = tmp_path / "ok.npz"
    _make_valid_npz(p_ok, input_size=(64, 64))
    det = VisualDetector(model_path=p_ok, input_size=(64, 64))
    info1 = det.load_model()
    assert info1["backend"] == "numpy_linear_model"
    info2 = det.load_model()
    assert info2["status"] == "loaded"

    frame = np.full((64, 64, 3), 120, dtype=np.uint8)
    out = det.analyze_frame(frame, faces=[{"score_hint": 1}])
    assert out["status"] == "success"
    assert out["face_predictions"][0]["bbox"] == [0, 0, 64, 64]


def test_visual_detector_torch_checkpoint_and_unloaded_score(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    pt_mod_path = tmp_path / "module.pt"
    torch.save(nn.Linear(4, 2), pt_mod_path)
    det_mod = VisualDetector(checkpoint_path=pt_mod_path)
    res_mod = det_mod.load_model()
    assert res_mod["backend"] == "torch_checkpoint"
    assert res_mod["status"] == "loaded"

    with pytest.raises(ModelUnavailableError):
        det_mod._compute_region_score(np.ones((16, 16, 3), dtype=np.uint8))

    pt_sd_path = tmp_path / "state.pth"
    torch.save({"model_state_dict": {"module.w": torch.tensor([1.0])}}, pt_sd_path)
    det_sd = VisualDetector(checkpoint_path=pt_sd_path)
    assert det_sd.load_model()["backend"] == "torch_checkpoint"
    assert "w" in det_sd._model

    pt_empty_sd = tmp_path / "empty_sd.pt"
    torch.save({"state_dict": {}}, pt_empty_sd)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=pt_empty_sd).load_model()

    pt_bad_obj = tmp_path / "bad_obj.pt"
    torch.save(["invalid"], pt_bad_obj)
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=pt_bad_obj).load_model()

    pt_corrupt = tmp_path / "corrupt.pt"
    pt_corrupt.write_bytes(b"corrupted-pt")
    with pytest.raises(ModelUnavailableError):
        VisualDetector(checkpoint_path=pt_corrupt).load_model()
