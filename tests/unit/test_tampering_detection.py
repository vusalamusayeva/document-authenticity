from pathlib import Path

import cv2
import numpy as np
import pytest

from doc_auth.forensics.tampering_detection import TamperingDetector


def test_init_and_extract_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TamperingDetector(anomaly_threshold=1.5)
    with pytest.raises(ValueError):
        TamperingDetector(anomaly_threshold=-0.1)
    with pytest.raises(ValueError):
        TamperingDetector(jpeg_quality=0)
    with pytest.raises(ValueError):
        TamperingDetector(jpeg_quality=101)

    det = TamperingDetector(anomaly_threshold=0.5, jpeg_quality=85)
    assert det._extract_image(np.array([], dtype=np.uint8)) is None
    assert det._extract_image({"other": "val"}) is None

    img = np.full((32, 48, 3), 140, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (40, 25), (20, 20, 20), 2)
    assert det._extract_image({"frame": img}) is not None

    img_path = tmp_path / "tamper_sample.png"
    assert cv2.imwrite(str(img_path), img)
    loaded = det._extract_image({"path": str(img_path)})
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (32, 48, 3)


def test_detect_branches_and_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    det = TamperingDetector(anomaly_threshold=0.0, jpeg_quality=90)

    missing = det.detect(None)
    assert missing["status"] == "not_applicable"
    assert missing["tampered"] is None

    # High-contrast pattern gives non-zero ELA difference
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    res_high = det.detect(noisy)
    assert res_high["status"] == "success"
    assert res_high["tampered"] is True
    assert 0.0 <= res_high["tampering_score"] <= 1.0
    assert det.analyze(noisy)["tampering_score"] == pytest.approx(res_high["tampering_score"])
    assert det.run(noisy)["tampering_score"] == pytest.approx(res_high["tampering_score"])
    assert det(noisy)["tampering_score"] == pytest.approx(res_high["tampering_score"])

    # High threshold keeps flat image untampered
    det_strict = TamperingDetector(anomaly_threshold=0.99, jpeg_quality=95)
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    res_low = det_strict.detect(flat)
    assert res_low["status"] == "success"
    assert res_low["tampered"] is False

    # Encode failure branch
    monkeypatch.setattr(cv2, "imencode", lambda *args, **kwargs: (False, None))
    enc_err = det.detect(flat)
    assert enc_err["status"] == "error"

    # Decode shape mismatch branch
    monkeypatch.setattr(cv2, "imencode", lambda *args, **kwargs: (True, np.array([1, 2, 3], dtype=np.uint8)))
    monkeypatch.setattr(cv2, "imdecode", lambda *args, **kwargs: np.zeros((10, 10, 3), dtype=np.uint8))
    dec_err = det.detect(flat)
    assert dec_err["status"] == "error"
