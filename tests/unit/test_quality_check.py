import cv2
import numpy as np
import pytest

from doc_auth.preprocessing import quality_check as qc


def test_validate_and_preprocess_image_branches() -> None:
    assert qc.validate_image(None) is False  # type: ignore[arg-type]
    assert qc.validate_image("not-an-array") is False  # type: ignore[arg-type]
    assert qc.validate_image(np.array([], dtype=np.uint8)) is False

    valid_img = np.full((32, 32, 3), 120, dtype=np.uint8)
    assert qc.validate_image(valid_img) is True

    with pytest.raises(ValueError):
        qc.preprocess_image(np.array([], dtype=np.uint8))

    # Color path with grayscale=False, denoise, contrast, deskew, and binarization
    skewed = np.full((64, 96, 3), 255, dtype=np.uint8)
    cv2.line(skewed, (10, 12), (85, 36), (0, 0, 0), 3)
    res_color = qc.preprocess_image(
        skewed,
        config={
            "grayscale": False,
            "denoise": True,
            "contrast": True,
            "deskew": True,
            "binarization": True,
        },
    )
    assert isinstance(res_color, np.ndarray)
    assert res_color.ndim == 2
    assert res_color.shape == (64, 96)

    # Grayscale input path with denoise & contrast disabled
    gray_in = np.full((24, 24), 140, dtype=np.uint8)
    res_gray = qc.preprocess_image(
        gray_in,
        config={"grayscale": True, "denoise": False, "contrast": False, "deskew": False},
    )
    assert res_gray.shape == (24, 24)


def test_quality_dict_blur_brightness_and_run_quality_check() -> None:
    q = qc._QualityDict({"passed": True, "custom": 7}, unpack_pair=(12.5, True))
    assert q.custom == 7
    assert q.is_acceptable is True
    val, ok = q
    assert val == pytest.approx(12.5)
    assert ok is True
    with pytest.raises(AttributeError):
        _ = q.nonexistent_attr

    with pytest.raises(ValueError):
        qc.check_blur(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError):
        qc.check_brightness(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError):
        qc.run_quality_check(np.array([], dtype=np.uint8))

    # Dark image and bright image checks with min_b/max_b
    dark_img = np.full((20, 20), 10, dtype=np.uint8)
    dark_res = qc.check_brightness(dark_img, min_b=30.0, max_b=200.0)
    assert dark_res["is_dark"] is True
    assert dark_res["passed"] is False

    bright_img = np.full((20, 20, 3), 245, dtype=np.uint8)
    bright_res = qc.check_brightness(bright_img, min_b=30.0, max_b=200.0)
    assert bright_res["is_bright"] is True
    assert bright_res["passed"] is False

    # Sharp checkerboard image passes blur & brightness checks
    sharp = np.zeros((64, 64, 3), dtype=np.uint8)
    for r in range(0, 64, 4):
        for c in range(0, 64, 4):
            color = 210 if ((r // 4 + c // 4) % 2 == 0) else 60
            sharp[r : r + 4, c : c + 4] = color

    blur_res = qc.check_blur(sharp, threshold=10.0)
    assert blur_res["passed"] is True
    assert blur_res["is_blurry"] is False

    full_res = qc.run_quality_check(
        sharp,
        config={"blur_threshold": 10.0, "min_brightness": 20.0, "max_brightness": 240.0, "denoise": False},
    )
    assert full_res["status"] == "passed"
    assert full_res["passed"] is True
    assert full_res["processed_shape"] == [64, 64]
import numpy as np
import pytest

from doc_auth.preprocessing import quality_check as qc
from doc_auth.preprocessing.quality_check import _QualityDict, preprocess_image, run_quality_check


def test_quality_check_deskew_negative_45_and_quality_dict_iter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = np.full((80, 120, 3), 255, dtype=np.uint8)
    img[35:45, 20:100] = 0

    monkeypatch.setattr(
        qc.cv2,
        "minAreaRect",
        lambda _pts: ((60.0, 40.0), (80.0, 10.0), -60.0),
    )
    out = preprocess_image(
        img,
        config={"grayscale": True, "denoise": False, "contrast": False, "deskew": True, "binarization": False},
    )
    assert out.shape == (80, 120)

    qd = _QualityDict({"passed": True, "other": 1})
    assert qd.is_acceptable is True
    assert list(qd) == ["passed", "other"]

    q_res = run_quality_check(
        img,
        config={"denoise": False, "contrast": False, "deskew": False, "binarization": False},
    )
    assert "status" in list(q_res)
from doc_auth.preprocessing.quality_check import _QualityDict


def test_quality_dict_getattr_is_acceptable_fallback() -> None:
    q = _QualityDict({"passed": True})
    assert q.__getattr__("is_acceptable") is True
