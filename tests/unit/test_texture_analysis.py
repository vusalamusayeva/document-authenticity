import cv2
import numpy as np
import pytest

from doc_auth.signature_stamp.exceptions import SignatureStampError
from doc_auth.signature_stamp.texture_analysis import analyze_texture


def test_analyze_texture_textured_and_bbox() -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(120, 140, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 25), (90, 95), (255, 255, 255), 2)

    full_res = analyze_texture(img)
    assert set(full_res.keys()) == {
        "sharpness_variance",
        "noise_difference",
        "edge_density",
        "is_too_smooth",
        "has_halo_or_blending",
        "digital_manipulation_probability",
    }
    assert 0.0 <= full_res["digital_manipulation_probability"] <= 1.0

    cropped_res = analyze_texture(img, bbox=(20, 20, 100, 100))
    assert cropped_res["sharpness_variance"] >= 0.0

    empty_roi_res = analyze_texture(img, bbox=(50, 50, 50, 50))
    assert empty_roi_res["sharpness_variance"] == pytest.approx(full_res["sharpness_variance"])


def test_analyze_texture_smooth_and_errors() -> None:
    flat_gray = np.full((80, 80), 128, dtype=np.uint8)
    smooth_res = analyze_texture(flat_gray)
    assert smooth_res["is_too_smooth"] is True
    assert smooth_res["has_halo_or_blending"] is True
    assert smooth_res["digital_manipulation_probability"] == pytest.approx(1.0)

    with pytest.raises(SignatureStampError):
        analyze_texture(np.array([], dtype=np.uint8))

    with pytest.raises(SignatureStampError):
        analyze_texture(None)  # type: ignore[arg-type]

    with pytest.raises(SignatureStampError):
        analyze_texture(flat_gray, bbox=(1, 2, 3))  # type: ignore[arg-type]
