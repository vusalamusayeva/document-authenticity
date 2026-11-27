import cv2
import numpy as np
import pytest

from doc_auth.signature_stamp.signature_similarity import SignatureSimilarity


def _make_signature_img(seed: int = 1, draw_stroke: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((160, 320, 3), 245, dtype=np.uint8)
    if draw_stroke:
        cv2.ellipse(img, (160, 80), (90, 35), 10, 0, 360, (15, 15, 15), 4)
        cv2.line(img, (60, 95), (260, 65), (20, 20, 20), 3)
    noise = rng.integers(-5, 6, size=img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_signature_similarity_metrics_and_compare() -> None:
    img_a = _make_signature_img(seed=10, draw_stroke=True)
    img_b = _make_signature_img(seed=20, draw_stroke=True)
    blank = np.full((160, 320, 3), 255, dtype=np.uint8)

    assert 0.0 <= SignatureSimilarity.calculate_ssim(img_a, img_b) <= 1.0
    assert 0.0 <= SignatureSimilarity.calculate_hu_moments(img_a, img_b) <= 1.0
    assert 0.0 <= SignatureSimilarity.calculate_contour_similarity(img_a, img_b) <= 1.0
    assert SignatureSimilarity.calculate_contour_similarity(img_a, blank) == 0.0
    assert 0.0 <= SignatureSimilarity.calculate_orb_matching(img_a, img_b) <= 1.0

    summary = SignatureSimilarity.compare(img_a, img_b)
    for key in ("ssim", "hu_moments", "contour_similarity", "orb_matching", "total_similarity"):
        assert key in summary
        assert 0.0 <= summary[key] <= 1.0


def test_signature_similarity_empty_matches_and_exception_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    img_a = _make_signature_img(seed=1, draw_stroke=True)
    img_b = _make_signature_img(seed=2, draw_stroke=True)

    class _EmptyMatcher:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def match(self, *args, **kwargs):
            return []

    monkeypatch.setattr(cv2, "BFMatcher", _EmptyMatcher)
    assert SignatureSimilarity.calculate_orb_matching(img_a, img_b) == 0.0

    monkeypatch.setattr(
        SignatureSimilarity,
        "_resize_and_mask",
        classmethod(lambda cls, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    assert SignatureSimilarity.calculate_ssim(img_a, img_b) == 0.0
    assert SignatureSimilarity.calculate_hu_moments(img_a, img_b) == 0.0
    assert SignatureSimilarity.calculate_contour_similarity(img_a, img_b) == 0.0
    assert SignatureSimilarity.calculate_orb_matching(img_a, img_b) == 0.0
