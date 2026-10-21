"""
Preprocessing & Quality Check
- format yoxlanması, resize, denoising
- keyfiyyət skoru (bulanıqlıq, işıqlanma)
"""

from dataclasses import dataclass


@dataclass
class QualityResult:
    is_acceptable: bool
    blur_score: float
    brightness_score: float
    reason: str | None = None


def check_blur(image, threshold: float = 100.0) -> tuple[float, bool]:
    """Laplacian variance ilə bulanıqlıq yoxlanması (deterministic test üçün uyğun)."""
    raise NotImplementedError("cv2.Laplacian(image, cv2.CV_64F).var() ilə implement edin")


def check_brightness(image, min_b: int = 40, max_b: int = 220) -> tuple[float, bool]:
    raise NotImplementedError


def run_quality_check(image, cfg: dict) -> QualityResult:
    """
    Əsas giriş nöqtəsi: pipeline.yaml-dəki 'preprocessing' addımı bunu çağırır.
    reject_on_low_quality=true olduqda is_acceptable=False nəticəsi
    pipeline-ı dayandırıb "rədd/yenidən çəkim tələb et" mesajı qaytarmalıdır.
    """
    raise NotImplementedError
