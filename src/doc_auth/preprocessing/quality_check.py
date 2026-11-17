import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class QualityResult:
    is_acceptable: bool
    blur_score: float
    brightness_score: float
    reason: str | None = None


def check_blur(image, threshold: float = 100.0) -> tuple[float, bool]:
    """Laplacian variance ilə bulanıqlıq yoxlanması (deterministic test üçün uyğun)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_sharp = score >= threshold
    return score, is_sharp


def check_brightness(image, min_b: int = 40, max_b: int = 220) -> tuple[float, bool]:
    """Orta piksel intensivliyi ilə işıqlanma səviyyəsinin yoxlanması."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    score = float(np.mean(gray))
    is_acceptable = min_b <= score <= max_b
    return score, is_acceptable


def run_quality_check(image, cfg: dict) -> QualityResult:
    """
    Əsas giriş nöqtəsi: pipeline.yaml-dəki 'preprocessing' addımı bunu çağırır.
    reject_on_low_quality=true olduqda is_acceptable=False nəticəsi
    pipeline-ı dayandırıb rədd/yenidən çəkim tələb etməli.
    """
    blur_thresh = cfg.get("blur_threshold", 100.0)
    min_b = cfg.get("min_brightness", 40)
    max_b = cfg.get("max_brightness", 220)
    reject_on_low_q = cfg.get("reject_on_low_quality", True)

    blur_score, is_sharp = check_blur(image, blur_thresh)
    brightness_score, is_bright_ok = check_brightness(image, min_b, max_b)

    reasons = []
    if not is_sharp:
        reasons.append(f"Şəkil çox bulanıqdır (score: {blur_score:.2f} < {blur_thresh})")
    if not is_bright_ok:
        reasons.append(f"İşıqlanma səviyyəsi uyğun deyil (score: {brightness_score:.2f})")

    is_acceptable = True
    reason = None
    if reasons:
        reason = "; ".join(reasons)
        if reject_on_low_q:
            is_acceptable = False

    return QualityResult(
        is_acceptable=is_acceptable,
        blur_score=blur_score,
        brightness_score=brightness_score,
        reason=reason
    )