import cv2
import numpy as np
from typing import Dict, Any, Optional

def validate_image(image: np.ndarray) -> bool:
    """
    Şəkil massivinin etibarlı olub-olmadığını yoxlayır.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return False
    return True

def preprocess_image(image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Sənəd şəkilləri üçün ilkin emal və keyfiyyət yoxlaması həyata keçirir.
    Orijinal şəkli dəyişməmək üçün surəti (.copy()) üzərində işləyir.
    
    Parametrlər:
    - image: numpy.ndarray (giriş şəkli)
    - config: konfiqurasiya lüğəti (grayscale, denoise, contrast, deskew, binarization)
    """
    if not validate_image(image):
        raise ValueError("Etibarsız və ya boş şəkil daxil edildi.")

    # Orijinal obyektin zədələnməsinin qarşısını almaq üçün surətini çıxarırıq
    processed = image.copy()

    config = config or {}
    to_grayscale = config.get("grayscale", True)
    enable_denoise = config.get("denoise", True)
    enable_contrast = config.get("contrast", True)
    enable_deskew = config.get("deskew", True)
    enable_binarization = config.get("binarization", False)

    # 1. Grayscale çevrilməsi
    if len(processed.shape) == 3 and to_grayscale:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # 2. Denoise (Səs-küyün təmizlənməsi)
    if enable_denoise:
        if len(processed.shape) == 2:
            processed = cv2.fastNlMeansDenoising(processed, h=10)
        else:
            processed = cv2.fastNlMeansDenoisingColored(processed, h=10)

    # 3. Contrast (CLAHE - Kontrastın optimallaşdırılması)
    if enable_contrast:
        if len(processed.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        else:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 4. Deskew (Əyriliyin avtomatik düzəldilməsi)
    if enable_deskew:
        gray_for_deskew = processed if len(processed.shape) == 2 else cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray_for_deskew < 255))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.5:  # Yalnız əyrilik nəzərəçarpacaq dərəcədə olduqda döndər
                (h, w) = processed.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                processed = cv2.warpAffine(processed, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 5. Configurable Binarization (Ağ-qara ikili format)
    if enable_binarization:
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    return processed


class _QualityDict(dict):
    def __init__(self, data: Dict[str, Any], unpack_pair: tuple[Any, Any] | None = None):
        super().__init__(data)
        self._unpack_pair = unpack_pair

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        if name == "is_acceptable":
            return bool(self.get("passed", False))
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def is_acceptable(self) -> bool:
        return bool(self.get("passed", False))

    def __iter__(self):
        if self._unpack_pair is not None:
            return iter(self._unpack_pair)
        return super().__iter__()


def check_blur(image: np.ndarray, threshold: float = 100.0) -> _QualityDict:
    if not validate_image(image):
        raise ValueError("Etibars?z v? ya bo? ??kil daxil edildi.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = bool(laplacian_var < float(threshold))
    passed = not is_blurry
    val = round(laplacian_var, 4)
    return _QualityDict(
        {
            "score": val,
            "blur_score": val,
            "laplacian_variance": val,
            "threshold": float(threshold),
            "is_blurry": is_blurry,
            "is_sharp": passed,
            "passed": passed,
            "is_acceptable": passed,
        },
        unpack_pair=(val, passed),
    )


def check_brightness(
    image: np.ndarray,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
    min_b: Optional[float] = None,
    max_b: Optional[float] = None,
) -> _QualityDict:
    if not validate_image(image):
        raise ValueError("Etibars?z v? ya bo? ??kil daxil edildi.")
    lo = float(min_b if min_b is not None else min_brightness)
    hi = float(max_b if max_b is not None else max_brightness)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mean_val = float(np.mean(gray))
    is_dark = bool(mean_val < lo)
    is_bright = bool(mean_val > hi)
    passed = not (is_dark or is_bright)
    val = round(mean_val, 4)
    return _QualityDict(
        {
            "score": val,
            "mean_brightness": val,
            "min_brightness": lo,
            "max_brightness": hi,
            "is_dark": is_dark,
            "is_bright": is_bright,
            "passed": passed,
            "is_acceptable": passed,
        },
        unpack_pair=(val, passed),
    )


def run_quality_check(
    image: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> _QualityDict:
    if not validate_image(image):
        raise ValueError("Etibars?z v? ya bo? ??kil daxil edildi.")
    cfg = config or {}
    blur_res = check_blur(image, threshold=float(cfg.get("blur_threshold", 100.0)))
    min_b = cfg.get("min_brightness", cfg.get("min_b", 40.0))
    max_b = cfg.get("max_brightness", cfg.get("max_b", 220.0))
    bright_res = check_brightness(
        image,
        min_brightness=float(min_b),
        max_brightness=float(max_b),
    )
    processed = preprocess_image(image, config=cfg)
    passed = bool(blur_res["passed"] and bright_res["passed"])
    return _QualityDict(
        {
            "status": "passed" if passed else "warning",
            "passed": passed,
            "is_acceptable": passed,
            "blur_score": blur_res["blur_score"],
            "laplacian_variance": blur_res["laplacian_variance"],
            "is_blurry": blur_res["is_blurry"],
            "brightness_score": bright_res["mean_brightness"],
            "mean_brightness": bright_res["mean_brightness"],
            "is_dark": bright_res["is_dark"],
            "is_bright": bright_res["is_bright"],
            "blur": blur_res,
            "brightness": bright_res,
            "processed_shape": list(processed.shape),
            "processed_image": processed,
        }
    )
