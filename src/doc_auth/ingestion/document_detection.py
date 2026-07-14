"""
Document Detection (pipeline addım 2)
- sənədin kadr daxilində tapılması (bounding box / segmentation)
- perspektiv düzəlişi (perspective warp), kəsilmə (cropping)
"""

from dataclasses import dataclass


@dataclass
class DetectionResult:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    cropped_image: object  # warped/cropped şəkil


def detect_document(image, cfg: dict) -> DetectionResult:
    """configs/model.yaml -> document_detection parametrlərini istifadə edir."""
    raise NotImplementedError


def perspective_warp(image, bbox) -> object:
    """4 künc nöqtəsinə əsasən perspektiv düzəlişi edir."""
    raise NotImplementedError
