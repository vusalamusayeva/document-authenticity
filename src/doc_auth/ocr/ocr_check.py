"""
OCR + font, ölçü, düzülüş yoxlanması (pipeline addım 4b)
Diqqət: OCR preprocessing forensic preprocessing-dən ayrı saxlanılmalıdır —
forensic analiz üçün original rəng/noise/compression izləri qorunmalıdır,
OCR üçün isə şəkil azad şəkildə dəyişdirilə bilər.
"""

from dataclasses import dataclass


@dataclass
class OCRResult:
    extracted_text: dict  # field_name -> text
    font_consistency_score: float
    layout_consistency_score: float


def run_ocr(image, cfg: dict) -> OCRResult:
    """configs/model.yaml -> feature_extraction.ocr istifadə edir (məs. PaddleOCR)."""
    raise NotImplementedError


def check_font_consistency(extracted_fields: dict) -> float:
    raise NotImplementedError


def check_layout(extracted_fields: dict, template: dict) -> float:
    raise NotImplementedError
