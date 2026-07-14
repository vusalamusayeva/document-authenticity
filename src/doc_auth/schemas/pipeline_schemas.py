"""
Pipeline daxili modullar arasında paylaşılan data strukturları.
Qeyd: bunlar api/schemas/-dəki API request/response modellərindən fərqlidir —
o hissə HTTP giriş/çıxışı üçündür, bura isə modullar arası daxili kontraktdır.
"""

from dataclasses import dataclass, field


@dataclass
class PipelineContext:
    """Bir sənədin pipeline boyu daşınan vəziyyəti."""

    image: object
    document_type: str | None = None
    quality_ok: bool = True
    detection_bbox: tuple[float, float, float, float] | None = None
    module_scores: dict = field(default_factory=dict)  # {"forensics": 0.2, "ocr": 0.1, ...}
    final_label: str | None = None  # authentic / suspicious / forged
    confidence: float | None = None
    explanation_mask: object = None


@dataclass
class LabelSchema:
    """Bölmə 5: labeling qaydalarına uyğun label strukturu."""

    document_level: str  # authentic / forged / suspicious
    tamper_type: str | None  # copy-move, splicing, text-edit, recapture, full-synthetic, signature-forgery
    difficulty: str | None  # easy / medium / hard
    region_mask_path: str | None = None
