"""
Explainability Layer (pipeline addım 6)
- hansı bölgənin/əlamətin şübhəli olduğunu göstərən mask
"""

from dataclasses import dataclass


@dataclass
class ExplanationResult:
    mask: object  # şübhəli bölgələri göstərən piksel-səviyyəli mask
    top_contributing_modules: list[str]  # məs. ["forensics", "ocr"]


def generate_explanation(module_outputs: dict, cfg: dict) -> ExplanationResult:
    """
    module_outputs: forensics/ocr/signature_stamp/metadata modullarının nəticələri.
    configs/model.yaml -> explainability.method (məs. grad_cam) istifadə olunur.
    """
    raise NotImplementedError
