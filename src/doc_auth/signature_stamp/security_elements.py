"""
Security-element yoxlanması (pipeline addım 4c): imza, möhür/ştamp,
hologram, mikroyazı və digər təhlükəsizlik elementlərinin doğrulanması.

Signature təhlili üçün CEDAR kimi dataset-lər istifadə oluna bilər (bax configs/data.yaml).
"""

from dataclasses import dataclass

from doc_auth.utils.base_model import BaseDocModel


@dataclass
class SecurityElementResult:
    signature_match_score: float | None
    stamp_present: bool
    stamp_authenticity_score: float | None
    hologram_detected: bool | None


class SignatureVerifier(BaseDocModel):
    """İmza saxtakarlığı aşkarlanması (məs. CEDAR dataset ilə öyrədilə bilər)."""

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self._model = None

    def fit(self, X, y=None, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


def detect_stamp(image) -> dict:
    raise NotImplementedError
