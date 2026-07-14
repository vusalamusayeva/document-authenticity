"""
ELA, noise-map, CNN-based tampering detection (pipeline addım 4a)
Tamper növləri: copy-move, splicing, text-edit, recapture, full-synthetic.
"""

from dataclasses import dataclass

from doc_auth.utils.base_model import BaseDocModel


@dataclass
class TamperingResult:
    tamper_score: float
    tamper_type: str | None
    heatmap: object  # ELA / noise-map nəticəsi, explainability üçün istifadə olunur


def compute_ela(image, quality: int = 90) -> object:
    """Error Level Analysis — sıxılma səviyyəsi fərqlərini aşkarlayır."""
    raise NotImplementedError


def compute_noise_map(image) -> object:
    raise NotImplementedError


class TamperingCNN(BaseDocModel):
    """configs/model.yaml -> feature_extraction.tampering_cnn"""

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self._model = None

    def fit(self, X, y=None, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError
