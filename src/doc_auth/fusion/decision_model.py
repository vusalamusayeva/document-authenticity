"""
Fusion / Decision Model
- hər modulun (ELA/CNN, OCR, security-element, metadata) skorlarını birləşdirir
- çıxış: authentic / suspicious / forged + confidence balı
"""

from typing import Any

from doc_auth.utils.base_model import BaseDocModel

LABELS = ["authentic", "suspicious", "forged"]


class FusionDecisionModel(BaseDocModel):
    """
    sklearn-vari fit/predict interfeysinə uyğun fusion modeli.
    Konkret alqoritm (məs. GradientBoosting) configs/model.yaml-dan oxunur.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self._model = None  # underlying estimator burada saxlanılır

    def fit(self, X: Any, y: Any = None, **kwargs) -> "FusionDecisionModel":
        raise NotImplementedError(
            "Alt modüllərin (ELA, OCR, security, metadata) skorlarını "
            "feature vektoruna çevirib estimator-u öyrədin."
        )

    def predict(self, X: Any) -> Any:
        raise NotImplementedError

    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "FusionDecisionModel":
        raise NotImplementedError
