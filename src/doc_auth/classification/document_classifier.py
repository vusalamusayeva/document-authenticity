"""
Document Type Classification (pipeline addım 3)
- sənəd növünün müəyyən edilməsi (ID kart, pasport, sürücülük vəsiqəsi, sertifikat və s.)
"""

from doc_auth.utils.base_model import BaseDocModel


class DocumentClassifier(BaseDocModel):
    """configs/model.yaml -> document_classification parametrlərinə uyğun."""

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self._model = None

    def fit(self, X, y=None, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError
