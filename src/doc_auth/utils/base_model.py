"""
Bütün model class-ları üçün sklearn-vari fit/predict interfeysi.
Bu interfeys sayəsində modullar bir-birini asanlıqla əvəz edə bilər.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseDocModel(ABC):
    """Bütün doc_auth modellərinin əsas sinfi."""

    @abstractmethod
    def fit(self, X: Any, y: Any = None, **kwargs) -> "BaseDocModel":
        """Modeli öyrədir. Chaining üçün self qaytarır."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Proqnoz qaytarır."""
        raise NotImplementedError

    def predict_proba(self, X: Any) -> Any:
        """Ehtimal/skor qaytarır (mövcud olduqda override edilməlidir)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} predict_proba dəstəkləmir"
        )

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BaseDocModel":
        raise NotImplementedError
