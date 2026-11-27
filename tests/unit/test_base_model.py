from pathlib import Path
import pytest

from doc_auth.utils.base_model import BaseDocModel


class _MinimalDocModel(BaseDocModel):
    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        return X


class _DummyDocModel(BaseDocModel):
    def __init__(self) -> None:
        self.fitted = False

    def fit(self, X, y=None, **kwargs):
        self.fitted = True
        return self

    def predict(self, X):
        return ["valid" for _ in range(len(X))]

    def predict_proba(self, X):
        return [[0.1, 0.9] for _ in range(len(X))]

    def save(self, path: str) -> None:
        Path(path).write_text("fitted=" + str(self.fitted), encoding="utf-8")

    @classmethod
    def load(cls, path: str):
        obj = cls()
        obj.fitted = Path(path).read_text(encoding="utf-8").endswith("True")
        return obj


def test_base_doc_model_abstract_and_defaults(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        BaseDocModel()  # type: ignore[abstract]

    minimal = _MinimalDocModel()
    assert minimal.fit(["a"]) is minimal
    assert minimal.predict(["a"]) == ["a"]
    with pytest.raises(NotImplementedError):
        minimal.predict_proba(["a"])
    with pytest.raises(NotImplementedError):
        minimal.save(str(tmp_path / "m.bin"))
    with pytest.raises(NotImplementedError):
        _MinimalDocModel.load(str(tmp_path / "m.bin"))


def test_base_doc_model_concrete_subclass(tmp_path: Path) -> None:
    model = _DummyDocModel()
    assert not model.fitted
    model.fit(["doc1", "doc2"], [1, 1])
    assert model.fitted
    assert model.predict(["doc1"]) == ["valid"]
    assert len(model.predict_proba(["doc1"])) == 1

    artifact = tmp_path / "model.state"
    model.save(str(artifact))
    restored = _DummyDocModel.load(str(artifact))
    assert restored.fitted is True
import pytest
from doc_auth.utils.base_model import BaseDocModel


class _ConcreteDocModel(BaseDocModel):
    def fit(self, X, y=None, **kwargs):
        return super().fit(X, y, **kwargs)

    def predict(self, X):
        return X


def test_base_doc_model_super_fit_raises_not_implemented() -> None:
    model = _ConcreteDocModel()
    with pytest.raises(NotImplementedError):
        model.fit([[1.0]], [1])
