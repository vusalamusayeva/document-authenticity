from pathlib import Path
import pytest

from doc_auth.utils.base_model import BaseDocModel


class _DummyDocModel(BaseDocModel):
    def __init__(self) -> None:
        self.fitted = False

    def fit(self, x, y=None):
        self.fitted = True
        return self

    def predict(self, x):
        return ["valid" for _ in range(len(x))]

    def predict_proba(self, x):
        return [[0.1, 0.9] for _ in range(len(x))]

    def save(self, path) -> None:
        Path(path).write_text("fitted=" + str(self.fitted), encoding="utf-8")

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.fitted = Path(path).read_text(encoding="utf-8").endswith("True")
        return obj


def test_base_doc_model_abstract_and_concrete(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        BaseDocModel()  # type: ignore[abstract]

    model = _DummyDocModel()
    assert not model.fitted
    model.fit(["doc1", "doc2"], [1, 1])
    assert model.fitted
    assert model.predict(["doc1"]) == ["valid"]
    assert len(model.predict_proba(["doc1"])) == 1

    artifact = tmp_path / "model.state"
    model.save(artifact)
    restored = _DummyDocModel.load(artifact)
    assert restored.fitted is True
