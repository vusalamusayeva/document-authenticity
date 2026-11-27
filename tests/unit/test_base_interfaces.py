from typing import Any, Dict
import numpy as np
import pytest

from doc_auth.deepfake.base import BaseDeepfakeDetector
from doc_auth.utils.base_model import BaseDocModel


class _DummyDetector(BaseDeepfakeDetector):
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        return super().analyze_frame(frame)


class _DummyModel(BaseDocModel):
    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "BaseDocModel":
        return self

    def predict(self, X: Any) -> Any:
        return super().predict(X)


def test_base_deepfake_detector_interface() -> None:
    with pytest.raises(TypeError):
        BaseDeepfakeDetector()  # type: ignore[abstract]

    det = _DummyDetector()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    assert det.analyze_frame(frame) is None


def test_base_doc_model_interface() -> None:
    with pytest.raises(TypeError):
        BaseDocModel()  # type: ignore[abstract]

    model = _DummyModel()
    assert model.fit([1, 2]) is model

    with pytest.raises(NotImplementedError):
        model.predict([1, 2])

    with pytest.raises(NotImplementedError):
        model.predict_proba([1, 2])

    with pytest.raises(NotImplementedError):
        model.save("model.pt")

    with pytest.raises(NotImplementedError):
        BaseDocModel.load("model.pt")
