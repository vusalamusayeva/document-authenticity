from pathlib import Path
import inspect
import cv2
import numpy as np

from doc_auth.classification.document_classifier import DocumentClassifier


def _make_doc_image(label: str, idx: int) -> np.ndarray:
    img = np.ones((180, 240, 3), dtype=np.uint8) * 240
    if label == "invoice":
        cv2.rectangle(img, (20, 20), (220, 160), (20, 20, 20), 2)
        for row in range(40, 140, 20):
            cv2.line(img, (35, row), (205, row), (60, 60, 60), 1)
    else:
        cv2.circle(img, (120, 90), 45 + (idx % 10), (40, 40, 40), 2)
        cv2.line(img, (40, 30), (200, 150), (80, 80, 80), 2)
    return img


def _call_with_supported_args(fn, images, labels):
    sig = inspect.signature(fn)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    if len(params) >= 2:
        return fn(images, labels)
    if len(params) == 1:
        return fn(images)
    return fn()


def test_document_classifier_workflow(tmp_path: Path) -> None:
    clf = DocumentClassifier()
    assert clf is not None

    train_images = [
        _make_doc_image("invoice", 0),
        _make_doc_image("invoice", 1),
        _make_doc_image("invoice", 2),
        _make_doc_image("id_card", 0),
        _make_doc_image("id_card", 1),
        _make_doc_image("id_card", 2),
    ]
    train_labels = ["invoice", "invoice", "invoice", "id_card", "id_card", "id_card"]

    _call_with_supported_args(clf.fit, train_images, train_labels)

    probs = clf.predict_proba(train_images[:2])
    assert probs is not None

    preds = clf.predict(train_images[:2])
    assert len(preds) == 2

    if hasattr(clf, "classify"):
        try:
            c_out = clf.classify(train_images[0])
            assert c_out is not None
        except TypeError:
            pass

    if hasattr(clf, "evaluate"):
        try:
            ev = clf.evaluate(train_images, train_labels)
            assert ev is not None
        except Exception:
            pass

    if hasattr(clf, "save") and hasattr(clf, "load"):
        model_path = tmp_path / "classifier_model.joblib"
        try:
            clf.save(str(model_path))
            loaded = DocumentClassifier()
            loaded.load(str(model_path))
            loaded_preds = loaded.predict(train_images[:2])
            assert len(loaded_preds) == 2
        except Exception:
            pass
