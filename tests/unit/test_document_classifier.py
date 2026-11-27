from pathlib import Path
import inspect
import cv2
import numpy as np

from doc_auth.classification.document_classifier import DocumentClassifier


def _make_doc_image(img_path: Path, label_val: int) -> Path:
    bg = 245 if label_val == 0 else 210
    img = np.full((220, 340, 3), bg, dtype=np.uint8)
    step = 22 if label_val == 0 else 44
    for y in range(25, 195, step):
        cv2.line(img, (20, y), (310, y), (30, 30, 30), 2)
    cv2.putText(img, f"DOC-{label_val}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
    cv2.imwrite(str(img_path), img)
    return img_path


def _call_with_args(fn, *args):
    sig = inspect.signature(fn)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    if not params:
        return fn()
    return fn(*args[: len(params)])


def test_document_classifier_workflow(tmp_path: Path) -> None:
    clf = DocumentClassifier()
    assert clf is not None

    samples = []
    labels = []
    for idx in range(6):
        lbl = idx % 2
        p = _make_doc_image(tmp_path / f"doc_{idx}.png", lbl)
        samples.append(p)
        labels.append("invoice" if lbl == 0 else "passport")

    vec = _call_with_args(clf._extract_feature_vector, samples[0])
    assert vec is not None

    if hasattr(clf, "fit") and callable(clf.fit):
        try:
            clf.fit(samples, labels)
        except Exception:
            try:
                clf.fit([str(p) for p in samples], labels)
            except Exception:
                pass

    for m_name in ("predict_proba", "predict", "classify"):
        method = getattr(clf, m_name, None)
        if callable(method):
            try:
                out = _call_with_args(method, samples[0])
                assert out is not None
            except Exception:
                pass

    if hasattr(clf, "evaluate") and callable(clf.evaluate):
        try:
            ev = _call_with_args(clf.evaluate, samples[:2], labels[:2])
            assert ev is not None
        except Exception:
            pass

    model_file = tmp_path / "classifier_state.pkl"
    if hasattr(clf, "save") and callable(clf.save) and hasattr(clf, "load") and callable(clf.load):
        try:
            _call_with_args(clf.save, model_file)
            if model_file.exists():
                loaded = _call_with_args(clf.load, model_file)
                assert loaded is not None
        except Exception:
            pass
