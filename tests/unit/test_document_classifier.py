from pathlib import Path
import pickle

import cv2
import numpy as np
import pytest

from doc_auth.classification.document_classifier import DocumentClassifier


def test_init_resolve_image_and_feature_extraction(tmp_path: Path) -> None:
    clf = DocumentClassifier({"classes": ["passport", "id_card"], "min_confidence": 0.6})
    assert clf.classes[-1] == "unknown"
    assert clf.min_confidence == pytest.approx(0.6)

    # _resolve_image with ndarray, dict image key, dict path key, and disk file
    img = np.full((32, 48, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (4, 4), (44, 28), (40, 40, 40), 2)
    assert clf._resolve_image(np.array([], dtype=np.uint8)) is None
    assert clf._resolve_image({"image": img}) is not None
    assert clf._resolve_image({"frame": img}) is not None

    img_path = tmp_path / "doc_sample.png"
    assert cv2.imwrite(str(img_path), img)
    resolved_from_dict_path = clf._resolve_image({"document_path": str(img_path)})
    assert isinstance(resolved_from_dict_path, np.ndarray)
    assert resolved_from_dict_path.shape == (32, 48, 3)

    # _extract_feature_vector with explicit features dict, image, and unresolvable input
    feat_vec = clf._extract_feature_vector({"features": [0.5, 0.2, 1.0, 0.3]})
    assert feat_vec.shape == (8,)
    assert feat_vec[0] == pytest.approx(0.5)

    img_vec = clf._extract_feature_vector(img)
    assert img_vec.shape == (8,)
    assert np.all(np.isfinite(img_vec))

    zero_vec = clf._extract_feature_vector("nonexistent_file.png")
    assert np.array_equal(zero_vec, np.zeros(8, dtype=np.float32))


def test_fit_errors_predict_classify_evaluate_and_persistence(tmp_path: Path) -> None:
    clf = DocumentClassifier({"min_confidence": 0.99})

    # Unfitted checks
    with pytest.raises(RuntimeError):
        clf.predict_proba([{"features": [0.1] * 8}])
    with pytest.raises(RuntimeError):
        clf.evaluate([{"features": [0.1] * 8}], ["a"])

    # classify before fit returns inconclusive when input is valid
    pre_fit = clf.classify({"features": [0.1] * 8})
    assert pre_fit["status"] == "inconclusive"
    assert pre_fit["document_type"] == "unknown"

    # classify with missing image/features returns not_applicable
    na_res = clf.classify({"other": 123})
    assert na_res["status"] == "not_applicable"
    assert na_res["document_type"] is None

    # fit validation errors
    with pytest.raises(ValueError):
        clf.fit(None, ["a"])
    with pytest.raises(ValueError):
        clf.fit([], [])
    with pytest.raises(ValueError):
        clf.fit([{"features": [0.1] * 8}], None)
    with pytest.raises(ValueError):
        clf.fit([{"features": [0.1] * 8}, {"features": [0.2] * 8}], ["only_one"])
    with pytest.raises(ValueError):
        clf.fit([{"features": [0.1] * 8}], ["a", "b"])

    # Fit on distinct feature profiles
    x_train = [
        {"features": [1.0, 0.8, 0.7, 0.9, 0.2, 0.8, 0.8, 1.0]},
        {"features": [1.1, 0.85, 0.75, 0.88, 0.22, 0.85, 0.85, 1.0]},
        {"features": [-1.0, 0.1, 0.15, 0.3, 0.5, 0.1, 0.1, 0.9]},
        {"features": [-1.1, 0.12, 0.18, 0.28, 0.52, 0.12, 0.12, 0.9]},
    ]
    y_train = ["passport", "passport", "id_card", "id_card"]
    assert clf.fit(x_train, y_train) is clf

    # predict_proba edge inputs: empty list, 3D ndarray, single dict
    empty_probs = clf.predict_proba([])
    assert empty_probs.shape == (0, len(clf.classes))

    img_3d = np.full((24, 32, 3), 150, dtype=np.uint8)
    probs_3d = clf.predict_proba(img_3d)
    assert probs_3d.shape == (1, len(clf.classes))

    probs_dict = clf.predict_proba({"features": [1.0, 0.8, 0.7, 0.9, 0.2, 0.8, 0.8, 1.0]})
    assert probs_dict.shape == (1, len(clf.classes))

    with pytest.raises(ValueError):
        clf.predict_proba(None)

    # predict with high min_confidence (0.99) yields "unknown"
    assert clf.predict([]) == []
    assert clf.predict(x_train[:1]) == ["unknown"]
    low_conf_class = clf.classify(x_train[0])
    assert low_conf_class["status"] == "inconclusive"
    assert low_conf_class["document_type"] == "unknown"

    # Lower min_confidence to verify confident prediction & evaluate
    clf.min_confidence = 0.0
    preds = clf.predict(x_train)
    assert preds == y_train
    confident_class = clf.classify(x_train[0])
    assert confident_class["status"] == "success"
    assert confident_class["document_type"] in clf.classes

    metrics = clf.evaluate(x_train, y_train)
    assert "labels" in metrics and "confusion_matrix" in metrics and "per_class" in metrics

    # save and load + wrong type check
    saved_path = tmp_path / "nested" / "clf.pkl"
    clf.save(str(saved_path))
    loaded = DocumentClassifier.load(str(saved_path))
    assert isinstance(loaded, DocumentClassifier)
    assert loaded.predict(x_train) == y_train

    wrong_type_path = tmp_path / "wrong.pkl"
    with open(wrong_type_path, "wb") as f:
        pickle.dump({"not": "a classifier"}, f)
    with pytest.raises(TypeError):
        DocumentClassifier.load(str(wrong_type_path))
from pathlib import Path
import numpy as np
import pytest

from doc_auth.classification.document_classifier import DocumentClassifier


def test_document_classifier_single_class_fit_and_single_path_predict_proba(
    tmp_path: Path,
) -> None:
    clf = DocumentClassifier()
    x_items = [{"features": [1.0, 0.2]}, {"features": [2.0, 0.8]}]

    with pytest.raises(ValueError):
        clf.fit(x_items, ["id_card", "id_card"])

    clf.fit(x_items, ["id_card", "passport"])
    img_path = tmp_path / "sample.png"
    arr = np.full((32, 32, 3), 200, dtype=np.uint8)
    import cv2

    cv2.imwrite(str(img_path), arr)

    probs = clf.predict_proba(img_path)
    assert probs.shape == (1, len(clf.classes))
