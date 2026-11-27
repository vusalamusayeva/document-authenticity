import numpy as np

from doc_auth.deepfake.visualization import DeepfakeVisualizer


def test_draw_annotations_high_and_low_scores() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    original = frame.copy()
    viz = DeepfakeVisualizer()

    result = {
        "faces": [
            {"bbox": [10, 20, 40, 50], "deepfake_score": 0.85},
            {"bbox": [70, 5, 30, 30], "deepfake_score": 0.20},
            {"bbox": [1, 2], "deepfake_score": 0.90},
        ]
    }

    annotated = viz.draw_annotations(frame, result)
    assert annotated.shape == frame.shape
    assert np.array_equal(frame, original)
    assert not np.array_equal(annotated, frame)
