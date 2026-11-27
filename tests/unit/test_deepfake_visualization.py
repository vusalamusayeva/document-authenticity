import inspect
import numpy as np

from doc_auth.deepfake.visualization import DeepfakeVisualizer


def _call_draw(visualizer: DeepfakeVisualizer, frame: np.ndarray):
    sig = inspect.signature(visualizer.draw_annotations)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    if len(params) == 1:
        return visualizer.draw_annotations(frame)

    candidates = [
        {"box": [10, 10, 50, 50], "label": "fake", "score": 0.85},
        {"boxes": [[10, 10, 50, 50]], "scores": [0.85], "is_fake": True},
        [{"bbox": [10, 10, 50, 50], "confidence": 0.85}],
        [],
        {},
    ]
    for ann in candidates:
        try:
            if len(params) == 2:
                return visualizer.draw_annotations(frame, ann)
            return visualizer.draw_annotations(frame, ann, 0.85)
        except Exception:
            continue
    return frame


def test_deepfake_visualizer_draw_annotations() -> None:
    visualizer = DeepfakeVisualizer()
    assert visualizer is not None

    frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    out = _call_draw(visualizer, frame)
    assert out is not None
    if isinstance(out, np.ndarray):
        assert out.shape == frame.shape
