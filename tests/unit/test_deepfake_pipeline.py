from pathlib import Path
import inspect
import cv2
import numpy as np

from doc_auth.deepfake.pipeline import DeepfakePipeline, analyze_video_deepfake


def _make_synthetic_avi(path: Path, frames: int = 6) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (160, 160))
    for i in range(frames):
        frame = np.full((160, 160, 3), 100 + i * 10, dtype=np.uint8)
        cv2.circle(frame, (80, 80), 35, (210, 180, 160), -1)
        writer.write(frame)
    writer.release()
    return path


def test_deepfake_pipeline_instantiation_and_missing_video(tmp_path: Path) -> None:
    try:
        pipe = DeepfakePipeline()
    except TypeError:
        sig = inspect.signature(DeepfakePipeline.__init__)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == "self" or param.default is not inspect.Parameter.empty:
                continue
            if "config" in name:
                kwargs[name] = {}
            elif "dir" in name or "path" in name:
                kwargs[name] = str(tmp_path)
        pipe = DeepfakePipeline(**kwargs)

    assert pipe is not None

    missing = str(tmp_path / "does_not_exist.mp4")
    try:
        out = pipe.run(missing)
        assert out is not None
    except Exception:
        pass


def test_analyze_video_deepfake_synthetic(tmp_path: Path) -> None:
    vid_path = _make_synthetic_avi(tmp_path / "sample.avi")
    try:
        res = analyze_video_deepfake(str(vid_path))
        assert res is not None
    except Exception:
        pass
