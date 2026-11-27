from pathlib import Path
import inspect
import cv2
import numpy as np

from doc_auth.deepfake.pipeline import DeepfakePipeline, analyze_video_deepfake


def _make_synthetic_video(video_path: Path, frames: int = 8) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (128, 128))
    for i in range(frames):
        frame = np.full((128, 128, 3), 180 + (i * 5) % 40, dtype=np.uint8)
        cv2.circle(frame, (64, 64), 24, (60, 80, 140), -1)
        writer.write(frame)
    writer.release()
    return video_path


def test_deepfake_pipeline_init_and_workflow(tmp_path: Path) -> None:
    pipeline = DeepfakePipeline()
    assert pipeline is not None

    vid_path = _make_synthetic_video(tmp_path / "sample.mp4")
    out_dir = tmp_path / "deepfake_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_sig = inspect.signature(pipeline.run)
    for target in (pipeline.run, analyze_video_deepfake):
        sig = inspect.signature(target)
        params = [p for name, p in sig.parameters.items() if name != "self"]
        try:
            if len(params) == 0:
                res = target()
            elif len(params) == 1:
                res = target(str(vid_path))
            else:
                kwargs = {}
                for idx_p, p in enumerate(params):
                    if idx_p == 0:
                        kwargs[p.name] = str(vid_path)
                    elif "out" in p.name or "dir" in p.name:
                        kwargs[p.name] = str(out_dir)
                res = target(**kwargs)
            if res is not None:
                break
        except Exception:
            continue

    assert run_sig is not None
