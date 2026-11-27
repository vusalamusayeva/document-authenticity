from pathlib import Path
import cv2
import numpy as np

from doc_auth.deepfake.frame_sampler import (
    FrameSampler,
    FrameSamplingConfig,
    sample_video_frames,
)


def _make_video(video_path: Path, num_frames: int = 10) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (128, 128))
    for i in range(num_frames):
        frame = np.full((128, 128, 3), 150 + (i * 7) % 80, dtype=np.uint8)
        cv2.rectangle(frame, (20, 20), (108, 108), (50, 90, 170), 2)
        writer.write(frame)
    writer.release()
    return video_path


def test_frame_sampler_and_helper(tmp_path: Path) -> None:
    vid_path = _make_video(tmp_path / "sample.mp4", num_frames=10)
    cfg = FrameSamplingConfig()
    sampler = FrameSampler(config=cfg)

    sampled = list(sampler.sample(vid_path))
    assert len(sampled) >= 1

    via_call = list(sampler(vid_path))
    assert len(via_call) >= 1

    via_frames = list(sampler.sample_frames(vid_path))
    assert len(via_frames) >= 1

    direct_frames = list(sample_video_frames(vid_path))
    assert len(direct_frames) >= 1
