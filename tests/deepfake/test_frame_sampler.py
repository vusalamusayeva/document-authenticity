import pytest
from doc_auth.deepfake.frame_sampler import FrameSampler
from doc_auth.deepfake.exceptions import VideoLoadError

def test_frame_sampler_invalid_path():
    sampler = FrameSampler()
    with pytest.raises(VideoLoadError):
        for _ in sampler.sample_frames("invalid_video_path.mp4"):
            pass