from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from doc_auth.deepfake import frame_sampler as fs_mod
from doc_auth.deepfake.exceptions import InvalidConfigError, VideoValidationError
from doc_auth.deepfake.frame_sampler import FrameSampler, FrameSamplingConfig, sample_video_frames


class _FakeCap:
    def __init__(
        self,
        frames: List[Tuple[bool, Any]],
        pos_msec_seq: List[float] | None = None,
        native_fps: float = 0.0,
        boom_on_read: bool = False,
    ) -> None:
        self._frames = list(frames)
        self._pos_seq = list(pos_msec_seq or [])
        self._native_fps = native_fps
        self._boom_on_read = boom_on_read
        self._idx = 0

    def get(self, prop_id: int) -> float:
        if prop_id == 5:
            return self._native_fps
        if prop_id == 0:
            if self._pos_seq:
                i = min(self._idx - 1, len(self._pos_seq) - 1)
                return self._pos_seq[max(i, 0)]
            return 0.0
        return 0.0

    def read(self) -> Tuple[bool, Any]:
        if self._boom_on_read:
            raise RuntimeError("cap read failure")
        if self._idx >= len(self._frames):
            return False, None
        item = self._frames[self._idx]
        self._idx += 1
        return item


class _FakeReader:
    def __init__(self, cap: _FakeCap, metadata: Dict[str, Any] | None = None) -> None:
        self.cap = cap
        self.metadata = metadata if metadata is not None else {}

    def __enter__(self) -> "_FakeReader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


def test_frame_sampling_config_validation() -> None:
    with pytest.raises(InvalidConfigError):
        FrameSamplingConfig(sampling_fps=True)  # type: ignore[arg-type]
    with pytest.raises(InvalidConfigError):
        FrameSamplingConfig(sampling_fps=0.0)
    with pytest.raises(InvalidConfigError):
        FrameSamplingConfig(max_frames=False)  # type: ignore[arg-type]
    with pytest.raises(InvalidConfigError):
        FrameSamplingConfig(max_frames=0)


def test_sample_video_frames_and_sampler_wrappers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dummy_video = tmp_path / "sample.mp4"
    dummy_video.write_bytes(b"fake")

    f1 = np.zeros((8, 8, 3), dtype=np.uint8)
    f2 = np.ones((8, 8, 3), dtype=np.uint8) * 50
    f3 = np.ones((8, 8, 3), dtype=np.uint8) * 100

    # native_fps <= 0 falls back to 25.0; pos_msec == 0 uses raw_idx / native_fps
    cap = _FakeCap(
        frames=[(True, f1), (True, f2), (True, f3), (False, None)],
        pos_msec_seq=[0.0, 0.0, 0.0],
        native_fps=0.0,
    )
    monkeypatch.setattr(fs_mod, "VideoStreamReader", lambda path, cfg: _FakeReader(cap, metadata={"fps": None}))

    sampler = FrameSampler(sampling_fps=25.0, max_frames=2)
    stream_items = list(sampler.sample_frames(dummy_video))
    assert len(stream_items) == 2
    assert stream_items[0]["sample_index"] == 0
    assert stream_items[1]["sample_index"] == 1

    collected = sampler.sample(dummy_video)
    assert len(collected) == 1

    cap2 = _FakeCap(frames=[(True, f1), (True, f2)], pos_msec_seq=[1500.0, 2500.0], native_fps=30.0)
    monkeypatch.setattr(fs_mod, "VideoStreamReader", lambda path, cfg: _FakeReader(cap2, metadata={"fps": 30.0}))
    via_call = list(sampler(dummy_video))
    assert len(via_call) == 2
    assert via_call[0]["timestamp"] == pytest.approx(1.5)
    assert via_call[1]["timestamp"] == pytest.approx(2.5)

    # Unexpected exception inside reader is wrapped in VideoValidationError
    boom_cap = _FakeCap(frames=[], boom_on_read=True)
    monkeypatch.setattr(fs_mod, "VideoStreamReader", lambda path, cfg: _FakeReader(boom_cap))
    with pytest.raises(VideoValidationError):
        list(sample_video_frames(dummy_video))
from pathlib import Path
from typing import Any
import pytest

from doc_auth.deepfake import frame_sampler as fs_mod
from doc_auth.deepfake.exceptions import InvalidConfigError
from doc_auth.deepfake.frame_sampler import sample_video_frames


def test_sample_video_frames_reraises_invalid_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "dummy.mp4"
    video.write_bytes(b"fake")

    class _DummyCap:
        def get(self, _prop: int) -> float:
            return 25.0

        def read(self) -> tuple[bool, Any]:
            raise InvalidConfigError("bad sampler runtime config")

    class _DummyReader:
        def __init__(self, _path: Any, _cfg: Any) -> None:
            self.cap = _DummyCap()
            self.metadata = {"fps": 25.0}

        def __enter__(self) -> "_DummyReader":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(fs_mod, "VideoStreamReader", _DummyReader)

    with pytest.raises(InvalidConfigError, match="bad sampler runtime config"):
        list(sample_video_frames(video))
