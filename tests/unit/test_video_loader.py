import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from doc_auth.deepfake.exceptions import InvalidConfigError, VideoLoadError, VideoValidationError
from doc_auth.deepfake.video_loader import (
    VideoLoaderConfig,
    VideoStreamReader,
    probe_video_metadata,
    validate_video,
)


class _DummyCapture:
    def __init__(self, opened: bool = True, props: dict | None = None, frames: list | None = None) -> None:
        self._opened = opened
        self._props = props or {}
        self._frames = list(frames or [])
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop_id: int) -> float:
        return float(self._props.get(prop_id, 0.0))

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


def test_probe_video_metadata_ffprobe_and_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")
    with pytest.raises(VideoLoadError):
        probe_video_metadata(empty_file)

    with pytest.raises(VideoLoadError):
        probe_video_metadata(tmp_path / "missing.mp4")

    vid = tmp_path / "sample.mp4"
    vid.write_bytes(b"fake-video-bytes")

    # ffprobe not found -> raises VideoLoadError
    def _raise_fnf(*args, **kwargs):
        raise FileNotFoundError("ffprobe missing")

    monkeypatch.setattr(subprocess, "run", _raise_fnf)
    with pytest.raises(VideoLoadError):
        probe_video_metadata(vid)

    # ffprobe non-zero returncode -> fallback to cv2.VideoCapture
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout=""))
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda p: _DummyCapture(
            opened=True,
            props={
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 60,
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
            },
        ),
    )
    meta_fb = probe_video_metadata(vid)
    assert meta_fb["width"] == 640
    assert meta_fb["height"] == 480
    assert meta_fb["fps"] == 30.0
    assert meta_fb["duration"] == pytest.approx(2.0)

    # ffprobe timeout -> fallback raises VideoLoadError when capture fails to open
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1.0)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    monkeypatch.setattr(cv2, "VideoCapture", lambda p: _DummyCapture(opened=False))
    with pytest.raises(VideoLoadError):
        probe_video_metadata(vid, timeout_sec=1.0)

    # ffprobe non-zero returncode with invalid dimensions -> VideoValidationError
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout=""))
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda p: _DummyCapture(opened=True, props={cv2.CAP_PROP_FRAME_WIDTH: 0, cv2.CAP_PROP_FRAME_HEIGHT: 0}),
    )
    with pytest.raises(VideoValidationError):
        probe_video_metadata(vid)


def test_probe_video_metadata_json_parsing_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"video-data")

    # Invalid JSON
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="{bad-json"))
    with pytest.raises(VideoLoadError):
        probe_video_metadata(vid)

    # Missing video stream
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [{"codec_type": "audio"}]})),
    )
    with pytest.raises(VideoValidationError):
        probe_video_metadata(vid)

    # Invalid dimensions in video stream
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"streams": [{"codec_type": "video", "width": 0, "height": 100}]}),
        ),
    )
    with pytest.raises(VideoValidationError):
        probe_video_metadata(vid)

    # Invalid stream duration/nb_frames falls back to 0.0 / 0
    invalid_meta_payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "60/2",
                "duration": "invalid",
                "nb_frames": "invalid",
            }
        ],
        "format": {},
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(invalid_meta_payload)),
    )
    meta_invalid = probe_video_metadata(vid)
    assert meta_invalid["fps"] == 30.0
    assert meta_invalid["duration"] == pytest.approx(0.0)
    assert meta_invalid["frame_count"] == 0

    # Valid stream duration with missing nb_frames computes frame_count from duration * fps
    computed_frames_payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "60/2",
                "duration": "2.5",
            }
        ],
        "format": {},
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(computed_frames_payload)),
    )
    meta = probe_video_metadata(vid)
    assert meta["fps"] == 30.0
    assert meta["duration"] == pytest.approx(2.5)
    assert meta["frame_count"] == 75


def test_validate_video_and_stream_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vid = tmp_path / "stream.mp4"
    vid.write_bytes(b"stream-bytes")

    with pytest.raises(InvalidConfigError):
        validate_video(vid, VideoLoaderConfig(probe_timeout_sec=0.0))

    payload = {
        "streams": [{"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "24", "nb_frames": "24"}]
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload)),
    )

    with pytest.raises(VideoValidationError):
        validate_video(vid, VideoLoaderConfig(max_width=1280))

    with pytest.raises(VideoValidationError):
        validate_video(vid, VideoLoaderConfig(max_height=720))

    reader_unopened = VideoStreamReader(vid)
    assert reader_unopened.cap is None
    with pytest.raises(VideoLoadError):
        reader_unopened.read_frame()

    monkeypatch.setattr(cv2, "VideoCapture", lambda p: _DummyCapture(opened=False))
    with pytest.raises(VideoLoadError):
        with VideoStreamReader(vid):
            pass

    frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
    cap_obj = _DummyCapture(opened=True, frames=[frame1])
    monkeypatch.setattr(cv2, "VideoCapture", lambda p: cap_obj)

    with VideoStreamReader(vid) as reader:
        assert reader.cap is cap_obj
        got = reader.read_frame()
        assert got is not None and got.shape == (10, 10, 3)
        assert reader.read_frame() is None

    assert cap_obj.released is True
    assert reader.cap is None
import json
import subprocess
from pathlib import Path
from typing import Any
import pytest

from doc_auth.deepfake import video_loader as vl_mod
from doc_auth.deepfake.exceptions import VideoLoadError
from doc_auth.deepfake.video_loader import probe_video_metadata


def test_video_loader_unopened_cv2_fallback_and_invalid_fps_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vid = tmp_path / "sample.mp4"
    vid.write_bytes(b"dummy-video-bytes")

    # 1. ffprobe fails -> fallback cv2.VideoCapture fails to open -> VideoLoadError (lines 56-57)
    def _ffprobe_fail(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="probe err")

    class _ClosedCap:
        def __init__(self, _path: str) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(vl_mod.subprocess, "run", _ffprobe_fail)
    monkeypatch.setattr(vl_mod.cv2, "VideoCapture", _ClosedCap)
    with pytest.raises(VideoLoadError):
        probe_video_metadata(vid)

    # 2. Invalid fractional and non-fractional r_frame_rate -> defaults to 25.0 (lines 98-99, 103-104)
    for bad_fps in ("30/not_a_num", "not_a_float"):
        payload = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 480,
                    "r_frame_rate": bad_fps,
                    "duration": "2.0",
                    "nb_frames": "50",
                }
            ],
            "format": {"duration": "2.0"},
        }

        def _ffprobe_ok(
            cmd: list[str], _p: str = json.dumps(payload), **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(cmd, returncode=0, stdout=_p, stderr="")

        monkeypatch.setattr(vl_mod.subprocess, "run", _ffprobe_ok)
        meta = probe_video_metadata(vid)
        assert meta["fps"] == 25.0
