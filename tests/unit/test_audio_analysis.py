import io
import json
from pathlib import Path
import subprocess
import wave

import numpy as np
import pytest

from doc_auth.deepfake.audio_analysis import AudioLipSyncAnalyzer


def _make_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


def test_audio_analyzer_init_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AudioLipSyncAnalyzer(sample_rate=0)

    analyzer = AudioLipSyncAnalyzer()
    res = analyzer.analyze(tmp_path / "missing.mp4")
    assert res["status"] == "file_not_found"
    assert res["has_audio"] is False


def test_audio_analyzer_probe_and_pcm_error_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    analyzer = AudioLipSyncAnalyzer(max_duration_diff_sec=0.5)

    def _probe_missing(*args, **kwargs):
        raise FileNotFoundError("ffprobe not found")

    monkeypatch.setattr(subprocess, "run", _probe_missing)
    res_unavail = analyzer.analyze(video)
    assert res_unavail["status"] == "probe_unavailable"

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 1, "", "probe failed"),
    )
    assert "_probe_error" in analyzer._probe_streams(video)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "{bad-json", ""),
    )
    assert "_probe_error" in analyzer._probe_streams(video)

    wav_data = _make_wav_bytes(np.array([1000, -1000, 1000, -1000], dtype=np.int16))

    def _ffmpeg_ok(cmd, *args, **kwargs):
        out_path = Path(cmd[-1])
        out_path.write_bytes(wav_data)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _ffmpeg_ok)
    rms = analyzer._extract_pcm_rms(video)
    assert isinstance(rms, float) and rms > 0.0

    empty_wav = _make_wav_bytes(np.array([], dtype=np.int16))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *a, **kw: (Path(cmd[-1]).write_bytes(empty_wav), subprocess.CompletedProcess(cmd, 0, "", ""))[1],
    )
    assert analyzer._extract_pcm_rms(video) is None

    orig_unlink = Path.unlink

    def _bad_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.suffix == ".wav":
            raise OSError("busy")
        orig_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _bad_unlink)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *a, **kw: (Path(cmd[-1]).write_bytes(b"not-a-wav"), subprocess.CompletedProcess(cmd, 0, "", ""))[1],
    )
    assert analyzer._extract_pcm_rms(video) is None
    monkeypatch.setattr(Path, "unlink", orig_unlink)


def test_audio_analyzer_no_audio_and_risk_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")
    analyzer = AudioLipSyncAnalyzer(max_duration_diff_sec=0.5)

    probe_no_audio = json.dumps({
        "streams": [{"codec_type": "video", "duration": "4.0"}],
        "format": {"duration": "4.0"},
    })
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, probe_no_audio, ""),
    )
    res_no_audio = analyzer.analyze(video)
    assert res_no_audio["status"] == "no_audio"
    assert res_no_audio["video_duration"] == pytest.approx(4.0)

    probe_with_audio = json.dumps({
        "streams": [
            {"codec_type": "video", "duration": "10.0"},
            {"codec_type": "audio", "duration": "7.0"},
        ],
        "format": {"duration": "10.0"},
    })
    silent_wav = _make_wav_bytes(np.zeros(160, dtype=np.int16))

    def _run_both(cmd, *args, **kwargs):
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, probe_with_audio, "")
        Path(cmd[-1]).write_bytes(silent_wav)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _run_both)
    res_risk = analyzer.analyze(video)
    assert res_risk["status"] == "success"
    assert res_risk["has_audio"] is True
    assert res_risk["duration_diff"] == pytest.approx(3.0)
    assert res_risk["rms_energy"] == pytest.approx(0.0)
    assert res_risk["is_suspicious"] is True
import json
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

from doc_auth.deepfake import audio_analysis as aa_mod
from doc_auth.deepfake.audio_analysis import AudioLipSyncAnalyzer


def test_audio_extract_pcm_rms_failures_and_empty_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"dummy-media")
    analyzer = AudioLipSyncAnalyzer()

    # 1. ffmpeg returns non-zero exit code -> returns None
    def _ffmpeg_fail(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"partial")
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="ffmpeg error")

    monkeypatch.setattr(aa_mod.subprocess, "run", _ffmpeg_fail)
    assert analyzer._extract_pcm_rms(media) is None

    # 2. ffmpeg creates valid WAV header with 0 frames -> samples.size == 0 -> returns None
    def _ffmpeg_zero_frames(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        out_path = Path(cmd[-1])
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(aa_mod.subprocess, "run", _ffmpeg_zero_frames)
    assert analyzer._extract_pcm_rms(media) is None


def test_audio_analyze_invalid_video_and_audio_durations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "bad_dur.mp4"
    media.write_bytes(b"dummy-media")
    analyzer = AudioLipSyncAnalyzer()

    probe_payload = {
        "streams": [
            {"codec_type": "video", "duration": "not-a-float"},
            {"codec_type": "audio", "duration": "also-invalid"},
        ],
        "format": {"duration": "bad-format-dur"},
    }
    monkeypatch.setattr(analyzer, "_probe_streams", lambda _p: probe_payload)
    monkeypatch.setattr(analyzer, "_extract_pcm_rms", lambda _p: 0.05)

    res = analyzer.analyze(media)
    assert res["status"] == "success"
    assert res["has_audio"] is True
    assert res["video_duration"] is None
    assert res["audio_duration"] is None
    assert res["duration_diff"] is None
from pathlib import Path
import subprocess
from typing import Any
import wave
import pytest

from doc_auth.deepfake import audio_analysis as aa_mod
from doc_auth.deepfake.audio_analysis import AudioLipSyncAnalyzer


def test_extract_pcm_rms_returns_none_on_zero_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src_file = tmp_path / "silent.mp4"
    src_file.write_bytes(b"video")

    class _ZeroWave:
        def __enter__(self) -> "_ZeroWave":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def getnframes(self) -> int:
            return 0

        def readframes(self, _n: int) -> bytes:
            return b""

    def _fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        out_wav = Path(cmd[-1])
        out_wav.write_bytes(b"RIFF0000WAVEfmt ")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(aa_mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(wave, "open", lambda *_a, **_kw: _ZeroWave())

    analyzer = AudioLipSyncAnalyzer()
    assert analyzer._extract_pcm_rms(src_file) is None
from pathlib import Path
import subprocess
import wave
import pytest

from doc_auth.deepfake import audio_analysis as aa_mod


def test_extract_pcm_rms_returns_none_when_samples_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_file = tmp_path / "sample.mp4"
    src_file.write_bytes(b"video")

    class _DummyProc:
        returncode = 0
        stdout = ""
        stderr = ""

    class _DummyWave:
        def __enter__(self) -> "_DummyWave":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def getnframes(self) -> int:
            return 1

        def readframes(self, _n: int) -> bytes:
            return b"\x00"

    def _fake_run(cmd: list[str], **_kwargs: object) -> _DummyProc:
        Path(cmd[-1]).write_bytes(b"RIFF")
        return _DummyProc()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(wave, "open", lambda *_args, **_kwargs: _DummyWave())

    analyzer = aa_mod.AudioLipSyncAnalyzer()
    assert analyzer._extract_pcm_rms(src_file) is None
