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

    # 1. ffprobe raises FileNotFoundError -> probe_unavailable
    def _probe_missing(*args, **kwargs):
        raise FileNotFoundError("ffprobe not found")

    monkeypatch.setattr(subprocess, "run", _probe_missing)
    res_unavail = analyzer.analyze(video)
    assert res_unavail["status"] == "probe_unavailable"

    # 2. ffprobe non-zero returncode
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 1, "", "probe failed"),
    )
    assert "_probe_error" in analyzer._probe_streams(video)

    # 3. ffprobe invalid JSON
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "{bad-json", ""),
    )
    assert "_probe_error" in analyzer._probe_streams(video)

    # 4. _extract_pcm_rms success + empty frames + wave.Error + unlink OSError
    wav_data = _make_wav_bytes(np.array([1000, -1000, 1000, -1000], dtype=np.int16))

    def _ffmpeg_ok(cmd, *args, **kwargs):
        out_path = Path(cmd[-1])
        out_path.write_bytes(wav_data)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _ffmpeg_ok)
    rms = analyzer._extract_pcm_rms(video)
    assert isinstance(rms, float) and rms > 0.0

    # Empty wav frames -> None
    empty_wav = _make_wav_bytes(np.array([], dtype=np.int16))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *a, **kw: (Path(cmd[-1]).write_bytes(empty_wav), subprocess.CompletedProcess(cmd, 0, "", ""))[1],
    )
    assert analyzer._extract_pcm_rms(video) is None

    # wave.Error + unlink OSError
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

    # 1. No audio stream
    probe_no_audio = json.dumps({
        "streams": [{"codec_type": "video", "duration": "invalid"}],
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

    # 2. Audio present with duration mismatch and very low RMS (< 1e-4)
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
