import json
import math
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class AudioLipSyncAnalyzer:
    """
    `ffprobe` ilə audio stream metadata-sını yoxlayır, `ffmpeg` ilə PCM WAV çıxarır
    və real səs enerjisi / müddət uyğunsuzluğu əsasında lip-sync/audio riski hesablayır.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration_diff_sec: float = 0.75,
        timeout_sec: float = 15.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate müsbət olmalıdır.")
        self.sample_rate = int(sample_rate)
        self.max_duration_diff_sec = float(max_duration_diff_sec)
        self.timeout_sec = float(timeout_sec)

    def _probe_streams(self, video_path: Path) -> Dict[str, Any]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return {"_probe_error": str(exc)}

        if proc.returncode != 0 or not proc.stdout:
            return {"_probe_error": proc.stderr.strip() or "ffprobe uğursuz oldu"}

        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            return {"_probe_error": str(exc)}

    def _extract_pcm_rms(self, video_path: Path) -> Optional[float]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "wav",
            str(wav_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
            if proc.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
                return None

            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
            if not frames:
                return None
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size == 0:
                return None
            return float(np.sqrt(np.mean(np.square(samples))))
        except (FileNotFoundError, subprocess.TimeoutExpired, wave.Error, OSError):
            return None
        finally:
            if wav_path.exists():
                try:
                    wav_path.unlink()
                except OSError:
                    pass

    def analyze(self, video_path: str | Path) -> Dict[str, Any]:
        path = Path(video_path)
        if not path.exists() or not path.is_file():
            return {
                "status": "file_not_found",
                "has_audio": False,
                "audio_duration": None,
                "video_duration": None,
                "duration_diff": None,
                "lip_sync_score": None,
                "is_suspicious": None,
            }

        probe = self._probe_streams(path)
        if "_probe_error" in probe:
            return {
                "status": "probe_unavailable",
                "has_audio": None,
                "audio_duration": None,
                "video_duration": None,
                "duration_diff": None,
                "lip_sync_score": None,
                "is_suspicious": None,
                "detail": probe["_probe_error"],
            }

        streams = probe.get("streams", [])
        fmt = probe.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        video_dur: Optional[float] = None
        v_raw = (video_stream or {}).get("duration") or fmt.get("duration")
        if v_raw is not None:
            try:
                val = float(v_raw)
                if not math.isnan(val) and not math.isinf(val) and val > 0:
                    video_dur = val
            except ValueError:
                video_dur = None

        if audio_stream is None:
            return {
                "status": "no_audio",
                "has_audio": False,
                "audio_duration": None,
                "video_duration": video_dur,
                "duration_diff": None,
                "lip_sync_score": None,
                "is_suspicious": None,
            }

        audio_dur: Optional[float] = None
        a_raw = audio_stream.get("duration") or fmt.get("duration")
        if a_raw is not None:
            try:
                val = float(a_raw)
                if not math.isnan(val) and not math.isinf(val) and val > 0:
                    audio_dur = val
            except ValueError:
                audio_dur = None

        duration_diff = (
            abs(float(video_dur) - float(audio_dur))
            if (video_dur is not None and audio_dur is not None)
            else None
        )
        rms = self._extract_pcm_rms(path)

        duration_rms_risk = 0.0
        if duration_diff is not None and duration_diff > self.max_duration_diff_sec:
            duration_rms_risk += min(0.6, duration_diff / 2.0)
        if rms is not None and rms < 1e-4:
            duration_rms_risk += 0.25
        duration_rms_score = round(min(1.0, max(0.0, duration_rms_risk)), 4)

        # Lip-sync modeli ?v?zin? yaln?z audio/video m?dd?t uy?unlu?u v? RMS s?s enerjisi ?l??l?r
        lip_sync_score = round(min(1.0, max(0.0, 0.1 + duration_rms_risk)), 4)
        is_suspicious = bool(lip_sync_score >= 0.5)

        return {
            "status": "success",
            "has_audio": True,
            "signal_type": "audio_duration_and_rms_energy",
            "is_lip_sync_model": False,
            "audio_duration": audio_dur,
            "video_duration": video_dur,
            "duration_diff": round(duration_diff, 4) if duration_diff is not None else None,
            "rms_energy": round(rms, 6) if rms is not None else None,
            "audio_duration_rms_score": duration_rms_score,
            "lip_sync_score": lip_sync_score,
            "is_suspicious": is_suspicious,
        }