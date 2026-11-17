import os
import subprocess
import json
from pathlib import Path
from doc_auth.schemas.pipeline_schemas import DeepfakeConfig, VideoMetadata
from doc_auth.deepfake.exceptions import VideoLoadError

class VideoLoader:
    def __init__(self, config: DeepfakeConfig = None):
        self.config = config or DeepfakeConfig()

    def load_and_validate(self, video_path: str) -> VideoMetadata:
        path = Path(video_path)
        if not path.exists():
            raise VideoLoadError(f"Video faylı tapılmadı: {video_path}")

        # Fayl ölçüsü yoxlanışı (MB)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise VideoLoadError(f"Fayl ölçüsü limitdən böyükdür: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB")

        # ffprobe vasitəsilə metadata oxunması (shell=False, timeout ilə təhlükəsiz)
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                shell=False
            )
            if result.returncode != 0:
                raise VideoLoadError(f"ffprobe xətası: {result.stderr.strip()}")
            
            data = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            raise VideoLoadError("ffprobe əmri vaxt aşımına (timeout) uğradı")
        except Exception as e:
            raise VideoLoadError(f"Video metadata oxunarkən xəta baş verdi: {str(e)}")

        format_info = data.get("format", {})
        duration_sec = float(format_info.get("duration", 0.0))

        if duration_sec > self.config.max_duration_sec:
            raise VideoLoadError(f"Video müddəti limitdən uzundur: {duration_sec}s > {self.config.max_duration_sec}s")

        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

        if not video_stream:
            raise VideoLoadError("Videoda video axını (stream) tapılmadı")

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # FPS parsing
        r_frame_rate = video_stream.get("r_frame_rate", "30/1")
        try:
            num, den = map(int, r_frame_rate.split("/"))
            fps = num / den if den > 0 else 30.0
        except Exception:
            fps = 30.0

        total_frames = int(video_stream.get("nb_frames", int(duration_sec * fps)))

        return VideoMetadata(
            filename=path.name,
            format=format_info.get("format_name", "unknown"),
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            has_audio=audio_stream is not None,
            audio_duration_sec=float(audio_stream.get("duration", duration_sec)) if audio_stream else None
        )