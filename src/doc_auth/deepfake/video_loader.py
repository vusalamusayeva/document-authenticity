import os
import subprocess
import json
from typing import Dict, Any, Optional
from doc_auth.deepfake.exceptions import VideoLoadError, InvalidConfigError

def validate_video_metadata(video_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    ffprobe vasitəsilə video faylının formatını, ölçüsünü, müddətini və resolution-unu yoxlayır.
    """
    if not os.path.exists(video_path):
        raise VideoLoadError(f"Video faylı tapılmadı: {video_path}")

    config = config or {}
    max_file_size_mb = config.get("max_file_size_mb", 500)
    max_duration_sec = config.get("max_duration_sec", 300)
    allowed_formats = config.get("allowed_formats", [".mp4", ".mov", ".avi", ".webm"])

    _, ext = os.path.splitext(video_path)
    if ext.lower() not in allowed_formats:
        raise InvalidConfigError(f"Dəstəklənməyən video formatı: {ext}. Icazə verilənlər: {allowed_formats}")

    file_size_bytes = os.path.getsize(video_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        raise VideoLoadError(f"Video ölçüsü limiti keçdi: {file_size_mb:.2f}MB > {max_file_size_mb}MB")

    cmd = [
        "ffprobe", 
        "-v", "quiet", 
        "-print_format", "json", 
        "-show_format", 
        "-show_streams", 
        video_path
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
            raise VideoLoadError(f"ffprobe xətası: {result.stderr}")
        
        data = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        raise VideoLoadError("ffprobe əmri vaxt aşımına (timeout) uğradı.")
    except Exception as e:
        raise VideoLoadError(f"Video metadatası oxunarkən xəta baş verdi: {str(e)}")

    format_info = data.get("format", {})
    duration = float(format_info.get("duration", 0))
    if duration > max_duration_sec:
        raise VideoLoadError(f"Video müddəti limiti keçdi: {duration}s > {max_duration_sec}s")

    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise VideoLoadError("Videoda etibarlı video stream tapılmadı.")

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    return {
        "duration": duration,
        "size_mb": file_size_mb,
        "width": width,
        "height": height,
        "format": ext.lower()
    }

class VideoLoader:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config

    def load_metadata(self, video_path: str) -> Dict[str, Any]:
        return validate_video_metadata(video_path, self.config)