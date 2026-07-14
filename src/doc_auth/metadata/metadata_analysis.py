"""
Metadata analizi (pipeline addım 4d)
- EXIF yoxlanması, compression tarixçəsi, screen-recapture izləri
"""

from dataclasses import dataclass


@dataclass
class MetadataResult:
    exif_consistent: bool
    compression_anomaly_detected: bool
    recapture_suspected: bool
    raw_exif: dict


def analyze_exif(image_path: str) -> dict:
    raise NotImplementedError


def detect_recapture_artifacts(image) -> bool:
    """Ekrandan/çapdan yenidən çəkim (moiré pattern, s.) aşkarlanması."""
    raise NotImplementedError
