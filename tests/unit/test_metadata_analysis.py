from pathlib import Path
import pytest
import cv2
import numpy as np

from doc_auth.metadata.metadata_analysis import MetadataAnalyzer


def test_metadata_analyzer_init_and_path_resolution(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        MetadataAnalyzer(chunk_size=0)

    analyzer = MetadataAnalyzer(chunk_size=16)
    sample = tmp_path / "doc.txt"
    sample.write_bytes(b"hello metadata world" * 5)

    assert analyzer._resolve_path(str(sample)) == sample
    assert analyzer._resolve_path(sample) == sample
    assert analyzer._resolve_path({"file_path": str(sample)}) == sample
    assert analyzer._resolve_path({"file_path": str(tmp_path / "missing.txt")}) is None
    assert analyzer._resolve_path(tmp_path / "missing.txt") is None
    assert analyzer._resolve_path(12345) is None

    digest = analyzer._compute_full_sha256(sample)
    assert isinstance(digest, str) and len(digest) == 64


def test_metadata_analyzer_image_and_corrupt_file(tmp_path: Path) -> None:
    analyzer = MetadataAnalyzer()

    img_path = tmp_path / "sample.png"
    canvas = np.ones((120, 180, 3), dtype=np.uint8) * 230
    cv2.putText(canvas, "META", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite(str(img_path), canvas)

    meta, errors = analyzer._extract_embedded_metadata(img_path, ".png")
    assert meta.get("format") == "PNG"
    assert meta.get("size") == [180, 120]
    assert errors == []

    bad_img = tmp_path / "corrupt.png"
    bad_img.write_bytes(b"not-a-valid-png")
    _, bad_errors = analyzer._extract_embedded_metadata(bad_img, ".png")
    assert len(bad_errors) >= 1
