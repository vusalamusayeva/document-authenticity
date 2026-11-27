from pathlib import Path
import os
import time
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


def test_metadata_analyze_signals_and_wrappers(tmp_path: Path, monkeypatch) -> None:
    analyzer = MetadataAnalyzer()

    missing_res = analyzer.analyze(tmp_path / "does_not_exist.pdf")
    assert missing_res["status"] == "not_found"
    assert missing_res["suspicious"] is None

    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    empty_res = analyzer.analyze(empty_file)
    assert empty_res["suspicious"] is True
    assert any(sig["field"] == "size_bytes" for sig in empty_res["suspicious_signals"])

    no_ext = tmp_path / "noext_file"
    no_ext.write_bytes(b"plain text content")
    no_ext_res = analyzer.analyze(no_ext)
    assert no_ext_res["suspicious"] is True
    assert any(sig["field"] == "extension" for sig in no_ext_res["suspicious_signals"])

    doc_file = tmp_path / "edited.pdf"
    doc_file.write_bytes(b"%PDF-1.4 fake test")

    now = time.time()
    os.utime(doc_file, (now - 120, now - 120))
    stat = doc_file.stat()
    os.utime(doc_file, (stat.st_atime, stat.st_ctime - 10.0))

    monkeypatch.setattr(
        analyzer,
        "_extract_embedded_metadata",
        lambda _p, _ext: ({"pdf_producer": "Adobe Photoshop 2026"}, []),
    )

    res = analyzer.analyze(doc_file)
    assert res["status"] == "success"
    assert res["suspicious"] is True
    assert any(sig["field"] == "modified_at_utc" for sig in res["suspicious_signals"])
    assert any(sig["field"] == "pdf_producer" for sig in res["suspicious_signals"])

    assert analyzer.extract(doc_file)["suspicious"] is True
    assert analyzer.inspect(doc_file)["suspicious"] is True
    assert analyzer(doc_file)["suspicious"] is True
