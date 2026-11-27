from pathlib import Path
from typing import Any

import fitz  # type: ignore
import PIL.ExifTags as pil_exif
import PIL.Image as pil_image
import pytest
from PIL import Image, PngImagePlugin

from doc_auth.metadata.metadata_analysis import MetadataAnalyzer


def test_metadata_analyzer_invalid_chunk_size_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        MetadataAnalyzer(chunk_size=0)

    analyzer = MetadataAnalyzer()
    missing_res = analyzer.analyze(tmp_path / "missing.pdf")
    assert missing_res["status"] == "not_found"
    assert missing_res["suspicious"] is None


def test_metadata_analyzer_signals_and_wrappers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    zero_no_ext = tmp_path / "raw_empty"
    zero_no_ext.write_bytes(b"")

    analyzer = MetadataAnalyzer()
    res = analyzer.analyze({"path": str(zero_no_ext)})
    assert res["status"] == "success"
    assert res["size_bytes"] == 0
    assert res["extension"] == ""
    assert res["suspicious"] is True
    assert res["metadata_present"] is False
    assert any(sig["field"] == "size_bytes" for sig in res["suspicious_signals"])
    assert any(sig["field"] == "extension" for sig in res["suspicious_signals"])

    dated_file = tmp_path / "dated.bin"
    dated_file.write_bytes(b"hello")
    orig_stat = Path.stat

    class _StatProxy:
        def __init__(self, st: Any) -> None:
            self._st = st
            self.st_size = st.st_size
            self.st_mtime = 1_000_000_000.0
            self.st_ctime = 1_700_000_000.0

    def _patched_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        st = orig_stat(self, *args, **kwargs)
        if self == dated_file:
            return _StatProxy(st)
        return st

    monkeypatch.setattr(Path, "stat", _patched_stat)
    dated_res = analyzer.analyze(dated_file)
    assert dated_res["suspicious"] is True
    assert any(sig["field"] == "modified_at_utc" for sig in dated_res["suspicious_signals"])


def test_metadata_analyzer_image_info_and_exif_editor_marker(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.png"
    img = Image.new("RGB", (24, 24), color=(100, 150, 200))
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("Software", "Adobe Photoshop 2025")
    png_info.add_text("Author", "QA Tester")
    img.save(img_path, pnginfo=png_info)

    analyzer = MetadataAnalyzer(chunk_size=16)
    res = analyzer.extract({"file_path": str(img_path)})
    assert res["status"] == "success"
    assert res["metadata_present"] is True
    assert res["metadata"]["Software"] == "Adobe Photoshop 2025"
    assert res["metadata"]["Author"] == "QA Tester"
    assert res["suspicious"] is True
    assert any("photoshop" in r.lower() for r in res["reasons"])


def test_metadata_analyzer_exif_dict_and_pdf_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    img_path = tmp_path / "exif_img.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xd9")

    class _FakeExif(dict):
        def __bool__(self) -> bool:
            return True

    class _FakeImg:
        format = "JPEG"
        mode = "RGB"
        size = (30, 20)
        info: dict[str, Any] = {"comment": "ok", "count": 3, "ratio": 1.5, "flag": True, "exif": b"raw-bytes"}

        def __enter__(self) -> "_FakeImg":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def getexif(self) -> _FakeExif:
            return _FakeExif({305: "GIMP 2.10", 271: b"\x00\x01"})

    monkeypatch.setattr(pil_image, "open", lambda p: _FakeImg())
    monkeypatch.setitem(pil_exif.TAGS, 305, "Software")
    monkeypatch.setitem(pil_exif.TAGS, 271, "Make")

    analyzer = MetadataAnalyzer()
    img_res = analyzer.inspect(img_path)
    assert img_res["status"] == "success"
    assert img_res["metadata"]["comment"] == "ok"
    assert img_res["metadata"]["count"] == 3
    assert img_res["metadata"]["exif"]["Software"] == "GIMP 2.10"
    assert "Make" not in img_res["metadata"]["exif"]
    assert img_res["suspicious"] is True

    pdf_path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test PDF")
    doc.set_metadata({"title": "Report", "author": "Analyst", "creator": "LibreOffice Writer"})
    doc.save(pdf_path)
    doc.close()

    pdf_res = analyzer(pdf_path)
    assert pdf_res["status"] == "success"
    assert pdf_res["metadata"]["pdf_page_count"] == 1
    assert pdf_res["metadata"]["pdf_creator"] == "LibreOffice Writer"
    assert pdf_res["suspicious"] is True
