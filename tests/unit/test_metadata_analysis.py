from pathlib import Path
from typing import Any

import fitz  # type: ignore
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
    # Simulate PIL Image with Exif tags including Software & raw bytes
    img_path = tmp_path / "exif_img.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xd9")

    class _FakeExif(dict):
        def __bool__(self) -> bool:
            return True

    class _FakeImg:
        format = "JPEG"
        mode = "RGB"
        size = (30, 20)
        info: dict[str, Any] = {"comment": "ok", "exif": b"raw-bytes"}

        def __enter__(self) -> "_FakeImg":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def getexif(self) -> _FakeExif:
            return _FakeExif({305: "GIMP 2.10", 271: b"\x00\x01"})

    import PIL.Image as pil_image
    import PIL.ExifTags as pil_exif

    monkeypatch.setattr(pil_image, "open", lambda p: _FakeImg())
    monkeypatch.setattr(pil_exif.TAGS, "get", lambda tag_id, default=None: {305: "Software", 271: "Make"}.get(tag_id, default))

    analyzer = MetadataAnalyzer()
    img_res = analyzer.inspect(img_path)
    assert img_res["status"] == "success"
    assert img_res["metadata"]["exif"]["Software"] == "GIMP 2.10"
    assert img_res["suspicious"] is True

    # PDF metadata via PyMuPDF
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
