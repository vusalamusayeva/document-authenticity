from pathlib import Path
import pytest

from doc_auth.ingestion.dataset_loader import DatasetLoader


def test_dataset_loader_file_and_custom_extensions(tmp_path: Path) -> None:
    doc = tmp_path / "passport.pdf"
    doc.write_bytes(b"%PDF-1.4 sample")

    custom_loader = DatasetLoader(allowed_extensions=["pdf", ".PNG"])
    assert custom_loader.allowed_extensions == {".pdf", ".png"}

    info = custom_loader.load_document(doc)
    assert info["status"] == "success"
    assert info["file_name"] == "passport.pdf"
    assert info["extension"] == ".pdf"
    assert info["size_bytes"] > 0
    assert info["is_supported_extension"] is True

    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("hello", encoding="utf-8")
    u_info = custom_loader.load_document(unsupported)
    assert u_info["is_supported_extension"] is False

    assert custom_loader.load(doc)["file_name"] == "passport.pdf"
    assert custom_loader.analyze(doc)["file_name"] == "passport.pdf"
    assert custom_loader(doc)["file_name"] == "passport.pdf"


def test_dataset_loader_directory_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loader = DatasetLoader()
    dataset_dir = tmp_path / "batch"
    dataset_dir.mkdir()

    f1 = dataset_dir / "a.png"
    f1.write_bytes(b"png")
    f2 = dataset_dir / "b.jpg"
    f2.write_bytes(b"jpg")
    ignored = dataset_dir / "ignore.csv"
    ignored.write_text("a,b", encoding="utf-8")

    result = loader.load(dataset_dir)
    assert result["status"] == "success"
    assert result["count"] == 2
    assert [item["file_name"] for item in result["items"]] == ["a.png", "b.jpg"]

    with pytest.raises(FileNotFoundError):
        loader.load_document(tmp_path / "missing.pdf")

    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "does_not_exist")

    special = tmp_path / "special_obj"
    special.write_text("x", encoding="utf-8")
    monkeypatch.setattr(Path, "is_file", lambda self: False if self == special else Path.is_file.__wrapped__(self) if hasattr(Path.is_file, "__wrapped__") else False)
    monkeypatch.setattr(Path, "is_dir", lambda self: False if self == special else False)
    with pytest.raises(ValueError):
        loader.load(special)
