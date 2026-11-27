from pathlib import Path
from typing import Any, ClassVar


class DatasetLoader:
    """
    Sənəd fayllarını və ya dataset qovluqlarını yükləyib strukturlaşdırılmış
    metadata və məzmun məlumatı qaytaran yükləyici.
    """

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
    }

    def __init__(self, allowed_extensions: list[str] | None = None) -> None:
        if allowed_extensions is not None:
            self.allowed_extensions = {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in allowed_extensions
            }
        else:
            self.allowed_extensions = set(self.SUPPORTED_EXTENSIONS)

    def load_document(self, path: str | Path) -> dict[str, Any]:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"Sənəd faylı tapılmadı: {file_path}")

        stat = file_path.stat()
        ext = file_path.suffix.lower()

        return {
            "status": "success",
            "path": str(file_path),
            "document_path": str(file_path),
            "file_name": file_path.name,
            "extension": ext,
            "size_bytes": int(stat.st_size),
            "is_supported_extension": ext in self.allowed_extensions,
        }

    def load(self, target: str | Path) -> dict[str, Any]:
        target_path = Path(target)
        if not target_path.exists():
            raise FileNotFoundError(f"Hədəf yol tapılmadı: {target_path}")

        if target_path.is_file():
            return self.load_document(target_path)

        if target_path.is_dir():
            items: list[dict[str, Any]] = []
            for entry in sorted(target_path.iterdir()):
                if entry.is_file() and entry.suffix.lower() in self.allowed_extensions:
                    items.append(self.load_document(entry))
            return {
                "status": "success",
                "directory": str(target_path),
                "count": len(items),
                "items": items,
            }

        raise ValueError(f"Dəstəklənməyən fayl sistemi obyekti: {target_path}")

    def analyze(self, target: str | Path) -> dict[str, Any]:
        return self.load(target)

    def __call__(self, target: str | Path) -> dict[str, Any]:
        return self.load(target)