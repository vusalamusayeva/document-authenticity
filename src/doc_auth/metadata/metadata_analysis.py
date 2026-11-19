import datetime
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional


class MetadataAnalyzer:
    """
    Faylın fiziki atributlarını, hash dəyərini və əsas metadata göstəricilərini
    oxuyub təhlil edən modul.
    """

    def __init__(self, max_hash_bytes: int = 16 * 1024 * 1024) -> None:
        if max_hash_bytes <= 0:
            raise ValueError("max_hash_bytes müsbət tam ədəd olmalıdır.")
        self.max_hash_bytes = max_hash_bytes

    def _resolve_path(self, payload: Any) -> Optional[Path]:
        if isinstance(payload, (str, Path)):
            p = Path(payload)
            return p if p.exists() and p.is_file() else None
        if isinstance(payload, dict):
            for key in ("document_path", "path", "file_path"):
                val = payload.get(key)
                if val:
                    p = Path(val)
                    if p.exists() and p.is_file():
                        return p
        return None

    def _compute_sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        read_bytes = 0
        with file_path.open("rb") as fh:
            while read_bytes < self.max_hash_bytes:
                chunk = fh.read(min(65536, self.max_hash_bytes - read_bytes))
                if not chunk:
                    break
                hasher.update(chunk)
                read_bytes += len(chunk)
        return hasher.hexdigest()

    def analyze(self, payload: Any) -> Dict[str, Any]:
        file_path = self._resolve_path(payload)
        if file_path is None:
            return {
                "status": "not_found",
                "suspicious": None,
                "reason": "Metadata analizi üçün etibarlı fayl tapılmadı.",
            }

        stat = file_path.stat()
        ext = file_path.suffix.lower()
        modified_at = datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc).isoformat()
        created_at = datetime.datetime.fromtimestamp(stat.st_ctime, tz=datetime.timezone.utc).isoformat()

        suspicious = False
        reasons = []
        if stat.st_size == 0:
            suspicious = True
            reasons.append("Fayl ölçüsü 0 baytdır.")
        if ext == "":
            suspicious = True
            reasons.append("Faylın genişlənməsi (extension) yoxdur.")

        return {
            "status": "success",
            "file_name": file_path.name,
            "extension": ext,
            "size_bytes": int(stat.st_size),
            "created_at_utc": created_at,
            "modified_at_utc": modified_at,
            "sha256": self._compute_sha256(file_path),
            "suspicious": suspicious,
            "reasons": reasons,
        }

    def extract(self, payload: Any) -> Dict[str, Any]:
        return self.analyze(payload)

    def inspect(self, payload: Any) -> Dict[str, Any]:
        return self.analyze(payload)

    def __call__(self, payload: Any) -> Dict[str, Any]:
        return self.analyze(payload)