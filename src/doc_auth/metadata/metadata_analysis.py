import datetime
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetadataAnalyzer:
    """
    Fayl?n fiziki atributlar?n?, tam fayl SHA-256 hash d?y?rini, EXIF/g?r?nt?/PDF metadata-s?n?
    v? konkret sah?/d?y?r ?sasl? ??bh? siqnallar?n? t?hlil ed?n modul.
    """

    EDITOR_MARKERS = (
        "photoshop",
        "gimp",
        "affinity",
        "illustrator",
        "canva",
        "pixlr",
        "snapseed",
        "lightroom",
        "inkscape",
        "paint.net",
        "acrobat",
        "foxit",
        "libreoffice",
        "fpdf",
        "reportlab",
    )

    def __init__(self, chunk_size: int = 65536) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size m?sb?t tam ?d?d olmal?d?r.")
        self.chunk_size = chunk_size

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

    def _compute_full_sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with file_path.open("rb") as fh:
            while True:
                chunk = fh.read(self.chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def _extract_embedded_metadata(self, file_path: Path, ext: str) -> tuple[Dict[str, Any], List[str]]:
        meta: Dict[str, Any] = {}
        parse_errors: List[str] = []

        if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            try:
                from PIL import Image, ExifTags  # type: ignore

                with Image.open(file_path) as img:
                    meta["format"] = img.format
                    meta["mode"] = img.mode
                    meta["size"] = list(img.size)
                    if img.info:
                        for k, v in img.info.items():
                            if k != "exif" and isinstance(v, (str, int, float, bool)):
                                meta[str(k)] = v
                    exif_obj = img.getexif()
                    if exif_obj:
                        exif_map: Dict[str, Any] = {}
                        for tag_id, val in exif_obj.items():
                            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                            if isinstance(val, bytes):
                                continue
                            exif_map[str(tag_name)] = val
                        if exif_map:
                            meta["exif"] = exif_map
            except Exception as exc:
                parse_errors.append(f"G?r?nt? metadata-s? oxuna bilm?di: {exc}")

        if ext == ".pdf":
            try:
                import fitz  # type: ignore

                doc = fitz.open(file_path)
                meta["pdf_page_count"] = len(doc)
                if doc.metadata:
                    for k, v in doc.metadata.items():
                        if v:
                            meta[f"pdf_{k.lower()}"] = str(v)
                doc.close()
            except Exception as exc:
                parse_errors.append(f"PDF metadata-s? oxuna bilm?di: {exc}")

        return meta, parse_errors

    def analyze(self, payload: Any) -> Dict[str, Any]:
        file_path = self._resolve_path(payload)
        if file_path is None:
            return {
                "status": "not_found",
                "suspicious": None,
                "reason": "Metadata analizi ???n etibarl? fayl tap?lmad?.",
                "suspicious_signals": [],
            }

        stat = file_path.stat()
        ext = file_path.suffix.lower()
        modified_at = datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc).isoformat()
        created_at = datetime.datetime.fromtimestamp(stat.st_ctime, tz=datetime.timezone.utc).isoformat()
        embedded, parse_errors = self._extract_embedded_metadata(file_path, ext)

        suspicious = False
        reasons: List[str] = []
        suspicious_signals: List[Dict[str, Any]] = []

        if stat.st_size == 0:
            suspicious = True
            msg = "Fayl ?l??s? 0 baytd?r."
            reasons.append(msg)
            suspicious_signals.append({"field": "size_bytes", "value": 0, "reason": msg})

        if ext == "":
            suspicious = True
            msg = "Fayl?n geni?l?nm?si (extension) yoxdur."
            reasons.append(msg)
            suspicious_signals.append({"field": "extension", "value": "", "reason": msg})

        if stat.st_mtime < stat.st_ctime - 1.0:
            suspicious = True
            msg = "Modifikasiya vaxt? yarad?lma vaxt?ndan ?vv?ldir."
            reasons.append(msg)
            suspicious_signals.append({
                "field": "modified_at_utc",
                "value": modified_at,
                "reason": msg,
            })

        for err in parse_errors:
            suspicious = True
            reasons.append(err)
            suspicious_signals.append({"field": "embedded_metadata", "value": None, "reason": err})

        software_fields: List[tuple[str, str]] = []
        exif_data = embedded.get("exif", {})
        if isinstance(exif_data, dict):
            for key in ("Software", "ProcessingSoftware", "HostComputer"):
                val = exif_data.get(key)
                if val:
                    software_fields.append((f"exif.{key}", str(val)))
        for key in ("Software", "software", "pdf_creator", "pdf_producer"):
            val = embedded.get(key)
            if val:
                software_fields.append((key, str(val)))

        for field_name, field_val in software_fields:
            lower_val = field_val.lower()
            for marker in self.EDITOR_MARKERS:
                if marker in lower_val:
                    suspicious = True
                    msg = f"Metadata daxilind? redaktor/generator izi a?karland? ({marker})."
                    reasons.append(msg)
                    suspicious_signals.append({
                        "field": field_name,
                        "value": field_val,
                        "reason": msg,
                    })
                    break

        metadata_present = bool(embedded)
        if not metadata_present and not parse_errors:
            reasons.append("Daxili EXIF/PDF metadata sah?l?ri tap?lmad?.")

        return {
            "status": "success",
            "file_name": file_path.name,
            "extension": ext,
            "size_bytes": int(stat.st_size),
            "created_at_utc": created_at,
            "modified_at_utc": modified_at,
            "sha256": self._compute_full_sha256(file_path),
            "hash_scope": "full_file",
            "metadata_present": metadata_present,
            "metadata": embedded,
            "software_tags": [val for _, val in software_fields],
            "suspicious": suspicious,
            "reasons": reasons,
            "suspicious_signals": suspicious_signals,
        }

    def extract(self, payload: Any) -> Dict[str, Any]:
        return self.analyze(payload)

    def inspect(self, payload: Any) -> Dict[str, Any]:
        return self.analyze(payload)

    def __call__(self, payload: Any) -> Dict[str, Any]:
        return self.analyze(payload)
