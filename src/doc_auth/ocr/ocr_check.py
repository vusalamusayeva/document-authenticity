import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class OCRChecker:
    """
    S?n?d g?r?nt?s?nd? real OCR (Tesseract / EasyOCR) vasit?sil? m?tn v? sah?
    ??xar???, etibarl?l?q (confidence) qiym?tl?ndirm?si v? diaqnostik
    m?tn s?xl??? analizi aparan modul.
    """

    def __init__(
        self,
        min_text_density: float = 0.002,
        min_confidence: float = 0.4,
        lang: str = "eng",
    ) -> None:
        if not (0.0 <= min_text_density <= 1.0):
            raise ValueError("min_text_density [0.0, 1.0] aral???nda olmal?d?r.")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError("min_confidence [0.0, 1.0] aral???nda olmal?d?r.")
        self.min_text_density = min_text_density
        self.min_confidence = min_confidence
        self.lang = lang

    def _resolve_image(self, payload: Any) -> np.ndarray | None:
        if isinstance(payload, np.ndarray):
            return payload if payload.size > 0 else None
        if isinstance(payload, dict):
            for key in ("image", "frame", "array", "data"):
                val = payload.get(key)
                if isinstance(val, np.ndarray) and val.size > 0:
                    return val
            doc_path = payload.get("document_path") or payload.get("path")
            if doc_path:
                return self._resolve_image(doc_path)
        if isinstance(payload, (str, Path)):
            path = Path(payload)
            if (
                path.exists()
                and path.is_file()
                and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
            ):
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                return img if img is not None and img.size > 0 else None
        return None

    def _compute_text_density(self, gray: np.ndarray) -> float:
        h, w = gray.shape[:2]
        total_pixels = float(max(1, w * h))
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            10,
        )
        foreground_pixels = float(np.count_nonzero(adaptive))
        return round(foreground_pixels / total_pixels, 6)

    def _run_tesseract(self, gray: np.ndarray) -> dict[str, Any]:
        import pytesseract  # type: ignore

        exe = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
        if shutil.which(exe) is None and not Path(exe).exists():
            raise RuntimeError(f"Tesseract binary tap?lmad?: {exe}")

        data = pytesseract.image_to_data(
            gray,
            lang=self.lang,
            output_type=pytesseract.Output.DICT,
        )
        words: list[str] = []
        confidences: list[float] = []
        fields: list[dict[str, Any]] = []

        raw_texts = data.get("text", [])
        raw_confs = data.get("conf", [])
        lefts = data.get("left", [])
        tops = data.get("top", [])
        widths = data.get("width", [])
        heights = data.get("height", [])

        for idx, raw_w in enumerate(raw_texts):
            token = str(raw_w).strip()
            if not token:
                continue
            conf_raw = raw_confs[idx] if idx < len(raw_confs) else -1
            try:
                conf_val = float(conf_raw)
            except (TypeError, ValueError):
                conf_val = -1.0
            norm_conf = round(max(0.0, min(1.0, conf_val / 100.0)), 4) if conf_val >= 0 else 0.0
            words.append(token)
            confidences.append(norm_conf)
            fields.append(
                {
                    "text": token,
                    "confidence": norm_conf,
                    "bbox": [
                        int(lefts[idx]) if idx < len(lefts) else 0,
                        int(tops[idx]) if idx < len(tops) else 0,
                        int(widths[idx]) if idx < len(widths) else 0,
                        int(heights[idx]) if idx < len(heights) else 0,
                    ],
                }
            )

        mean_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
        return {
            "engine": "tesseract",
            "text": " ".join(words),
            "confidence": mean_conf,
            "fields": fields,
        }

    def _run_easyocr(self, gray: np.ndarray) -> dict[str, Any]:
        import easyocr  # type: ignore

        reader = easyocr.Reader([self.lang if len(self.lang) == 2 else "en"], gpu=False, verbose=False)
        results = reader.readtext(gray)
        words: list[str] = []
        confidences: list[float] = []
        fields: list[dict[str, Any]] = []

        for bbox, text, conf in results:
            token = str(text).strip()
            if not token:
                continue
            norm_conf = round(max(0.0, min(1.0, float(conf))), 4)
            words.append(token)
            confidences.append(norm_conf)
            fields.append({"text": token, "confidence": norm_conf, "bbox": bbox})

        mean_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
        return {
            "engine": "easyocr",
            "text": " ".join(words),
            "confidence": mean_conf,
            "fields": fields,
        }

    def check(self, payload: Any) -> dict[str, Any]:
        image = self._resolve_image(payload)
        if image is None:
            return {
                "status": "not_applicable",
                "has_text": None,
                "suspicious": None,
                "confidence": 0.0,
                "text": "",
                "fields": [],
                "text_density": 0.0,
                "reason": "OCR analizi ???n g?r?nt? matrisi tap?lmad?.",
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        text_density = self._compute_text_density(gray)

        engine_errors: list[str] = []
        ocr_out: dict[str, Any] | None = None

        for runner in (self._run_tesseract, self._run_easyocr):
            try:
                ocr_out = runner(gray)
                break
            except (OSError, RuntimeError, ValueError, TypeError, AttributeError, ImportError) as exc:
                engine_errors.append(f"{runner.__name__}: {exc}")

        if ocr_out is None:
            return {
                "status": "ocr_unavailable",
                "has_text": None,
                "suspicious": None,
                "confidence": 0.0,
                "text": "",
                "fields": [],
                "text_density": text_density,
                "min_text_density": self.min_text_density,
                "error": " | ".join(engine_errors) if engine_errors else "OCR m?h?rriki ?l?atan deyil.",
                "reason": "Real OCR m?h?rriki i?l?dil? bilm?di; n?tic? etibarl? m?tn ??xar??? kimi t?qdim olunmur.",
            }

        extracted_text = str(ocr_out.get("text") or "")
        confidence = float(ocr_out.get("confidence") or 0.0)
        has_text = bool(len(extracted_text.strip()) > 0 and text_density >= self.min_text_density)
        low_conf_text = bool(has_text and confidence < self.min_confidence)

        return {
            "status": "success",
            "engine": ocr_out.get("engine"),
            "has_text": has_text,
            "suspicious": low_conf_text,
            "confidence": confidence,
            "text": extracted_text,
            "fields": ocr_out.get("fields", []),
            "text_density": text_density,
            "min_text_density": self.min_text_density,
            "min_confidence": self.min_confidence,
        }

    def run(self, payload: Any) -> dict[str, Any]:
        return self.check(payload)

    def analyze(self, payload: Any) -> dict[str, Any]:
        return self.check(payload)

    def __call__(self, payload: Any) -> dict[str, Any]:
        return self.check(payload)
