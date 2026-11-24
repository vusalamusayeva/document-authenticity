from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from doc_auth.classification.document_classifier import DocumentClassifier
from doc_auth.explainability import explain as explain_module
from doc_auth.forensics.tampering_detection import TamperingDetector
from doc_auth.fusion.decision_model import DecisionModel
from doc_auth.ingestion.dataset_loader import DatasetLoader
from doc_auth.ingestion.document_detection import DocumentDetector
from doc_auth.metadata.metadata_analysis import MetadataAnalyzer
from doc_auth.ocr.ocr_check import OCRChecker
from doc_auth.schemas.pipeline_schemas import ArtifactCheckResult, DocumentAuthSummary


def _default_explainer() -> Any:
    if hasattr(explain_module, "Explainer"):
        return explain_module.Explainer()
    for attr_name in dir(explain_module):
        obj = getattr(explain_module, attr_name)
        if callable(obj) and not attr_name.startswith("_"):
            return obj
    raise RuntimeError("doc_auth.explainability.explain modulunda izah komponenti tapılmadı.")


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "ndarray_shape": list(value.shape),
            "ndarray_dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(item) for item in value]
    return value


class AnalysisService:
    """
    Sənəd autentifikasiyası üçün ingestion, metadata, OCR, classification,
    forensics, fusion və explainability modullarını real axında birləşdirir.
    """

    def __init__(
        self,
        loader: Optional[Any] = None,
        detector: Optional[Any] = None,
        metadata_analyzer: Optional[Any] = None,
        ocr_checker: Optional[Any] = None,
        classifier: Optional[Any] = None,
        tampering_detector: Optional[Any] = None,
        decision_model: Optional[Any] = None,
        explainer: Optional[Any] = None,
    ) -> None:
        self.loader = loader if loader is not None else DatasetLoader()
        self.detector = detector if detector is not None else DocumentDetector()
        self.metadata_analyzer = metadata_analyzer if metadata_analyzer is not None else MetadataAnalyzer()
        self.ocr_checker = ocr_checker if ocr_checker is not None else OCRChecker()
        self.classifier = classifier if classifier is not None else DocumentClassifier()
        self.tampering_detector = tampering_detector if tampering_detector is not None else TamperingDetector()
        self.decision_model = decision_model if decision_model is not None else DecisionModel()
        self.explainer = explainer if explainer is not None else _default_explainer()

    @staticmethod
    def _invoke_module(component: Any, method_candidates: tuple[str, ...], payload: Any) -> Dict[str, Any]:
        for name in method_candidates:
            fn: Optional[Callable[..., Any]] = getattr(component, name, None)
            if callable(fn):
                res = fn(payload)
                if isinstance(res, dict):
                    return res
                return {"status": "success", "result": res}
        if callable(component):
            res = component(payload)
            if isinstance(res, dict):
                return res
            return {"status": "success", "result": res}
        raise RuntimeError(f"{type(component).__name__} üçün uyğun icra metodu tapılmadı.")

    def _run_classifier(
        self,
        detected: Dict[str, Any],
        ocr: Dict[str, Any],
        ingested: Dict[str, Any] | None = None,
        document_path: Path | None = None,
    ) -> Dict[str, Any]:
        payload: Any = None
        for container in (detected, ingested or {}):
            if isinstance(container, dict):
                for key in ("image", "frame", "array", "aligned_image", "page_image"):
                    if key in container and container[key] is not None:
                        payload = container[key]
                        break
            if payload is not None:
                break

        if payload is None and isinstance((ingested or {}).get("pages"), list) and len(ingested["pages"]) > 0:
            first_page = ingested["pages"][0]
            if isinstance(first_page, dict):
                for key in ("image", "frame", "array"):
                    if key in first_page and first_page[key] is not None:
                        payload = first_page[key]
                        break

        if payload is None and document_path is not None:
            payload = {
                "document_path": str(document_path),
                "features": [
                    float(detected.get("area_ratio") or 0.0),
                    float(ocr.get("text_density") or 0.0),
                    1.0 if detected.get("detected") else 0.0,
                    float(ocr.get("char_count") or 0.0) / 1000.0,
                ],
            }

        for name in ("classify", "predict", "predict_proba", "analyze"):
            fn: Optional[Callable[..., Any]] = getattr(self.classifier, name, None)
            if callable(fn):
                try:
                    out = fn(payload)
                    if isinstance(out, dict):
                        return out
                    return {"status": "success", "predictions": out}
                except Exception as exc:
                    return {"status": "unavailable", "error": str(exc)}
        return {"status": "unavailable", "error": "Uy?un klassifikasiya metodu tap?lmad?."}

    def analyze(self, document_path: str | Path) -> Dict[str, Any]:
        path = Path(document_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Sənəd faylı tapılmadı: {path}")

        ingested = self._invoke_module(self.loader, ("load", "load_document", "analyze"), path)
        detected = self._invoke_module(self.detector, ("detect", "detect_document", "analyze"), ingested)
        metadata = self._invoke_module(self.metadata_analyzer, ("analyze", "extract", "inspect"), path)
        ocr = self._invoke_module(self.ocr_checker, ("check", "run", "analyze"), detected)
        classification = self._run_classifier(detected, ocr, ingested=ingested, document_path=path)
        forensics = self._invoke_module(self.tampering_detector, ("detect", "analyze", "run"), detected)

        fusion_input = {
            "ingestion": ingested,
            "detection": detected,
            "metadata": metadata,
            "ocr": ocr,
            "classification": classification,
            "forensics": forensics,
        }
        decision = self._invoke_module(self.decision_model, ("decide", "predict", "fuse", "analyze"), fusion_input)
        explanation = self._invoke_module(
            self.explainer,
            ("explain", "generate", "analyze"),
            {"decision": decision, "signals": fusion_input},
        )

        return _sanitize_for_json({
            "status": decision.get("status", "success"),
            "document_path": str(path),
            "ingestion": ingested,
            "detection": detected,
            "metadata": metadata,
            "ocr": ocr,
            "classification": classification,
            "forensics": forensics,
            "decision": decision,
            "explanation": explanation,
        })

    def analyze_summary(self, document_path: str | Path) -> DocumentAuthSummary:
        raw = self.analyze(document_path)
        checks: List[ArtifactCheckResult] = []
        for mod_name in ("ingestion", "detection", "metadata", "ocr", "classification", "forensics"):
            mod_res = raw.get(mod_name) if isinstance(raw.get(mod_name), dict) else {}
            sanitized_details = _sanitize_for_json(mod_res)
            checks.append(
                ArtifactCheckResult(
                    module=mod_name,
                    status=str(mod_res.get("status", "unknown")),
                    score=float(mod_res["score"]) if isinstance(mod_res.get("score"), (int, float, np.number)) else None,
                    is_suspicious=mod_res.get("is_suspicious") if isinstance(mod_res.get("is_suspicious"), (bool, np.bool_)) else None,
                    details=sanitized_details if isinstance(sanitized_details, dict) else {"value": sanitized_details},
                )
            )
        decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
        risk_val = decision.get("risk_score")
        return DocumentAuthSummary(
            request_id=Path(document_path).name,
            status=str(raw.get("status", "completed")),
            overall_risk_score=float(risk_val) if isinstance(risk_val, (int, float, np.number)) else None,
            decision=str(decision.get("decision", "inconclusive")),
            checks=checks,
            metadata={"document_path": str(document_path)},
        )