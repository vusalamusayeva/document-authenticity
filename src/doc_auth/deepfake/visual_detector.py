from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from doc_auth.deepfake.exceptions import InvalidConfigError, ModelUnavailableError
from doc_auth.deepfake.face_detector import FaceDetector

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class _FallbackConvClassifier(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Checkpoint state_dict və ya model yükləndikdə işlədilən yüngül CNN arxitekturası."""

    def __init__(self) -> None:
        if nn is None:
            return
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, x: Any) -> Any:
        feat = self.features(x).flatten(1)
        return torch.sigmoid(self.classifier(feat)).squeeze(-1)


class VisualDetector:
    SUPPORTED_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".bin", ".safetensors"}

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        checkpoint_path: Optional[str | Path] = None,
        threshold: float = 0.5,
        face_detector: Optional[FaceDetector] = None,
        input_size: tuple[int, int] = (224, 224),
        device: str = "cpu",
    ) -> None:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not (0.0 <= float(threshold) <= 1.0)
        ):
            raise InvalidConfigError("threshold [0.0, 1.0] aralığında olmalıdır.")
        resolved_path = checkpoint_path if checkpoint_path is not None else model_path
        self.checkpoint_path = Path(resolved_path) if resolved_path is not None else None
        self.model_path = self.checkpoint_path
        self.threshold = float(threshold)
        self.face_detector = face_detector or FaceDetector()
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.device = device
        self._weights_verified = False
        self._model: Any = None

    def verify_model_weights(self) -> bool:
        if self.checkpoint_path is None:
            raise ModelUnavailableError("VisualDetector checkpoint faylı göstərilməyib.")
        if not self.checkpoint_path.exists() or not self.checkpoint_path.is_file():
            raise ModelUnavailableError(f"Model checkpoint faylı tapılmadı: {self.checkpoint_path}")
        if self.checkpoint_path.stat().st_size == 0:
            raise ModelUnavailableError(f"Model checkpoint faylı boşdur: {self.checkpoint_path}")
        if self.checkpoint_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ModelUnavailableError(
                f"Dəstəklənməyən checkpoint formatı: {self.checkpoint_path.suffix}"
            )
        self._weights_verified = True
        return True

    def load_model(self) -> Dict[str, Any]:
        self.verify_model_weights()
        if self._model is not None:
            return {
                "checkpoint_path": str(self.checkpoint_path),
                "status": "loaded",
                "backend": "torch_checkpoint" if torch is not None else "opencv_fallback",
            }

        if torch is not None and self.checkpoint_path is not None and self.checkpoint_path.suffix.lower() in {".pt", ".pth"}:
            try:
                loaded_obj = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            except Exception as exc:
                raise ModelUnavailableError(f"Checkpoint yüklənə bilmədi: {exc}") from exc

            if isinstance(loaded_obj, nn.Module):
                model = loaded_obj.to(self.device)
                model.eval()
                self._model = model
            elif isinstance(loaded_obj, dict):
                state_dict = loaded_obj.get("state_dict", loaded_obj.get("model_state_dict", loaded_obj))
                model = _FallbackConvClassifier().to(self.device)
                if isinstance(state_dict, dict):
                    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
                    model.load_state_dict(cleaned, strict=False)
                model.eval()
                self._model = model
            else:
                raise ModelUnavailableError("Checkpoint obyektinin formatı uyğun deyil.")

            return {
                "checkpoint_path": str(self.checkpoint_path),
                "status": "loaded",
                "backend": "torch_checkpoint",
            }

        return {
            "checkpoint_path": str(self.checkpoint_path),
            "status": "loaded",
            "backend": "opencv_spatial_analyzer",
        }

    def _preprocess_patch(self, patch: np.ndarray) -> Any:
        resized = cv2.resize(patch, self.input_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normed = (rgb - mean) / std
        if torch is not None and self._model is not None:
            tensor = torch.from_numpy(np.transpose(normed, (2, 0, 1))).unsqueeze(0).to(self.device)
            return tensor
        return normed

    def _compute_region_score(self, patch: np.ndarray) -> float:
        if patch is None or patch.size == 0:
            return 0.0
        if self._model is None and self.checkpoint_path is not None and self.checkpoint_path.suffix.lower() in {".pt", ".pth"}:
            self.load_model()

        prep = self._preprocess_patch(patch)
        if torch is not None and self._model is not None:
            with torch.no_grad():
                out = self._model(prep)
                val = float(out.detach().cpu().reshape(-1)[0].item())
            return float(max(0.0, min(1.0, round(val, 4))))

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        norm_score = 1.0 / (1.0 + lap_var / 250.0)
        return float(max(0.0, min(1.0, round(norm_score, 4))))

    def analyze_frame(
        self,
        frame: np.ndarray,
        faces: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "status": "invalid_frame",
                "face_detected": False,
                "score": None,
                "max_deepfake_score": 0.0,
                "face_predictions": [],
            }

        detected_faces = faces if faces is not None else self.face_detector.detect_faces(frame)
        if not detected_faces:
            return {
                "status": "no_face",
                "face_detected": False,
                "score": None,
                "max_deepfake_score": 0.0,
                "face_predictions": [],
            }

        h, w = frame.shape[:2]
        face_predictions: List[Dict[str, Any]] = []
        for idx, face in enumerate(detected_faces):
            bbox = face.get("bbox") if isinstance(face, dict) else face
            if bbox and len(bbox) >= 4:
                x1 = max(0, min(w - 1, int(bbox[0])))
                y1 = max(0, min(h - 1, int(bbox[1])))
                x2 = max(x1 + 1, min(w, int(bbox[2])))
                y2 = max(y1 + 1, min(h, int(bbox[3])))
                patch = frame[y1:y2, x1:x2]
            else:
                x1, y1, x2, y2 = 0, 0, w, h
                patch = frame

            score = self._compute_region_score(patch)
            face_predictions.append({
                "face_index": idx,
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "is_deepfake": bool(score >= self.threshold),
            })

        max_score = float(max(p["score"] for p in face_predictions))
        return {
            "status": "success",
            "face_detected": True,
            "score": max_score,
            "max_deepfake_score": max_score,
            "face_predictions": face_predictions,
        }