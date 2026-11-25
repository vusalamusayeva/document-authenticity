import hashlib
import json
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


class VisualDetector:
    # Bu m?hitd? h?qiq?t?n y?kl?n? bil?n formatlar (.npz h?mi??, .pt/.pth yaln?z torch olduqda)
    SUPPORTED_SUFFIXES = {".npz", ".pt", ".pth"}

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        checkpoint_path: Optional[str | Path] = None,
        threshold: float = 0.5,
        face_detector: Optional[FaceDetector] = None,
        input_size: tuple[int, int] = (224, 224),
        device: str = "cpu",
        expected_sha256: Optional[str] = None,
    ) -> None:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not (0.0 <= float(threshold) <= 1.0)
        ):
            raise InvalidConfigError("threshold [0.0, 1.0] aral???nda olmal?d?r.")
        resolved_path = checkpoint_path if checkpoint_path is not None else model_path
        self.checkpoint_path = Path(resolved_path) if resolved_path is not None else None
        self.model_path = self.checkpoint_path
        self.threshold = float(threshold)
        self.face_detector = face_detector or FaceDetector()
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.device = device
        self.expected_sha256 = expected_sha256.strip().lower() if expected_sha256 else None
        self._weights_verified = False
        self._model: Any = None
        self._weights_data: Optional[Dict[str, np.ndarray]] = None

    @staticmethod
    def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest().lower()

    def verify_model_weights(self) -> bool:
        if self.checkpoint_path is None:
            raise ModelUnavailableError("VisualDetector checkpoint fayl? g?st?rilm?yib.")
        if not self.checkpoint_path.exists() or not self.checkpoint_path.is_file():
            raise ModelUnavailableError(f"Model checkpoint fayl? tap?lmad?: {self.checkpoint_path}")
        if self.checkpoint_path.stat().st_size == 0:
            raise ModelUnavailableError(f"Model checkpoint fayl? bo?dur: {self.checkpoint_path}")
        suffix = self.checkpoint_path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ModelUnavailableError(
                f"D?st?kl?nm?y?n checkpoint format?: {self.checkpoint_path.suffix}"
            )
        if suffix in {".pt", ".pth"} and torch is None:
            raise ModelUnavailableError(
                "PyTorch qura?d?r?lmay?b; .pt/.pth checkpoint bu m?hitd? y?kl?n? bilm?z."
            )
        if self.expected_sha256:
            actual_sha256 = self.compute_file_sha256(self.checkpoint_path)
            if actual_sha256 != self.expected_sha256:
                raise ModelUnavailableError(
                    f"Model SHA-256 uy?unsuzlu?u: g?zl?nil?n {self.expected_sha256}, al?nan {actual_sha256}."
                )
        self._weights_verified = True
        return True

    def load_model(self) -> Dict[str, Any]:
        self.verify_model_weights()
        assert self.checkpoint_path is not None
        suffix = self.checkpoint_path.suffix.lower()

        if self._model is not None or self._weights_data is not None:
            return {
                "checkpoint_path": str(self.checkpoint_path),
                "status": "loaded",
                "backend": "numpy_linear_model" if self._weights_data is not None else "torch_checkpoint",
            }

        if suffix == ".npz":
            try:
                archive = np.load(self.checkpoint_path, allow_pickle=False)
                required_keys = {"weights", "bias", "laplacian_scale"}
                keys = set(archive.files)
                if not required_keys.issubset(keys):
                    missing = sorted(required_keys - keys)
                    raise ModelUnavailableError(
                        f"NPZ model artefakt?nda vacib a?arlar ?at??m?r: {missing}"
                    )
                weights = np.asarray(archive["weights"], dtype=np.float32)
                bias = np.asarray(archive["bias"], dtype=np.float32).reshape(-1)
                lap_scale = np.asarray(archive["laplacian_scale"], dtype=np.float32).reshape(-1)
                if weights.ndim != 1 or weights.shape[0] != 4 or bias.shape[0] != 1 or lap_scale.shape[0] != 1:
                    raise ModelUnavailableError(
                        "NPZ model ??kil?rinin ?l??l?ri uy?un deyil (weights: (4,), bias: (1,), laplacian_scale: (1,))."
                    )
                if not (np.all(np.isfinite(weights)) and np.all(np.isfinite(bias)) and np.all(np.isfinite(lap_scale))):
                    raise ModelUnavailableError("NPZ model parametrl?rind? NaN/Inf d?y?r var.")
                if "input_size" in keys:
                    ckpt_size = np.asarray(archive["input_size"]).reshape(-1)
                    if ckpt_size.size != 2 or (int(ckpt_size[0]), int(ckpt_size[1])) != self.input_size:
                        raise ModelUnavailableError(
                            f"Checkpoint input_size uy?unsuzlu?u: g?zl?nil?n {self.input_size}, al?nan {tuple(int(x) for x in ckpt_size)}."
                        )

                if "class_mapping" in keys:
                    raw_map = str(np.asarray(archive["class_mapping"]).reshape(-1)[0])
                    parsed_map = json.loads(raw_map)
                    if not isinstance(parsed_map, dict) or parsed_map.get("0") != "real" or parsed_map.get("1") != "deepfake":
                        raise ModelUnavailableError(
                            f"Checkpoint class_mapping uy?unsuzlu?u: {parsed_map}"
                        )

                self._weights_data = {
                    "weights": weights,
                    "bias": bias,
                    "laplacian_scale": lap_scale,
                }
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(f"NPZ model artefakt? y?kl?n? bilm?di: {exc}") from exc

            return {
                "checkpoint_path": str(self.checkpoint_path),
                "status": "loaded",
                "backend": "numpy_linear_model",
            }

        if torch is not None and nn is not None and suffix in {".pt", ".pth"}:
            try:
                loaded_obj = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except Exception as exc:
                raise ModelUnavailableError(f"PyTorch checkpoint y?kl?n? bilm?di: {exc}") from exc

            if isinstance(loaded_obj, nn.Module):
                model = loaded_obj.to(self.device)
                model.eval()
                self._model = model
            elif isinstance(loaded_obj, dict):
                state_dict = loaded_obj.get(
                    "state_dict",
                    loaded_obj.get("model_state_dict", loaded_obj),
                )
                if not isinstance(state_dict, dict) or len(state_dict) == 0:
                    raise ModelUnavailableError("Checkpoint daxilind? etibarl? state_dict tap?lmad?.")
                self._model = {
                    str(k).removeprefix("module."): v
                    for k, v in state_dict.items()
                }
            else:
                raise ModelUnavailableError("Checkpoint obyektinin format? uy?un deyil.")

            return {
                "checkpoint_path": str(self.checkpoint_path),
                "status": "loaded",
                "backend": "torch_checkpoint",
            }

        raise ModelUnavailableError(f"Model y?kl?n? bilm?di: {self.checkpoint_path}")

    def _extract_features(self, patch: np.ndarray) -> np.ndarray:
        resized = cv2.resize(patch, self.input_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normed = (rgb - mean) / std

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        channel_std = float(np.mean(np.std(normed, axis=(0, 1))))
        channel_mean = float(np.mean(np.abs(normed)))
        high_freq_ratio = float(np.mean(np.abs(np.diff(gray, axis=1))))

        return np.array(
            [channel_mean, channel_std, lap_var, high_freq_ratio],
            dtype=np.float32,
        )

    def _compute_region_score(self, patch: np.ndarray) -> float:
        if patch is None or patch.size == 0:
            return 0.0
        if self._weights_data is None and self._model is None:
            self.load_model()

        if self._weights_data is not None:
            feats = self._extract_features(patch)
            w = self._weights_data["weights"].copy()
            b = float(self._weights_data["bias"][0])
            scale = float(self._weights_data["laplacian_scale"][0])
            feats[2] = feats[2] / max(scale, 1e-6)
            logit = float(np.dot(w, feats) + b)
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))
            return float(max(0.0, min(1.0, round(prob, 4))))

        raise ModelUnavailableError("Aktiv model ??kil?ri tap?lmad?.")

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
