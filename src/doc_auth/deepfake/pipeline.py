from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from doc_auth.deepfake.audio_analysis import AudioLipSyncAnalyzer
from doc_auth.deepfake.exceptions import ModelUnavailableError, VideoValidationError
from doc_auth.deepfake.face_detector import FaceDetector
from doc_auth.deepfake.visual_detector import VisualDetector


class DeepfakePipeline:
    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        visual_detector: Optional[VisualDetector] = None,
        audio_analyzer: Optional[AudioLipSyncAnalyzer] = None,
        frame_sampler: Optional[Callable[[str | Path], Iterable[Dict[str, Any]]]] = None,
        risk_scorer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.face_detector = face_detector or FaceDetector()
        self.visual_detector = visual_detector
        self.audio_analyzer = audio_analyzer or AudioLipSyncAnalyzer()
        self.frame_sampler = frame_sampler
        self.risk_scorer = risk_scorer

    def run(
        self,
        video_path: Any,
        frames: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if isinstance(video_path, (bytes, bytearray, memoryview)) or not isinstance(video_path, (str, Path)):
            return {
                "status": "failed",
                "is_deepfake": None,
                "risk_score": None,
                "error_type": "InvalidInputError",
                "error": "Etibarsız video yolu və ya xam bayt daxil edilib.",
                "module_statuses": {
                    "visual": "failed",
                    "audio": "not_run",
                },
                "frames_analyzed": 0,
                "faces_detected": 0,
            }

        path = Path(video_path)

        try:
            audio_result = self.audio_analyzer.analyze(path)
        except Exception as exc:
            audio_result = {"status": "error", "error": str(exc)}

        if self.visual_detector is None:
            return {
                "status": "model_unavailable",
                "is_deepfake": None,
                "risk_score": None,
                "module_statuses": {
                    "visual": "model_unavailable",
                    "audio": audio_result.get("status", "unknown"),
                },
                "audio": audio_result,
                "frames_analyzed": 0,
                "faces_detected": 0,
            }

        try:
            self.visual_detector.verify_model_weights()
        except ModelUnavailableError as exc:
            return {
                "status": "model_unavailable",
                "is_deepfake": None,
                "risk_score": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "module_statuses": {
                    "visual": "model_unavailable",
                    "audio": audio_result.get("status", "unknown"),
                },
                "audio": audio_result,
                "frames_analyzed": 0,
                "faces_detected": 0,
            }

        try:
            frame_stream = (
                frames
                if frames is not None
                else (self.frame_sampler(path) if self.frame_sampler is not None else [])
            )

            sampled_frames: List[Dict[str, Any]] = []
            for item in frame_stream:
                if isinstance(item, np.ndarray):
                    sampled_frames.append({"frame": item, "timestamp": None})
                elif isinstance(item, dict) and "frame" in item:
                    sampled_frames.append(item)
        except VideoValidationError as exc:
            return {
                "status": "failed",
                "is_deepfake": None,
                "risk_score": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "module_statuses": {
                    "visual": "failed",
                    "audio": audio_result.get("status", "unknown"),
                },
                "audio": audio_result,
                "frames_analyzed": 0,
                "faces_detected": 0,
            }

        if not sampled_frames:
            return {
                "status": "no_frames",
                "is_deepfake": None,
                "risk_score": None,
                "module_statuses": {
                    "visual": "no_frames",
                    "audio": audio_result.get("status", "unknown"),
                },
                "audio": audio_result,
                "frames_analyzed": 0,
                "faces_detected": 0,
            }

        total_faces = 0
        all_face_predictions: List[Dict[str, Any]] = []
        for idx, item in enumerate(sampled_frames):
            frame = item["frame"]
            ts = item.get("timestamp")
            faces = self.face_detector.detect_faces(frame)
            if not faces:
                continue
            total_faces += len(faces)
            frame_pred = self.visual_detector.analyze_frame(frame, faces=faces)
            if frame_pred.get("status") == "success":
                for fp in frame_pred.get("face_predictions", []):
                    entry = dict(fp)
                    entry["frame_index"] = idx
                    entry["timestamp"] = ts
                    all_face_predictions.append(entry)

        if total_faces == 0 or not all_face_predictions:
            return {
                "status": "no_face",
                "is_deepfake": None,
                "risk_score": None,
                "module_statuses": {
                    "visual": "no_face",
                    "audio": audio_result.get("status", "unknown"),
                },
                "audio": audio_result,
                "frames_analyzed": len(sampled_frames),
                "faces_detected": 0,
            }

        scores = [p["score"] for p in all_face_predictions]
        max_visual_score = float(max(scores))

        if self.risk_scorer is not None:
            scored = self.risk_scorer({
                "visual": {"status": "success", "score": max_visual_score},
                "audio": audio_result,
            })
            risk_score = scored.get("risk_score")
            decision = scored.get("decision", "inconclusive")
            if decision == "inconclusive" or risk_score is None:
                is_deepfake = None
                overall_status = "partial"
            else:
                is_deepfake = bool(scored.get("is_suspicious", risk_score >= 0.5))
                overall_status = "success"
        else:
            risk_score = max_visual_score
            is_deepfake = risk_score >= self.visual_detector.threshold
            overall_status = "success"

        return {
            "status": overall_status,
            "is_deepfake": is_deepfake,
            "risk_score": risk_score,
            "max_visual_score": max_visual_score,
            "module_statuses": {
                "visual": "success",
                "audio": audio_result.get("status", "unknown"),
            },
            "audio": audio_result,
            "frames_analyzed": len(sampled_frames),
            "faces_detected": total_faces,
            "face_predictions": all_face_predictions,
        }


def analyze_video_deepfake(
    video_path: Any,
    frames: Optional[Iterable[Dict[str, Any]]] = None,
    face_detector: Optional[FaceDetector] = None,
    visual_detector: Optional[VisualDetector] = None,
    audio_analyzer: Optional[AudioLipSyncAnalyzer] = None,
    frame_sampler: Optional[Callable[[str | Path], Iterable[Dict[str, Any]]]] = None,
    risk_scorer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    pipeline = DeepfakePipeline(
        face_detector=face_detector,
        visual_detector=visual_detector,
        audio_analyzer=audio_analyzer,
        frame_sampler=frame_sampler,
        risk_scorer=risk_scorer,
    )
    return pipeline.run(video_path=video_path, frames=frames)