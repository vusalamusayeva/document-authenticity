from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from doc_auth.deepfake.aggregation import aggregate_visual_predictions
from doc_auth.deepfake.audio_analysis import AudioLipSyncAnalyzer
from doc_auth.deepfake.exceptions import ModelUnavailableError, VideoValidationError
from doc_auth.deepfake.face_detector import FaceDetector
from doc_auth.deepfake.frame_sampler import FrameSampler
from doc_auth.deepfake.risk_scoring import DeepfakeRiskScorer
from doc_auth.deepfake.temporal_analysis import TemporalAnalyzer, assign_face_tracks
from doc_auth.deepfake.video_loader import validate_video
from doc_auth.deepfake.visual_detector import VisualDetector


class DeepfakePipeline:
    """
    Standart production konfiqurasiyası ilə video validation, frame sampling,
    face detection, visual inference, temporal analysis, aggregation,
    audio/lip-sync və risk scoring komponentlərini birləşdirən pipeline.
    """

    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        visual_detector: Optional[VisualDetector] = None,
        audio_analyzer: Optional[AudioLipSyncAnalyzer] = None,
        frame_sampler: Optional[Callable[[str | Path], Iterable[Dict[str, Any]]]] = None,
        temporal_analyzer: Optional[TemporalAnalyzer] = None,
        risk_scorer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        video_validator: Optional[Callable[[str | Path], Any]] = None,
        aggregator: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.face_detector = face_detector or FaceDetector()
        self.visual_detector = visual_detector
        self.audio_analyzer = audio_analyzer or AudioLipSyncAnalyzer()

        self._default_sampler = FrameSampler()
        self.frame_sampler = frame_sampler or self._default_sampler.sample

        self.temporal_analyzer = temporal_analyzer or TemporalAnalyzer()
        self._default_risk_scorer = DeepfakeRiskScorer()
        self.risk_scorer = risk_scorer or self._default_risk_scorer.score

        self.video_validator = video_validator or validate_video
        self.aggregator = aggregator or aggregate_visual_predictions

    def _run_validator(self, path: Path, allow_missing_stub_path: bool = False) -> None:
        if self.video_validator is None:
            return
        if allow_missing_stub_path and not path.exists():
            return
        try:
            self.video_validator(path)
        except TypeError:
            self.video_validator(str(path))

    def _sample_frames(self, path: Path, frames: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if frames is not None:
            raw_stream = frames
        else:
            try:
                raw_stream = self.frame_sampler(path)
            except TypeError:
                raw_stream = self.frame_sampler(str(path))

        sampled: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_stream):
            if isinstance(item, np.ndarray):
                sampled.append({
                    "frame": item,
                    "frame_index": idx,
                    "timestamp": None,
                })
            elif isinstance(item, dict) and "frame" in item:
                entry = dict(item)
                entry.setdefault("frame_index", idx)
                entry.setdefault("timestamp", None)
                sampled.append(entry)
        return sampled

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
                    "validator": "failed",
                    "visual": "failed",
                    "temporal": "not_run",
                    "audio": "not_run",
                },
                "frames_analyzed": 0,
                "faces_detected": 0,
            }

        path = Path(video_path)
        custom_sampler = self.frame_sampler is not self._default_sampler.sample

        try:
            self._run_validator(
                path,
                allow_missing_stub_path=(frames is not None or custom_sampler),
            )
            validator_status = "passed"
        except VideoValidationError as exc:
            return {
                "status": "failed",
                "is_deepfake": None,
                "risk_score": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "module_statuses": {
                    "validator": "failed",
                    "visual": "failed",
                    "temporal": "not_run",
                    "audio": "not_run",
                },
                "frames_analyzed": 0,
                "faces_detected": 0,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "is_deepfake": None,
                "risk_score": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "module_statuses": {
                    "validator": "failed",
                    "visual": "failed",
                    "temporal": "not_run",
                    "audio": "not_run",
                },
                "frames_analyzed": 0,
                "faces_detected": 0,
            }

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
                    "validator": validator_status,
                    "visual": "model_unavailable",
                    "temporal": "not_run",
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
                    "validator": validator_status,
                    "visual": "model_unavailable",
                    "temporal": "not_run",
                    "audio": audio_result.get("status", "unknown"),
                },
                "audio": audio_result,
                "frames_analyzed": 0,
                "faces_detected": 0,
            }

        try:
            sampled_frames = self._sample_frames(path, frames=frames)
        except VideoValidationError as exc:
            return {
                "status": "failed",
                "is_deepfake": None,
                "risk_score": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "module_statuses": {
                    "validator": validator_status,
                    "sampler": "failed",
                    "visual": "failed",
                    "temporal": "not_run",
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
                    "validator": validator_status,
                    "sampler": "empty",
                    "visual": "no_frames",
                    "temporal": "not_run",
                    "audio": audio_result.get("status", "unknown"),
                },
                "audio": audio_result,
                "frames_analyzed": 0,
                "faces_detected": 0,
            }

        total_faces = 0
        all_face_predictions: List[Dict[str, Any]] = []
        frame_face_items: List[Dict[str, Any]] = []

        for idx, item in enumerate(sampled_frames):
            frame = item["frame"]
            frame_idx = item.get("frame_index", idx)
            ts = item.get("timestamp")
            faces = self.face_detector.detect_faces(frame)
            if not faces:
                frame_face_items.append({
                    "frame_index": frame_idx,
                    "timestamp": ts if ts is not None else float(idx),
                    "faces": [],
                })
                continue
            total_faces += len(faces)
            frame_pred = self.visual_detector.analyze_frame(frame, faces=faces)
            frame_faces_payload: List[Dict[str, Any]] = []
            if frame_pred.get("status") == "success":
                for i_f, fp in enumerate(frame_pred.get("face_predictions", [])):
                    entry = dict(fp)
                    entry["frame_index"] = frame_idx
                    entry["timestamp"] = ts
                    all_face_predictions.append(entry)
                    frame_faces_payload.append({
                        "bbox": entry.get("bbox", faces[i_f].get("bbox", [0, 0, 0, 0]) if i_f < len(faces) else [0, 0, 0, 0]),
                        "score": entry.get("score", 0.0),
                        "is_deepfake": entry.get("is_deepfake", False),
                    })
            frame_face_items.append({
                "frame_index": frame_idx,
                "timestamp": ts if ts is not None else float(idx),
                "faces": frame_faces_payload,
            })

        if total_faces == 0 or not all_face_predictions:
            return {
                "status": "no_face",
                "is_deepfake": None,
                "risk_score": None,
                "module_statuses": {
                    "validator": validator_status,
                    "sampler": "success",
                    "visual": "no_face",
                    "temporal": "not_run",
                    "audio": audio_result.get("status", "unknown"),
                },
                "audio": audio_result,
                "frames_analyzed": len(sampled_frames),
                "faces_detected": 0,
            }

        try:
            tracks = assign_face_tracks(frame_face_items)
            temporal_summary = self.temporal_analyzer.analyze(tracks)
            temporal_status = "success"
        except Exception as exc:
            temporal_summary = {"status": "error", "error": str(exc), "temporal_score": None}
            temporal_status = "error"

        try:
            agg_result = self.aggregator(all_face_predictions)
            visual_agg_score = float(
                agg_result.get(
                    "aggregated_score",
                    agg_result.get("visual_score", max(p["score"] for p in all_face_predictions)),
                )
            )
            agg_status = "success"
        except Exception:
            visual_agg_score = float(max(p["score"] for p in all_face_predictions))
            agg_result = {"aggregated_score": visual_agg_score}
            agg_status = "fallback_max"

        max_visual_score = float(max(p["score"] for p in all_face_predictions))
        temporal_score = temporal_summary.get("temporal_score")

        signals_payload: Dict[str, Any] = {
            "visual": {
                "status": "success",
                "score": visual_agg_score,
                "max_score": max_visual_score,
                "aggregation": agg_result,
            },
            "temporal": {
                "status": temporal_status,
                "score": temporal_score,
                "summary": temporal_summary,
            },
            "audio": audio_result,
        }

        if self.risk_scorer is not None:
            scored = self.risk_scorer(signals_payload)
            risk_score = scored.get("risk_score")
            decision = scored.get("decision", "inconclusive")
            if decision == "inconclusive" or risk_score is None:
                is_deepfake = None
                overall_status = "partial"
            else:
                is_deepfake = bool(scored.get("is_suspicious", risk_score >= 0.5))
                overall_status = "success"
        else:
            risk_score = visual_agg_score
            is_deepfake = risk_score >= self.visual_detector.threshold
            overall_status = "success"

        return {
            "status": overall_status,
            "is_deepfake": is_deepfake,
            "risk_score": risk_score,
            "max_visual_score": max_visual_score,
            "visual_score": visual_agg_score,
            "temporal_analysis": temporal_summary,
            "module_statuses": {
                "validator": validator_status,
                "sampler": "success",
                "visual": "success",
                "aggregation": agg_status,
                "temporal": temporal_status,
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
    temporal_analyzer: Optional[TemporalAnalyzer] = None,
    risk_scorer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_validator: Optional[Callable[[str | Path], Any]] = None,
    aggregator: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    pipeline = DeepfakePipeline(
        face_detector=face_detector,
        visual_detector=visual_detector,
        audio_analyzer=audio_analyzer,
        frame_sampler=frame_sampler,
        temporal_analyzer=temporal_analyzer,
        risk_scorer=risk_scorer,
        video_validator=video_validator,
        aggregator=aggregator,
    )
    return pipeline.run(video_path=video_path, frames=frames)