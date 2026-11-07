import logging
from typing import Dict, Any
from .schemas import DeepfakeConfig, DeepfakeAnalysisResult
from .video_loader import VideoLoader
from .frame_sampler import FrameSampler
from .face_detector import FaceDetector
from .visual_detector import VisualDetector
from .temporal_analysis import TemporalAnalyzer
from .audio_analysis import AudioAnalyzer
from .risk_scoring import RiskScorer

logger = logging.getLogger(__name__)

class DeepfakePipeline:
    def __init__(self, config: DeepfakeConfig = None):
        self.config = config or DeepfakeConfig()
        self.loader = VideoLoader(self.config)
        self.sampler = FrameSampler(self.config)
        self.detector = VisualDetector()
        self.temporal_analyzer = TemporalAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.risk_scorer = RiskScorer()

    def run(self, video_path: str) -> DeepfakeAnalysisResult:

        try:
            metadata = self.loader.load_and_validate(video_path)

            frame_analyses = []
            max_visual_score = 0.0

            for sampled in self.sampler.sample_frames(video_path):
                frame = sampled["frame"]
                visual_res = self.detector.analyze_frame(frame)
                
                max_visual_score = max(max_visual_score, visual_res["max_deepfake_score"])
                frame_analyses.append({
                    "frame_index": sampled["frame_index"],
                    "timestamp_ms": sampled["timestamp_ms"],
                    "face_detected": visual_res["face_detected"],
                    "max_deepfake_score": visual_res["max_deepfake_score"],
                    "bounding_boxes": [f["bbox"] for f in visual_res["faces"]]
                })

            temporal_score = self.temporal_analyzer.compute_temporal_score(frame_analyses)

            audio_result = self.audio_analyzer.analyze_audio(metadata)

            risk_result = self.risk_scorer.compute_risk(max_visual_score, temporal_score, audio_result)

            return DeepfakeAnalysisResult(
                status="success",
                metadata=metadata,
                frames_analyzed=len(frame_analyses),
                temporal_inconsistency_score=temporal_score,
                audio_status=audio_result,
                overall_risk_score=risk_result["overall_risk_score"],
                is_suspicious=risk_result["is_suspicious"]
            )

        except Exception as e:
            logger.error(f"Deepfake pipeline xətası: {str(e)}")
            return DeepfakeAnalysisResult(
                status="failed",
                error_message=str(e)
            )
        finally:
            self.sampler.cleanup()