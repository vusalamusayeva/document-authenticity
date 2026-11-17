import logging
from typing import Dict, Any
from doc_auth.schemas.pipeline_schemas import DeepfakeConfig, DeepfakeAnalysisResult
from doc_auth.deepfake.video_loader import VideoLoader
from doc_auth.deepfake.frame_sampler import FrameSampler
from doc_auth.deepfake.face_detector import FaceDetector
from doc_auth.deepfake.visual_detector import VisualDetector
from doc_auth.deepfake.temporal_analysis import TemporalAnalyzer
from doc_auth.deepfake.audio_analysis import AudioAnalyzer
from doc_auth.deepfake.risk_scoring import RiskScorer

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
        """
        Videonun təhlükəsiz yüklənməsi, frame sampling, vizual/temporal/audio analizi 
        və risk hesablanması prosesini idarə edir. Heç vaxt crash etmir.
        """
        try:
            # 1. Video metadata və təhlükəsizlik yoxlaması
            metadata = self.loader.load_and_validate(video_path)

            # 2. Frame-lərin sampling edilməsi və analizi
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

            # 3. Temporal uyğunsuzluq analizi
            temporal_score = self.temporal_analyzer.compute_temporal_score(frame_analyses)

            # 4. Audio analizi
            audio_result = self.audio_analyzer.analyze_audio(metadata)

            # 5. Risk skorunun hesablanması
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
            # Müvəqqəti faylların təmizlənməsi
            self.sampler.cleanup()