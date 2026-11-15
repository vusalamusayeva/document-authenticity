import pytest
from src.doc_auth.deepfake.pipeline import DeepfakePipeline
from src.doc_auth.deepfake.schemas import DeepfakeConfig

def test_pipeline_non_existent_video():
    pipeline = DeepfakePipeline()
    result = pipeline.run("non_existent_video_path_12345.mp4")
    assert result.status == "failed"
    assert result.error_message is not None

def test_pipeline_config_limits():
    config = DeepfakeConfig(max_file_size_mb=1, max_duration_sec=5.0)
    pipeline = DeepfakePipeline(config)
    assert pipeline.config.max_file_size_mb == 1
    assert pipeline.config.max_duration_sec == 5.0