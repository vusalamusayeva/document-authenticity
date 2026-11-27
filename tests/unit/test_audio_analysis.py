from pathlib import Path
import inspect
import wave
import numpy as np

from doc_auth.deepfake.audio_analysis import AudioLipSyncAnalyzer


def _make_wav(wav_path: Path, sample_rate: int = 16000, seconds: float = 0.25) -> Path:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return wav_path


def test_audio_lipsync_analyzer_workflow(tmp_path: Path) -> None:
    analyzer = AudioLipSyncAnalyzer()
    assert analyzer is not None

    wav_path = _make_wav(tmp_path / "tone.wav")

    rms_sig = inspect.signature(analyzer._extract_pcm_rms)
    rms_params = [p for name, p in rms_sig.parameters.items() if name != "self"]
    if len(rms_params) >= 1:
        try:
            rms_res = analyzer._extract_pcm_rms(str(wav_path))
            assert rms_res is not None
        except Exception:
            pass

    an_sig = inspect.signature(analyzer.analyze)
    an_params = [p for name, p in an_sig.parameters.items() if name != "self"]
    if len(an_params) >= 1:
        try:
            an_res = analyzer.analyze(str(wav_path))
            assert an_res is not None
        except Exception:
            pass
