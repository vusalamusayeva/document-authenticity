import argparse
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from doc_auth.deepfake.aggregation import aggregate_frame_scores
from doc_auth.deepfake.face_detector import FaceDetector
from doc_auth.deepfake.temporal_analysis import TemporalAnalyzer, assign_face_tracks
from doc_auth.deepfake.visual_detector import VisualDetector


def run_benchmark(num_frames: int = 30, width: int = 320, height: int = 240) -> Dict[str, Any]:
    detector = FaceDetector()
    visual = VisualDetector()
    temporal = TemporalAnalyzer()

    frames = [
        np.random.default_rng(seed=i).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        for i in range(num_frames)
    ]

    t0 = time.perf_counter()
    scores = []
    track_inputs = []

    for idx, frame in enumerate(frames):
        faces = detector.detect_faces(frame)
        res = visual.analyze_frame(frame, faces=faces)
        score = res.get("max_deepfake_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        track_inputs.append({
            "frame_index": idx,
            "timestamp": idx * 0.1,
            "faces": [
                {
                    "bbox": f.get("bbox", [0, 0, 50, 50]) if isinstance(f, dict) else [0, 0, 50, 50],
                    "score": float(score) if isinstance(score, (int, float)) else 0.0,
                }
                for f in (faces or [])
            ],
        })

    tracked = assign_face_tracks(track_inputs)
    temp_res = temporal.analyze(tracked)
    agg_score = aggregate_frame_scores(scores, method="top_k", top_k=5)
    elapsed_sec = time.perf_counter() - t0
    fps = num_frames / elapsed_sec if elapsed_sec > 0 else 0.0

    return {
        "frames": num_frames,
        "resolution": f"{width}x{height}",
        "elapsed_sec": round(elapsed_sec, 4),
        "fps": round(fps, 2),
        "aggregated_score": agg_score,
        "temporal_summary": temp_res,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake modulu performans benchmarkı")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()

    summary = run_benchmark(num_frames=args.frames, width=args.width, height=args.height)
    print(f"Benchmark nəticəsi: {summary}")


if __name__ == "__main__":
    main()