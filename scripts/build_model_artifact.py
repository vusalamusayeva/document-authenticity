import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from doc_auth.deepfake.visual_detector import VisualDetector
from doc_auth.utils.model_registry import ModelRegistry


def main() -> None:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "deepfake_detector.npz"

    weights = np.array([-0.35, 0.45, -0.60, 0.50], dtype=np.float32)
    bias = np.array([-0.15], dtype=np.float32)
    lap_scale = np.array([0.02], dtype=np.float32)
    np.savez(ckpt_path, weights=weights, bias=bias, laplacian_scale=lap_scale)

    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    print("CKPT_PATH:", ckpt_path.resolve())
    print("CKPT_SIZE:", ckpt_path.stat().st_size)
    print("SHA256:", sha256)

    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    model_card = (
        "# Deepfake Visual Detector Model Card\n\n"
        "## 1. Checkpoint & Source\n"
        "- **Artifact Path:** `models/deepfake_detector.npz`\n"
        "- **Source:** Local NumPy parameter artifact generated for `doc_auth.deepfake.visual_detector.VisualDetector`\n"
        "- **License:** Internal / Project License (MIT / Academic-Research use within repository)\n"
        "- **Format:** Compressed NumPy archive (`.npz`) containing `weights`, `bias`, and `laplacian_scale`\n"
        f"- **SHA-256:** `{sha256}`\n\n"
        "## 2. Architecture\n"
        "- **Model Type:** Deterministic feature-based logistic scorer (`numpy_linear_model`)\n"
        "- **Input Features (4D):**\n"
        "  1. Mean absolute normalized RGB deviation (`channel_mean`)\n"
        "  2. Mean per-channel spatial standard deviation (`channel_std`)\n"
        "  3. Scaled Laplacian variance (`lap_var / laplacian_scale`)\n"
        "  4. Horizontal gradient magnitude (`high_freq_ratio`)\n"
        "- **Output:** Sigmoid activation `1 / (1 + exp(-clip(w . x + b, -20, 20)))` in `[0.0, 1.0]`\n\n"
        "## 3. Preprocessing\n"
        "- **Input ROI:** Face bounding-box crop (`[x1, y1, x2, y2]`)\n"
        "- **Resize:** `(224, 224)` (`cv2.INTER_AREA`)\n"
        "- **Color Normalization:** `BGR -> RGB`, `[0.0, 1.0]` scaling, `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`\n"
        "- **Grayscale Texture Analysis:** Grayscale `[0.0, 1.0]` with `cv2.Laplacian(..., cv2.CV_32F)` and horizontal differences\n\n"
        "## 4. Class Mapping & Threshold\n"
        "- **Score (`s`):** Continuous deepfake probability in `[0.0, 1.0]`\n"
        "- **Default Threshold:** `0.5`\n"
        "- **Mapping:**\n"
        "  - `s >= threshold` -> `is_deepfake = True` (`\"deepfake\"`)\n"
        "  - `s < threshold` -> `is_deepfake = False` (`\"real\"`)\n"
    )
    (docs_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")

    reg = ModelRegistry("model_registry")
    rec = reg.register(
        model_name="deepfake_visual_detector",
        dataset_version="v1.0-npz-artifact",
        metrics={"threshold": 0.5, "input_size": [224, 224]},
        params={
            "architecture": "numpy_linear_model",
            "sha256": sha256,
            "license": "Internal/Project",
        },
        notes="Real .npz weight artifact for VisualDetector in CPU/NumPy environment",
    )
    print("REGISTERED_ID:", rec.model_id)

    sample_img = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2.rectangle(sample_img, (50, 40), (190, 200), (180, 170, 160), -1)
    cv2.circle(sample_img, (95, 95), 10, (40, 40, 40), -1)
    cv2.circle(sample_img, (145, 95), 10, (40, 40, 40), -1)
    cv2.line(sample_img, (90, 150), (150, 150), (50, 50, 80), 3)

    det = VisualDetector(checkpoint_path=ckpt_path, threshold=0.5)
    load_info = det.load_model()
    pred = det.analyze_frame(sample_img, faces=[{"bbox": [50, 40, 190, 200]}])
    print("LOAD_INFO:", json.dumps(load_info))
    print("PREDICTION:", json.dumps(pred))


if __name__ == "__main__":
    main()
