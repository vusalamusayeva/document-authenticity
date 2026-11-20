# Deepfake Visual Detector Model Card

## 1. Checkpoint & Source
- **Artifact Path:** `models/deepfake_detector.npz`
- **Source:** Local NumPy parameter artifact generated for `doc_auth.deepfake.visual_detector.VisualDetector`
- **License:** Internal / Project License (MIT / Academic-Research use within repository)
- **Format:** Compressed NumPy archive (`.npz`) containing `weights`, `bias`, and `laplacian_scale`
- **SHA-256:** `d4311015f2b40a1c36acd8583fee2486f4b9a1b2274438cd47f169f4f0db1b92`

## 2. Architecture
- **Model Type:** Deterministic feature-based logistic scorer (`numpy_linear_model`)
- **Input Features (4D):**
  1. Mean absolute normalized RGB deviation (`channel_mean`)
  2. Mean per-channel spatial standard deviation (`channel_std`)
  3. Scaled Laplacian variance (`lap_var / laplacian_scale`)
  4. Horizontal gradient magnitude (`high_freq_ratio`)
- **Output:** Sigmoid activation `1 / (1 + exp(-clip(w . x + b, -20, 20)))` in `[0.0, 1.0]`

## 3. Preprocessing
- **Input ROI:** Face bounding-box crop (`[x1, y1, x2, y2]`)
- **Resize:** `(224, 224)` (`cv2.INTER_AREA`)
- **Color Normalization:** `BGR -> RGB`, `[0.0, 1.0]` scaling, `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`
- **Grayscale Texture Analysis:** Grayscale `[0.0, 1.0]` with `cv2.Laplacian(..., cv2.CV_32F)` and horizontal differences

## 4. Class Mapping & Threshold
- **Score (`s`):** Continuous deepfake probability in `[0.0, 1.0]`
- **Default Threshold:** `0.5`
- **Mapping:**
  - `s >= threshold` -> `is_deepfake = True` (`"deepfake"`)
  - `s < threshold` -> `is_deepfake = False` (`"real"`)
