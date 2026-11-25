# Deepfake Visual Detector Model Card

## 1. Checkpoint & Source
- **Artifact Path:** `models/deepfake_detector.npz`
- **Model Architecture:** Logistic regression / linear visual-artifact scorer over face crop features (`channel_mean`, `channel_std`, normalized `lap_var`, `high_freq_ratio`)
- **Format:** Compressed NumPy archive (`.npz`) with `weights`, `bias`, `laplacian_scale`, `input_size`, `class_mapping`
- **SHA-256:** `040323df408ed884a2325f3161b314562a79988b9489d077ecc8472afab86844`

## 2. Dataset, Split Strategy & Synthetic Data Disclaimer
- **Dataset Source, License & Version:** Pending external licensed real-world deepfake benchmark dataset (e.g., FaceForensics++ / DFDC / Celeb-DF v2 under approved research/commercial license; versioned dataset manifest to be attached in Step 3 release).
- **Split Policy:** Train/validation/test splits must be grouped by **subject/identity and source video** (`GroupShuffleSplit` / identity-disjoint split) so frames from the same person or video never leak across train and test sets.
- **Important Disclaimer:** Synthetic or smoke-test video fixtures used in automated CI/unit checks do **not** represent real-world deepfake performance and must not be cited as production accuracy.

## 3. Evaluation Metrics & Threshold Calibration
- **Required Test Set Metrics (on held-out identity-disjoint test set):**
  - ROC-AUC, PR-AUC, Precision, Recall, F1 Score
  - Confusion Matrix (`[[TN, FP], [FN, TP]]`), False Positive count/rate, False Negative count/rate
- **Current Baseline Smoke Reference (`models/deepfake_detector.npz`):**
  - Input Size: `(224, 224)` (`cv2.INTER_AREA`), Class Mapping: `0 -> real`, `1 -> deepfake`, Default Threshold: `0.5`
  - Held-out check (`[[30, 0], [0, 30]]` on local fixture split): Accuracy `1.0`, Precision `1.0`, Recall `1.0`, F1 `1.0` ? must be re-calibrated on the real identity-disjoint validation set against product FP/FN risk tolerance.
