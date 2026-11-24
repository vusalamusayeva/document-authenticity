# Deepfake Visual Detector Model Card

## 1. Checkpoint & Source
- **Artifact Path:** `models/deepfake_detector.npz`
- **Training Method:** LogisticRegression fitted on stratified train split (75% of 240 face crops), evaluated on held-out 25% test split
- **Format:** Compressed NumPy archive (`.npz`) with `weights`, `bias`, `laplacian_scale`, `input_size`, `class_mapping`
- **SHA-256:** `040323df408ed884a2325f3161b314562a79988b9489d077ecc8472afab86844`

## 2. Held-Out Test Evaluation
- **Accuracy:** `1.0`
- **Precision:** `1.0`
- **Recall:** `1.0`
- **F1 Score:** `1.0`
- **Confusion Matrix (`[[TN, FP], [FN, TP]]`):** `[[30, 0], [0, 30]]`

## 3. Preprocessing & Class Mapping
- **Input Size:** `(224, 224)` (`cv2.INTER_AREA`)
- **Class Mapping:** `0 -> real`, `1 -> deepfake` (threshold `0.5`)
