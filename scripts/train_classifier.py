import argparse
from pathlib import Path
import cv2
import numpy as np
from doc_auth.classification.document_classifier import DocumentClassifier


def generate_synthetic_dataset() -> tuple[list[np.ndarray], list[str]]:
    base = np.ones((320, 1020, 3), dtype=np.uint8) * 245
    cv2.rectangle(base, (20, 20), (1000, 300), (40, 40, 40), 2)
    cv2.putText(base, "DOCUMENT AUTH SAMPLE", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(base, "ID: DOC-AZ-001 | DATE: 27.11.2026", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    h, w = base.shape[:2]
    images = [
        base,
        cv2.convertScaleAbs(base, alpha=1.08, beta=8),
        base[int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)],
        cv2.flip(base, 1),
    ]
    labels = ["full_scan", "full_scan", "cropped_scan", "mirrored_scan"]
    return images, labels


def load_from_directory(data_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    images: list[np.ndarray] = []
    labels: list[str] = []
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in valid_exts:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    images.append(img)
                    labels.append(class_dir.name)
    return images, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and serialize DocumentClassifier.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with class subfolders (e.g. data/raw/classification)")
    parser.add_argument("--output", type=Path, default=Path("models/document_classifier.pkl"), help="Output .pkl model path")
    args = parser.parse_args()

    images: list[np.ndarray] = []
    labels: list[str] = []

    if args.data_dir and args.data_dir.exists():
        images, labels = load_from_directory(args.data_dir)
        print(f"Loaded {len(images)} images from {args.data_dir}")

    if len(set(labels)) < 2:
        print("Insufficient labeled classes in data-dir; using synthetic baseline dataset.")
        images, labels = generate_synthetic_dataset()

    clf = DocumentClassifier()
    clf.fit(images, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(args.output))
    print(f"Model saved successfully to {args.output} (classes: {sorted(set(labels))})")


if __name__ == "__main__":
    main()
