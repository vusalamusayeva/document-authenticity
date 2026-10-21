import cv2
import numpy as np
from .exceptions import SignatureStampError

def analyze_texture(image: np.ndarray, bbox: tuple = None) -> dict:
    try:
        if image is None or image.size == 0:
            raise SignatureStampError("Doku analizi üçün şəkil tapılmadı.")

        if bbox is not None:
            ymin, xmin, ymax, xmax = bbox
            roi = image[ymin:ymax, xmin:xmax]
        else:
            roi = image

        if roi.size == 0:
            roi = image

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_diff = np.mean(np.abs(gray.astype(np.float32) - blur.astype(np.float32)))

        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / float(gray.size) if gray.size > 0 else 0.0

        is_too_smooth = lap_var < 10.0
        has_halo_artifact = noise_diff < 2.0

        forgery_probability = 0.0
        if is_too_smooth:
            forgery_probability += 0.5
        if has_halo_artifact:
            forgery_probability += 0.4
        
        forgery_probability = min(1.0, forgery_probability + (0.1 if edge_density < 0.01 else 0.0))

        return {
            "sharpness_variance": float(lap_var),
            "noise_difference": float(noise_diff),
            "edge_density": float(edge_density),
            "is_too_smooth": bool(is_too_smooth),
            "has_halo_or_blending": bool(has_halo_artifact),
            "digital_manipulation_probability": float(forgery_probability)
        }

    except Exception as e:
        raise SignatureStampError(f"Doku analizi zamanı xəta: {str(e)}")