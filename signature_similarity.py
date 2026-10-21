import cv2
import numpy as np
from .exceptions import SignatureStampError

class SignatureSimilarity:

    @staticmethod
    def _resize_and_mask(img1: np.ndarray, img2: np.ndarray, size=(300, 150)) -> tuple:
        r1 = cv2.resize(img1, size, interpolation=cv2.INTER_AREA)
        r2 = cv2.resize(img2, size, interpolation=cv2.INTER_AREA)
        
        if len(r1.shape) == 3:
            r1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
        if len(r2.shape) == 3:
            r2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY)
            
        # Foreground mask (binary inverse threshold) - fon deyil, imza/kontur üzərində işləmək üçün
        _, m1 = cv2.threshold(r1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, m2 = cv2.threshold(r2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return m1, m2

    @classmethod
    def calculate_ssim(cls, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            i1, i2 = cls._resize_and_mask(img1, img2)
            from skimage.metrics import structural_similarity as ssim
            score, _ = ssim(i1, i2, full=True)
            return float(max(0.0, score))
        except Exception:
            return 0.0

    @classmethod
    def calculate_hu_moments(cls, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            i1, i2 = cls._resize_and_mask(img1, img2)
            moments1 = cv2.moments(i1)
            moments2 = cv2.moments(i2)
            hu1 = cv2.HuMoments(moments1).flatten()
            hu2 = cv2.HuMoments(moments2).flatten()
            
            hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
            hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
            
            distance = cv2.norm(hu1, hu2, cv2.NORM_L2)
            similarity = 1.0 / (1.0 + distance)
            return float(max(0.0, similarity))
        except Exception:
            return 0.0

    @classmethod
    def calculate_contour_similarity(cls, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            i1, i2 = cls._resize_and_mask(img1, img2)
            
            cnts1, _ = cv2.findContours(i1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts2, _ = cv2.findContours(i2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not cnts1 or not cnts2:
                return 0.0
            
            c1 = max(cnts1, key=cv2.contourArea)
            c2 = max(cnts2, key=cv2.contourArea)
            
            match_val = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)
            similarity = max(0.0, 1.0 - match_val)
            return float(similarity)
        except Exception:
            return 0.0

    @classmethod
    def calculate_orb_matching(cls, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            i1, i2 = cls._resize_and_mask(img1, img2)
            
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(i1, None)
            kp2, des2 = orb.detectAndCompute(i2, None)
            
            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                return 0.0
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if not matches:
                return 0.0
            
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]
            
            score = len(good_matches) / max(len(kp1), len(kp2), 1)
            return float(min(1.0, score * 2))
        except Exception:
            return 0.0

    @classmethod
    def compare(cls, img1: np.ndarray, img2: np.ndarray) -> dict:
        ssim_score = cls.calculate_ssim(img1, img2)
        hu_score = cls.calculate_hu_moments(img1, img2)
        contour_score = cls.calculate_contour_similarity(img1, img2)
        orb_score = cls.calculate_orb_matching(img1, img2)
        
        total_score = (ssim_score * 0.3) + (hu_score * 0.2) + (contour_score * 0.3) + (orb_score * 0.2)
        
        return {
            "ssim": ssim_score,
            "hu_moments": hu_score,
            "contour_similarity": contour_score,
            "orb_matching": orb_score,
            "total_similarity": float(total_score)
        }