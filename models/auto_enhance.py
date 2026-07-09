"""
AI Auto-Enhancement & Quality Assessment Engine
=================================================
Analyzes passport photos using classical computer vision to evaluate quality
and compute pre-fill sliders for brightness, contrast, sharpness, and saturation.

Also provides a BRISQUE-inspired Image Quality Assessment (IQA) score out of 100.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# Tuning Targets
TARGET_MEAN_LUM     = 135.0   # Ideal average luminance (0-255)
TARGET_STD_LUM      = 55.0    # Ideal RMS contrast
TARGET_SATURATION   = 60.0    # Ideal saturation (0-255)
BLUR_THRESHOLD      = 80.0    # Laplacian variance below this = blurry

# Slider Clamps
MIN_BRIGHTNESS  = 0.7
MAX_BRIGHTNESS  = 1.5
MIN_CONTRAST    = 0.8
MAX_CONTRAST    = 1.4
MIN_SHARPNESS   = 1.0
MAX_SHARPNESS   = 2.0
MIN_SATURATION  = 0.9
MAX_SATURATION  = 1.4


def _pil_to_numpy_rgb(pil_image):
    try:
        img = pil_image.convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        logger.error(f"Failed to convert PIL to numpy in auto_enhance: {e}", exc_info=True)
        return np.zeros((100, 100, 3), dtype=np.uint8)


def _compute_luminance(rgb: np.ndarray) -> np.ndarray:
    try:
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)
        return 0.299 * r + 0.587 * g + 0.114 * b
    except Exception as e:
        logger.error(f"Failed to compute luminance: {e}", exc_info=True)
        return np.zeros(rgb.shape[:2], dtype=np.float32)


def _estimate_noise(lum: np.ndarray) -> float:
    """
    Estimates high-frequency noise using the standard deviation/mean of the residual
    between the original luminance and a Gaussian-blurred version.
    """
    if not CV2_AVAILABLE:
        return 3.0  # Safe default fallback
    try:
        if lum is None or lum.size == 0 or len(lum.shape) < 2:
            return 3.0
        gray = lum.astype(np.uint8)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        residual = cv2.absdiff(gray, blurred)
        return float(np.mean(residual))
    except Exception as e:
        logger.warning(f"Error estimating noise: {e}")
        return 3.0


def auto_enhance(pil_image) -> dict:
    """
    Analyzes a PIL image and returns enhancement recommendations and IQA metrics.
    """
    default_res = {
        "brightness": 1.0,
        "contrast":   1.0,
        "sharpness":  1.0,
        "saturation": 1.0,
        "biometric_score": 90,
        "analysis": {
            "mean_luminance":   TARGET_MEAN_LUM,
            "std_luminance":    TARGET_STD_LUM,
            "noise_score":      2.0,
            "blur_score":       100.0,
            "is_blurry":        False,
            "is_dark":          False,
            "is_overexposed":   False,
            "is_low_contrast":  False,
        }
    }

    if pil_image is None:
        logger.error("auto_enhance received None pil_image")
        return default_res

    try:
        rgb = _pil_to_numpy_rgb(pil_image)
        if rgb is None or rgb.size == 0:
            return default_res
            
        lum = _compute_luminance(rgb)
        if lum is None or lum.size == 0:
            return default_res

        mean_lum = float(np.mean(lum))
        std_lum = float(np.std(lum))
        noise_est = _estimate_noise(lum)

        # 1. Laplacian blur variance
        try:
            if CV2_AVAILABLE:
                lap_var = float(cv2.Laplacian(lum.astype(np.uint8), cv2.CV_64F).var())
            else:
                dx = lum[:, 1:] - lum[:, :-1]
                dy = lum[1:, :] - lum[:-1, :]
                lap_var = float(np.var(dx) + np.var(dy))
        except Exception as lap_e:
            logger.warning(f"Failed to compute Laplacian variance: {lap_e}")
            lap_var = 100.0

        # Calculate slider recommendations
        brightness = 1.0
        if mean_lum > 0:
            brightness = float(np.clip(TARGET_MEAN_LUM / mean_lum, MIN_BRIGHTNESS, MAX_BRIGHTNESS))

        contrast = 1.0
        if std_lum > 0:
            contrast = float(np.clip(TARGET_STD_LUM / std_lum, MIN_CONTRAST, MAX_CONTRAST))

        # Blur-based sharpness boost
        sharpness = 1.0
        if lap_var < BLUR_THRESHOLD and lap_var > 0:
            ratio = lap_var / BLUR_THRESHOLD
            sharpness = float(np.clip(MAX_SHARPNESS - ratio * (MAX_SHARPNESS - MIN_SHARPNESS), MIN_SHARPNESS, MAX_SHARPNESS))

        # Saturation estimation
        mean_sat = 60.0
        try:
            if CV2_AVAILABLE:
                bgr = rgb[:, :, ::-1].copy()
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                mean_sat = float(np.mean(hsv[:, :, 1]))
            else:
                r = rgb[:, :, 0].astype(np.float32)
                g = rgb[:, :, 1].astype(np.float32)
                b = rgb[:, :, 2].astype(np.float32)
                cmax = np.maximum(r, np.maximum(g, b))
                cmin = np.minimum(r, np.minimum(g, b))
                diff = cmax - cmin
                sat = np.where(cmax > 0, diff / cmax * 255.0, 0.0)
                mean_sat = float(np.mean(sat))
        except Exception as sat_e:
            logger.warning(f"Failed to estimate saturation: {sat_e}")

        saturation = 1.0
        if mean_sat > 0:
            saturation = float(np.clip(TARGET_SATURATION / mean_sat, MIN_SATURATION, MAX_SATURATION))

        # ── BRISQUE-inspired Biometric Quality Score Calculation ─────────────────
        # We break quality down into 4 components, each worth 25 points maximum.
        
        # A. Sharpness Score (25 pts): ideal lap_var >= 80
        score_sharp = np.clip((lap_var / max(1.0, BLUR_THRESHOLD)) * 25.0, 0.0, 25.0)
        
        # B. Noise Score (25 pts): lower is better. Ideal noise_est <= 2.5
        # Penalize starting from noise_est = 2.5 up to 6.5
        score_noise = np.clip(25.0 - (max(0.0, noise_est - 2.5) * 6.25), 0.0, 25.0)
        
        # C. Contrast Score (25 pts): ideal std_lum between 40 and 70
        score_contrast = np.clip((std_lum / 45.0) * 25.0, 0.0, 25.0)
        
        # D. Exposure Score (25 pts): ideal mean_lum around 135.
        deviation = abs(mean_lum - TARGET_MEAN_LUM)
        score_exposure = np.clip(25.0 - (deviation * 0.4), 0.0, 25.0)

        # Combined IQA Biometric Quality Score
        iqa_score = float(score_sharp + score_noise + score_contrast + score_exposure)
        iqa_score = round(np.clip(iqa_score, 10.0, 100.0), 0)

        return {
            "brightness": round(brightness, 3),
            "contrast":   round(contrast,   3),
            "sharpness":  round(sharpness,  3),
            "saturation": round(saturation, 3),
            "biometric_score": int(iqa_score),
            "analysis": {
                "mean_luminance":   round(mean_lum, 1),
                "std_luminance":    round(std_lum, 1),
                "noise_score":      round(noise_est, 1),
                "blur_score":       round(lap_var, 1),
                "is_blurry":        lap_var < BLUR_THRESHOLD,
                "is_dark":          mean_lum < TARGET_MEAN_LUM * 0.8,
                "is_overexposed":   mean_lum > TARGET_MEAN_LUM * 1.25,
                "is_low_contrast":  std_lum < TARGET_STD_LUM * 0.7,
            }
        }
    except Exception as e:
        logger.error(f"Error in auto_enhance calculation: {e}", exc_info=True)
        return default_res
