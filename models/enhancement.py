import numpy as np
from PIL import Image, ImageEnhance
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def apply_gamma_correction(img_np: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Applies gamma correction to a numpy RGB/BGR image.
    gamma > 1.0 brightens midtones/shadows.
    gamma < 1.0 darkens midtones.
    """
    if gamma == 1.0 or not CV2_AVAILABLE:
        return img_np
    try:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img_np, table)
    except Exception as e:
        logger.warning(f"Failed to apply gamma correction: {e}")
        return img_np


def _apply_gray_world_awb(img_np: np.ndarray) -> np.ndarray:
    """
    Applies the Gray World AWB algorithm to balance the colors.
    Computes the average of R, G, B channels and scales each channel
    so that their averages match the overall gray mean.
    """
    try:
        if img_np is None or not hasattr(img_np, 'shape') or len(img_np.shape) < 3:
            raise ValueError("Invalid numpy array passed to AWB")
            
        r = img_np[:, :, 0].astype(np.float32)
        g = img_np[:, :, 1].astype(np.float32)
        b = img_np[:, :, 2].astype(np.float32)
        
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)
        
        if mean_r == 0 or mean_g == 0 or mean_b == 0:
            return img_np
            
        mean_gray = (mean_r + mean_g + mean_b) / 3.0
        
        scale_r = np.clip(mean_gray / mean_r, 0.8, 1.25)
        scale_g = np.clip(mean_gray / mean_g, 0.8, 1.25)
        scale_b = np.clip(mean_gray / mean_b, 0.8, 1.25)
        
        r_new = np.clip(r * scale_r, 0, 255).astype(np.uint8)
        g_new = np.clip(g * scale_g, 0, 255).astype(np.uint8)
        b_new = np.clip(b * scale_b, 0, 255).astype(np.uint8)
        
        return np.stack([r_new, g_new, b_new], axis=2)
    except Exception as e:
        logger.error(f"AWB computation failed: {e}", exc_info=True)
        return img_np


def enhance_image(pil_image, brightness=1.0, contrast=1.0, sharpness=1.0, saturation=1.0,
                  denoise=False, white_balance=False, auto_clahe=False, gamma=1.0):
    """
    Enhance the image's brightness, contrast, sharpness, saturation using PIL.
    Optionally applies OpenCV-based Denoising, White Balance, CLAHE, and Gamma Correction.
    """
    if pil_image is None:
        logger.error("enhance_image received None image")
        return None

    try:
        img = pil_image.copy()
    except Exception as e:
        logger.error(f"Failed to copy PIL image in enhance_image: {e}", exc_info=True)
        return pil_image
    
    # ── Step 1: Optional White Balance (AWB) ────────────────────────
    if white_balance and CV2_AVAILABLE:
        try:
            img_np = np.array(img.convert("RGB"))
            balanced_np = _apply_gray_world_awb(img_np)
            img = Image.fromarray(balanced_np)
        except Exception as e:
            logger.warning(f"Failed to apply AWB: {e}")
            
    # ── Step 2: Optional Denoising (fastNlMeansDenoisingColored) ─────
    if denoise and CV2_AVAILABLE:
        try:
            img_np = np.array(img.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            denoised_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
            denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(denoised_rgb)
        except Exception as e:
            logger.warning(f"Failed to apply denoising: {e}")

    # ── Step 2.5: Optional CLAHE (Adaptive Brightness & Contrast) ─────
    if auto_clahe and CV2_AVAILABLE:
        try:
            img_np = np.array(img.convert("RGB"))
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b_ch = cv2.split(lab)
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe_obj.apply(l)
            limg = cv2.merge((cl, a, b_ch))
            rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            img = Image.fromarray(rgb)
        except Exception as e:
            logger.warning(f"Failed to apply CLAHE: {e}")
            
    # ── Step 2.7: Optional Gamma Correction ─────────────────────────
    if gamma != 1.0 and CV2_AVAILABLE:
        try:
            img_np = np.array(img.convert("RGB"))
            gamma_np = apply_gamma_correction(img_np, gamma)
            img = Image.fromarray(gamma_np)
        except Exception as e:
            logger.warning(f"Failed to apply Gamma Correction: {e}")
            
    # Normalize and clamp inputs to prevent PIL error/crash
    try:
        b_val = max(0.0, min(10.0, float(brightness)))
        c_val = max(0.0, min(10.0, float(contrast)))
        sa_val = max(0.0, min(10.0, float(saturation)))
        sh_val = max(-10.0, min(10.0, float(sharpness)))
    except Exception as e:
        logger.error(f"Input type conversion failed in enhance_image parameters: {e}")
        b_val, c_val, sa_val, sh_val = 1.0, 1.0, 1.0, 1.0

    # ── Step 3: Brightness ──────────────────────────────────────────
    if b_val != 1.0:
        try:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(b_val)
        except Exception as e:
            logger.warning(f"Failed to apply brightness enhancer: {e}")
        
    # ── Step 4: Contrast ─────────────────────────────────────────────
    if c_val != 1.0:
        try:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(c_val)
        except Exception as e:
            logger.warning(f"Failed to apply contrast enhancer: {e}")
        
    # ── Step 5: Saturation ───────────────────────────────────────────
    if sa_val != 1.0:
        try:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(sa_val)
        except Exception as e:
            logger.warning(f"Failed to apply saturation enhancer: {e}")
        
    # ── Step 6: Sharpness ────────────────────────────────────────────
    if sh_val != 1.0:
        try:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sh_val)
        except Exception as e:
            logger.warning(f"Failed to apply sharpness enhancer: {e}")
        
    return img
