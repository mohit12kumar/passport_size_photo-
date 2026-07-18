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


def _apply_skin_preservation_denoise(img_bgr):
    """
    Applies bilateral filtering (edge-preserving denoise) selectively to skin tones.
    This preserves hair, eyebrow, and eye details while smoothing out skin blemishes.
    """
    try:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # HSV skin color range boundaries
        lower_skin = np.array([0, 15, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Soften mask boundary
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        mask_norm = mask.astype(np.float32) / 255.0
        mask_norm = np.expand_dims(mask_norm, axis=2)
        
        # Run bilateral filter (edge preserving)
        denoised = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=20, sigmaSpace=20)
        
        # Blend skin regions with denoised, keep others original
        blended = img_bgr * (1.0 - mask_norm) + denoised * mask_norm
        return np.clip(blended, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.warning(f"Skin preservation denoise skipped: {e}")
        return img_bgr

def _apply_hdr_boost(img_bgr):
    """
    Enhances exposure locally by boosting shadows and pulling down hot highlights.
    Prevents face shadows and direct camera flash overexposure.
    """
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        
        # Shadow mask (L < 110)
        shadow_mask = cv2.threshold(l_ch, 110, 255, cv2.THRESH_BINARY_INV)[1]
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0).astype(np.float32) / 255.0
        
        # Highlight mask (L > 215)
        highlight_mask = cv2.threshold(l_ch, 215, 255, cv2.THRESH_BINARY)[1]
        highlight_mask = cv2.GaussianBlur(highlight_mask, (15, 15), 0).astype(np.float32) / 255.0
        
        l_float = l_ch.astype(np.float32)
        # Gentle shadow boost (up to 15%) & highlight reduction (up to 8%)
        boosted_l = l_float + (255.0 - l_float) * 0.15 * shadow_mask
        toned_l = boosted_l - boosted_l * 0.08 * highlight_mask
        
        l_new = np.clip(toned_l, 0, 255).astype(np.uint8)
        lab_new = cv2.merge([l_new, a_ch, b_ch])
        return cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.warning(f"HDR boost skipped: {e}")
        return img_bgr

def enhance_image(pil_image, brightness=1.0, contrast=1.0, sharpness=1.0, saturation=1.0,
                  denoise=False, white_balance=False, auto_clahe=False, gamma=1.0):
    """
    Enhance portrait photography using advanced CV algorithms and PIL fallback adjustments.
    """
    if pil_image is None:
        return None

    try:
        img = pil_image.copy()
    except Exception as e:
        logger.error(f"Failed to copy image in enhance_image: {e}", exc_info=True)
        return pil_image

    # ── Step 1: White Balance (AWB) ────────────────────────
    if white_balance and CV2_AVAILABLE:
        try:
            img_np = np.array(img.convert("RGB"))
            balanced_np = _apply_gray_world_awb(img_np)
            img = Image.fromarray(balanced_np)
        except Exception as e:
            logger.warning(f"Failed to apply AWB: {e}")

    # ── Step 2: Denoise with Skin & Edge Preservation ──────
    if denoise and CV2_AVAILABLE:
        try:
            img_np = np.array(img.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            denoised_bgr = _apply_skin_preservation_denoise(img_bgr)
            denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(denoised_rgb)
        except Exception as e:
            logger.warning(f"Failed to apply denoising: {e}")

    # ── Step 3: Local Contrast & HDR Exposure Boost ────────
    if auto_clahe and CV2_AVAILABLE:
        try:
            img_np = np.array(img.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Apply shadow/highlight mapping
            hdr_bgr = _apply_hdr_boost(img_bgr)
            
            # Apply CLAHE on L-channel
            lab = cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2LAB)
            l_ch, a, b = cv2.split(lab)
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe_obj.apply(l_ch)
            limg = cv2.merge((cl, a, b))
            rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            img = Image.fromarray(rgb)
        except Exception as e:
            logger.warning(f"Failed to apply CLAHE/HDR boost: {e}")

    # ── Step 4: Gamma Correction ─────────────────────────
    if gamma != 1.0 and CV2_AVAILABLE:
        try:
            img_np = np.array(img.convert("RGB"))
            gamma_np = apply_gamma_correction(img_np, gamma)
            img = Image.fromarray(gamma_np)
        except Exception as e:
            logger.warning(f"Failed to apply Gamma Correction: {e}")

    # Normalise slider inputs
    try:
        b_val = max(0.0, min(10.0, float(brightness)))
        c_val = max(0.0, min(10.0, float(contrast)))
        sa_val = max(0.0, min(10.0, float(saturation)))
        sh_val = max(-10.0, min(10.0, float(sharpness)))
    except Exception:
        b_val, c_val, sa_val, sh_val = 1.0, 1.0, 1.0, 1.0

    # ── Step 5: PIL adjustments ──────────────────────────
    if b_val != 1.0:
        try:
            img = ImageEnhance.Brightness(img).enhance(b_val)
        except Exception as e:
            logger.warning(f"Failed to adjust brightness: {e}")

    if c_val != 1.0:
        try:
            img = ImageEnhance.Contrast(img).enhance(c_val)
        except Exception as e:
            logger.warning(f"Failed to adjust contrast: {e}")

    if sa_val != 1.0:
        try:
            img = ImageEnhance.Color(img).enhance(sa_val)
        except Exception as e:
            logger.warning(f"Failed to adjust saturation: {e}")

    if sh_val != 1.0:
        try:
            img = ImageEnhance.Sharpness(img).enhance(sh_val)
        except Exception as e:
            logger.warning(f"Failed to adjust sharpness: {e}")

    return img
