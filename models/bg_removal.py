"""
Background Removal Engine
==========================
Uses rembg with the isnet-general-use model for clean portrait background removal.
This model handles fine hair edges, light/white hair, and complex textures much
better than u2net_human_seg which is designed for full-body silhouettes.

Post-processing steps:
  1. Soft alpha threshold (very low floor) to preserve hair semi-transparency
  2. Gentle Gaussian blur on alpha mask for natural feathering
  3. Composite over the user-requested solid colour
"""

import logging
import numpy as np
from PIL import Image, ImageColor, ImageFilter

logger = logging.getLogger(__name__)

# ── rembg availability check ──────────────────────────────────────────────────
REMBG_AVAILABLE = False
_rembg_session = None   # Cached session so the model loads only once

try:
    import rembg
    REMBG_AVAILABLE = True
    logger.info("rembg loaded successfully.")
except ImportError:
    logger.warning("rembg not installed. Background removal will be skipped.")


_sessions_cache = {}

def _get_session_by_name(model_name):
    """Get or create inference session for a specific model name."""
    global _sessions_cache
    if not REMBG_AVAILABLE:
        return None
    if model_name not in _sessions_cache or _sessions_cache[model_name] is None:
        try:
            # Unload any previously cached sessions to prevent concurrent models OOM
            for cached_name in list(_sessions_cache.keys()):
                if cached_name != model_name and _sessions_cache[cached_name] is not None:
                    logger.info(f"Unloading cached model '{cached_name}' to free memory.")
                    _sessions_cache[cached_name] = None
                    del _sessions_cache[cached_name]
            import gc
            gc.collect()

            logger.info(f"Creating rembg session for model '{model_name}'...")
            _sessions_cache[model_name] = rembg.new_session(model_name)
            logger.info(f"rembg '{model_name}' session created successfully.")
        except Exception as e:
            logger.warning(f"Could not load model '{model_name}': {e}")
            _sessions_cache[model_name] = None
    return _sessions_cache.get(model_name)


def _get_session():
    """
    Lazily initialise and cache the rembg inference session.
    """
    global _rembg_session
    if _rembg_session is None and REMBG_AVAILABLE:
        from config import REMBG_MODEL
        model_to_use = REMBG_MODEL
        if model_to_use == "u2net_human_seg":
            model_to_use = "isnet-general-use"

        session = _get_session_by_name(model_to_use)
        if session is None:
            # Fallback to other models during initialization
            for fallback_name in ["isnet-general-use", "u2net", "u2netp"]:
                session = _get_session_by_name(fallback_name)
                if session is not None:
                    break
        _rembg_session = session
        if _rembg_session is None:
            logger.error("All rembg models failed to load. Background removal will be unavailable.")
    return _rembg_session


def _clean_alpha_mask(alpha: np.ndarray,
                      blur_radius: float = 1.2,
                      low_thresh: int = 5,
                      high_thresh: int = 240) -> np.ndarray:
    """
    Gently clean up the raw alpha mask from rembg.
    Uses morphological operations to eliminate small background dust and fill holes,
    then applies Gaussian blur feathering.
    """
    try:
        if alpha is None or not hasattr(alpha, 'shape'):
            raise ValueError("Invalid alpha array passed to _clean_alpha_mask")

        cleaned = alpha.astype(np.float32)

        # Threshold to clip low-confidence pixels
        cleaned[cleaned < low_thresh]  = 0.0
        cleaned[cleaned > high_thresh] = 255.0

        cleaned_uint8 = cleaned.astype(np.uint8)
        
        # Morphological opening (remove noise speckles) and closing (fill small voids)
        try:
            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_uint8 = cv2.morphologyEx(cleaned_uint8, cv2.MORPH_OPEN, kernel)
            cleaned_uint8 = cv2.morphologyEx(cleaned_uint8, cv2.MORPH_CLOSE, kernel)
        except Exception as morph_err:
            logger.warning(f"Morphological mask cleaning skipped: {morph_err}")

        pil_alpha = Image.fromarray(cleaned_uint8, mode='L')

        # Feather alpha edges
        if blur_radius > 0:
            pil_alpha = pil_alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return np.array(pil_alpha, dtype=np.uint8)
    except Exception as e:
        logger.error(f"Error cleaning alpha mask: {e}", exc_info=True)
        return alpha if alpha is not None else np.zeros((100, 100), dtype=np.uint8)


def _run_rembg(img: Image.Image, session, alpha_matting: bool) -> Image.Image:
    """Helper to call rembg with or without session and alpha matting."""
    if session is not None:
        return rembg.remove(
            img,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
    else:
        return rembg.remove(
            img,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )


def remove_background(pil_image: Image.Image, bg_color_hex: str = "#FFFFFF") -> Image.Image:
    """
    Remove the background from a passport photo and replace it with a solid colour.
    Downscales the image for mask calculation to prevent ArrayMemoryError/bad allocation
    when alpha_matting is enabled.

    Args:
        pil_image     – Input PIL Image (any mode)
        bg_color_hex  – Target background colour as '#RRGGBB', or 'transparent'
                        to keep the alpha channel.

    Returns:
        PIL Image in RGB mode (or RGBA if transparent was requested).
    """
    if pil_image is None:
        logger.error("remove_background received None image")
        return Image.new("RGB", (300, 400), (255, 255, 255))

    try:
        img_rgba = pil_image.convert("RGBA")
    except Exception as e:
        logger.error(f"Failed to convert image to RGBA: {e}", exc_info=True)
        return pil_image.convert("RGB")

    if not REMBG_AVAILABLE:
        logger.warning("Background removal skipped — rembg not installed.")
        try:
            bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(bg, img_rgba).convert("RGB")
        except Exception:
            return pil_image.convert("RGB")

    w, h = img_rgba.size
    no_bg = None
    # Attempt chain: try alpha_matting ONLY once (primary model).
    # If it fails, all subsequent fallbacks use alpha_matting=False to prevent
    # repeated huge memory allocations from pymatting's sparse Laplacian.
    attempts = [
        ("primary",           True),   # Best quality - try matting once
        ("primary",           False),  # Same model, no matting - fast fallback
        ("isnet-general-use", False),  # Fallback model, no matting
        ("u2netp",            False),  # Lightweight fallback, no matting
    ]

    for model_key, alpha_matting in attempts:
        try:
            if model_key == "primary":
                session = _get_session()
            else:
                session = _get_session_by_name(model_key)

            # Pymatting (used when alpha_matting=True) constructs extremely large sparse matrices.
            # 480px cap gives safety margin; once alpha_matting fails, drop it completely.
            current_max_dim = 480 if alpha_matting else 1024
            if session is None:
                logger.warning(f"Skipping model={model_key} because session could not be initialized.")
                continue

            # Calculate downscaled size before calling rembg
            img_w, img_h = img_rgba.size
            if max(img_w, img_h) > current_max_dim:
                if img_w > img_h:
                    new_w = current_max_dim
                    new_h = int(img_h * (current_max_dim / img_w))
                else:
                    new_h = current_max_dim
                    new_w = int(img_w * (current_max_dim / img_h))
                img_for_rembg = img_rgba.resize((new_w, new_h), Image.Resampling.BILINEAR)
                logger.info(f"Downscaling image for background removal (model={model_key}, alpha_matting={alpha_matting}) from {img_w}x{img_h} to {new_w}x{new_h} to optimize memory.")
            else:
                img_for_rembg = img_rgba

            logger.info(f"Attempting background removal: model={model_key}, alpha_matting={alpha_matting}")
            no_bg = _run_rembg(img_for_rembg, session, alpha_matting)
            if no_bg is not None:
                break
        except (MemoryError, Exception) as e:
            logger.warning(f"Background removal failed with model={model_key}, alpha_matting={alpha_matting}: {e}")
            import gc
            gc.collect()
            continue

    if no_bg is None:
        logger.error("All background removal attempts failed. Returning original image.")
        import gc
        gc.collect()
        if bg_color_hex.strip().lower() == "transparent":
            return img_rgba
        try:
            rgb_color = ImageColor.getcolor(bg_color_hex.strip(), "RGB")
        except Exception:
            rgb_color = (255, 255, 255)
        bg = Image.new("RGBA", img_rgba.size, rgb_color + (255,))
        return Image.alpha_composite(bg, img_rgba).convert("RGB")

    # ── Clean up and resize the alpha mask ──
    try:
        r, g, b, a = no_bg.split()
        alpha_arr = np.array(a, dtype=np.uint8)
        clean_alpha_arr = _clean_alpha_mask(alpha_arr)
        clean_alpha = Image.fromarray(clean_alpha_arr, mode='L')
    except Exception as e:
        logger.warning(f"Alpha mask cleanup failed ({e}), using raw rembg alpha channel.")
        clean_alpha = no_bg.split()[3]

    if clean_alpha.size != (w, h):
        # Resize alpha mask back to original image size
        clean_alpha = clean_alpha.resize((w, h), Image.Resampling.BILINEAR)

    # ── Combine the high-res original image with the upscaled mask ──
    orig_r, orig_g, orig_b, _ = img_rgba.split()
    no_bg_highres = Image.merge("RGBA", (orig_r, orig_g, orig_b, clean_alpha))

    if bg_color_hex.strip().lower() == "transparent":
        import gc
        gc.collect()
        return no_bg_highres

    # ── Composite over solid background colour ──
    try:
        rgb_color = ImageColor.getcolor(bg_color_hex.strip(), "RGB")
    except (ValueError, AttributeError, Exception):
        logger.warning(f"Invalid hex color '{bg_color_hex}', defaulting to white.")
        rgb_color = (255, 255, 255)

    try:
        bg = Image.new("RGBA", img_rgba.size, rgb_color + (255,))
        composited = Image.alpha_composite(bg, no_bg_highres)
        return composited.convert("RGB")
    finally:
        import gc
        gc.collect()
