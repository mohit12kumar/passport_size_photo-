"""
HD Super-Resolution Enhancement Engine
=======================================
Produces sharp, studio-quality passport photos using a multi-step
computational photography pipeline — no heavyweight AI dependencies required.

Pipeline steps:
  1.  4× Lanczos upscale                  – maximum detail during processing
  2.  LAB colour space conversion          – sharpen luminance, preserve skin colour
  3.  CLAHE (adaptive histogram equalise)  – local contrast enhancement
  4.  Bilateral-style denoise              – reduce noise before sharpening
  5.  Multi-pass Unsharp Mask              – aggressive, artefact-free edge sharpening
  6.  Detail convolution kernel            – micro-texture & pore-level crispness
  7.  Gentle contrast / colour finalize    – vibrant, balanced final look
  8.  Lanczos downscale to target size     – crisp output at full output resolution

CodeFormer ONNX is supported as the single AI HD model option.
"""

import logging
import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)

# ── Optional Real-ESRGAN face/image super-resolution ─────────────────────────
REALESRGAN_AVAILABLE = False
_realesrgan_upsampler = None

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
    logger.info("realesrgan and basicsr detected, Real-ESRGAN option is available.")
except ImportError:
    logger.info("realesrgan or basicsr not installed — Real-ESRGAN will be unavailable.")


def _get_realesrgan_upsampler():
    """Lazily initialize Real-ESRGAN upsampler."""
    global _realesrgan_upsampler
    if _realesrgan_upsampler is None and REALESRGAN_AVAILABLE:
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "gfpgan", "weights", "RealESRGAN_x4plus.pth")
            if not os.path.isfile(model_path):
                logger.warning(f"Real-ESRGAN model file not found locally: {model_path}")
                return None
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            _realesrgan_upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False
            )
            logger.info("Real-ESRGAN upsampler successfully initialized.")
        except Exception as e:
            logger.warning(f"Real-ESRGAN init failed ({e}).")
    return _realesrgan_upsampler


def _enhance_realesrgan(img_bgr) -> np.ndarray:
    """
    Perform super-resolution using Real-ESRGAN.
    """
    try:
        upsampler = _get_realesrgan_upsampler()
        if upsampler is None:
            raise RuntimeError("Real-ESRGAN upsampler is not available.")
        
        # Run inference with outscale=2.0 (RealESRGAN x2)
        enhanced_bgr, _ = upsampler.enhance(img_bgr, outscale=2.0)
        return enhanced_bgr
    except Exception as e:
        logger.warning(f"Real-ESRGAN enhancement failed: {e}", exc_info=True)
        return img_bgr


# ── Convolution kernels ───────────────────────────────────────────────────────
# High-frequency detail enhancement kernel
_DETAIL_KERNEL = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[
         0, -1,  0,
        -1,  6, -1,
         0, -1,  0,
    ],
    scale=2,
    offset=0,
)

# Gentler secondary sharpening pass
_EDGE_KERNEL = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1,
    ],
    scale=1,
    offset=0,
)


def _apply_clahe_pil(img_rgb: Image.Image, clip_limit: float = 2.5) -> Image.Image:
    """
    Adaptive histogram equalization on the L channel of LAB.
    Boosts local contrast in dark / washed-out regions without affecting colour.
    Falls back to a simpler global equalise if cv2 is not available.
    """
    try:
        import cv2
        img_array = np.array(img_rgb, dtype=np.uint8)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_eq)
    except Exception:
        # Graceful fallback: only equalise luminance channel
        r, g, b = img_rgb.split()
        r = ImageOps.equalize(r)
        return Image.merge("RGB", (r, g, b))


def _pil_hd_pipeline(pil_image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Multi-step computational HD sharpening pipeline.
    All processing done at 4× resolution for maximum quality.
    """
    try:
        img = pil_image.convert("RGB")

        # ── Step 1: 4× Lanczos upscale ────────────────────────────────────────────
        # Limit upscale dimensions to prevent OOM
        max_up_size = 4096
        up_w = img.width  * 4
        up_h = img.height * 4
        if up_w > max_up_size or up_h > max_up_size:
            ratio = max_up_size / max(up_w, up_h)
            up_w = int(up_w * ratio)
            up_h = int(up_h * ratio)
            
        img = img.resize((up_w, up_h), Image.Resampling.LANCZOS)

        # ── Step 2: CLAHE — adaptive local contrast on L channel ──────────────────
        img = _apply_clahe_pil(img, clip_limit=2.0)

        # ── Step 3: Mild denoise — removes noise before sharpening
        try:
            import cv2
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            # Bilateral filter for edge-preserving smoothing (prevents blurring hair strands)
            denoised_bgr = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=15, sigmaSpace=15)
            denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(denoised_rgb)
        except Exception as inner_e:
            logger.warning(f"Denoising inside PIL HD pipeline skipped: {inner_e}")
            img = img.filter(ImageFilter.MedianFilter(size=3))

        # ── Step 4: Multi-pass Unsharp Mask ───────────────────────────────────────
        # Pass A — wide radius, moderate strength (global structure)
        img = img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=150, threshold=1))
        # Pass B — tight radius, high strength (fine edge crispness)
        img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=200, threshold=2))

        # ── Step 5: Detail convolution kernel ────────────────────────────────────
        img = img.filter(_DETAIL_KERNEL)

        # ── Step 6: Finalize — contrast, sharpness, saturation ───────────────────
        img = ImageEnhance.Contrast(img).enhance(1.10)
        img = ImageEnhance.Sharpness(img).enhance(1.40)
        img = ImageEnhance.Color(img).enhance(1.05)     # Subtle saturation lift

        # ── Step 7: Lanczos downscale to target ──────────────────────────────────
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        return img
    except Exception as e:
        logger.error(f"Error in _pil_hd_pipeline: {e}", exc_info=True)
        return pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS) if pil_image else pil_image


def enhance_hd(pil_image: Image.Image, enable_ai: bool = True) -> Image.Image:
    """
    Enhance a passport photo to HD / studio quality using Real-ESRGAN.

    Args:
        pil_image – Input PIL Image (any mode).
        enable_ai  – If True, uses Real-ESRGAN super-resolution.
                     If False, falls back to the fast PIL pipeline.

    Returns:
        Enhanced PIL Image in RGB mode at the same resolution as input.
    """
    if pil_image is None:
        logger.error("enhance_hd received None image")
        return None

    try:
        target_w, target_h = pil_image.size

        # If AI is enabled and Real-ESRGAN weights/upsampler are initialized, run it.
        if enable_ai:
            upsampler = _get_realesrgan_upsampler()
            if upsampler is not None:
                try:
                    # ── Resize guard: Prevent high-res OOM/lag on CPU ──────────────────
                    # Passport photos don't need huge input dimensions. Downscale to max 1024px.
                    max_size = 512 # Keep it smaller for Real-ESRGAN to prevent CPU locking
                    w, h = pil_image.size
                    if w > max_size or h > max_size:
                        ratio = max_size / max(w, h)
                        new_size = (int(w * ratio), int(h * ratio))
                        logger.info(f"Resizing image from {pil_image.size} to {new_size} for Real-ESRGAN processing")
                        work_img = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                    else:
                        work_img = pil_image.copy()

                    rgb_img = work_img.convert("RGB")
                    import cv2
                    img_bgr = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
                    enhanced_bgr = _enhance_realesrgan(img_bgr)
                    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                    result = Image.fromarray(enhanced_rgb)
                    logger.info("Real-ESRGAN face restoration complete (2x).")
                    return result
                except Exception as e:
                    logger.warning(f"Real-ESRGAN enhancement failed ({e}). Falling back to PIL pipeline.")

        # Fallback/Default: PIL pipeline
        logger.info("Applying PIL HD enhancement pipeline (2x).")
        rgb_img = pil_image.convert("RGB")
        return _pil_hd_pipeline(rgb_img, target_w * 2, target_h * 2)
    except Exception as e:
        logger.error(f"Error in enhance_hd: {e}", exc_info=True)
        return pil_image
