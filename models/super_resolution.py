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
import cv2

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── CodeFormer ONNX Face Restoration Class ──────────────────────────────────
class CodeFormerRestorer:
    def __init__(self):
        self.session = None
        self.initialized = False
        
    def _lazy_init(self):
        if self.initialized:
            return
        if not ONNX_AVAILABLE:
            logger.warning("onnxruntime is not installed. CodeFormer cannot initialize.")
            return
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "gfpgan", "weights", "codeformer.onnx")
            if not os.path.isfile(model_path):
                logger.warning(f"CodeFormer ONNX weights not found at: {model_path}")
                return
            
            # Determine execution providers (prefer CUDA, fallback to CPU)
            import torch
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers = [('CUDAExecutionProvider', {"cudnn_conv_algo_search": "DEFAULT"}), 'CPUExecutionProvider']
                
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.initialized = True
            logger.info("CodeFormer ONNX Restorer successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize CodeFormer ONNX: {e}")
            
    def restore_face(self, face_img_rgb, w=0.75):
        """
        Enhance a cropped face image using CodeFormer ONNX.
        """
        self._lazy_init()
        if not self.initialized or self.session is None:
            return face_img_rgb
            
        try:
            # Format inputs: face_img must be converted to RGB numpy array
            img_np = np.array(face_img_rgb.convert("RGB"))
            h_orig, w_orig = img_np.shape[:2]
            
            # Preprocess: resize to 512x512, float32, normalize, transpose to NCHW
            img_resized = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_LINEAR)
            img_float = img_resized.astype(np.float32) / 255.0
            img_norm = (img_float - 0.5) / 0.5
            img_chw = img_norm.transpose((2, 0, 1))
            img_nchw = np.expand_dims(img_chw, axis=0).astype(np.float32)
            
            # Fidelity weight parameter w
            w_input = np.array([w], dtype=np.double)
            
            inputs = self.session.get_inputs()
            input_names = [inp.name for inp in inputs]
            
            # Fit inputs to model nodes
            feed_dict = {}
            if len(input_names) >= 2:
                feed_dict[input_names[0]] = img_nchw
                feed_dict[input_names[1]] = w_input
            else:
                feed_dict[input_names[0]] = img_nchw
                
            outputs = self.session.run(None, feed_dict)
            output_tensor = outputs[0][0]
            
            # Postprocess: transpose back to HWC, denormalize, scale to uint8
            img_out_chw = output_tensor.transpose((1, 2, 0))
            img_out_norm = (img_out_chw.clip(-1, 1) + 1.0) * 0.5
            img_out_uint8 = (img_out_norm * 255.0).clip(0, 255).astype(np.uint8)
            
            # Scale back to cropped face box original dimensions
            img_restored = cv2.resize(img_out_uint8, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            return Image.fromarray(img_restored)
        except Exception as e:
            logger.warning(f"CodeFormer face restoration failed: {e}", exc_info=True)
            return face_img_rgb

_codeformer_restorer = None

def _get_codeformer_restorer():
    global _codeformer_restorer
    if _codeformer_restorer is None:
        _codeformer_restorer = CodeFormerRestorer()
    return _codeformer_restorer

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
        # Pass B — tight radius, high strength (fine edge crisis)
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
    Enhance a passport photo using CodeFormer Face Restoration and Real-ESRGAN.
    
    Guards Real-ESRGAN execution to run only on low-resolution inputs (< 600px)
    to optimize CPU speeds, and uses CodeFormer face enhancement.
    """
    if pil_image is None:
        return None

    try:
        target_w, target_h = pil_image.size

        # ── 1. CodeFormer Face Restoration (Local ONNX) ─────────────────────
        if enable_ai:
            try:
                restorer = _get_codeformer_restorer()
                restorer._lazy_init()
                if restorer.initialized:
                    from models.face_detector import detect_face_and_eyes, pil_to_cv2
                    cv_img = pil_to_cv2(pil_image)
                    face_box, eyes, landmarks = detect_face_and_eyes(cv_img)
                    
                    if face_box:
                        fx, fy, fw, fh = face_box
                        
                        # Add biometric padding around the face box (ears, hair, chin)
                        pad_x = int(fw * 0.25)
                        pad_y = int(fh * 0.35)
                        
                        x1 = max(0, fx - pad_x)
                        y1 = max(0, fy - pad_y)
                        x2 = min(pil_image.width, fx + fw + pad_x)
                        y2 = min(pil_image.height, fy + fh + pad_y)
                        
                        face_crop = pil_image.crop((x1, y1, x2, y2))
                        restored_face = restorer.restore_face(face_crop, w=0.75)
                        
                        # Circular feathered blending mask to paste back restored details
                        mask = Image.new("L", face_crop.size, 0)
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(mask)
                        draw.ellipse([pad_x, pad_y, face_crop.width - pad_x, face_crop.height - pad_y], fill=255)
                        mask = mask.filter(ImageFilter.GaussianBlur(radius=12))
                        
                        pil_image = pil_image.copy()
                        pil_image.paste(restored_face, (x1, y1), mask)
                        logger.info("CodeFormer face restoration successfully applied and blended.")
            except Exception as cf_err:
                logger.warning(f"CodeFormer face restoration skipped/failed: {cf_err}")

        # ── 2. Real-ESRGAN Super-Resolution (Guarded) ─────────────────────────
        # Only run Real-ESRGAN if enable_ai is True AND the photo is low-resolution (< 600px width/height)
        is_low_res = (target_w < 600 or target_h < 600)
        
        if enable_ai and is_low_res:
            upsampler = _get_realesrgan_upsampler()
            if upsampler is not None:
                try:
                    logger.info("Input resolution is low; executing Real-ESRGAN super-resolution.")
                    rgb_img = pil_image.convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
                    enhanced_bgr = _enhance_realesrgan(img_bgr)
                    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(enhanced_rgb)
                except Exception as e:
                    logger.warning(f"Real-ESRGAN failed ({e}). Falling back to fast HD pipeline.")

        # Fallback/Default: fast PIL HD pipeline
        logger.info("Applying fast PIL HD sharpening pipeline.")
        rgb_img = pil_image.convert("RGB")
        return _pil_hd_pipeline(rgb_img, target_w * 2, target_h * 2)
    except Exception as e:
        logger.error(f"Error in enhance_hd: {e}", exc_info=True)
        return pil_image
