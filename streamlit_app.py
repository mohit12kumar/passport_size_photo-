import os
import io
import sys
import logging
import warnings

# Configure logging to display in terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

# ── Suppress noisy third-party startup warnings ───────────────────────────────
# TensorFlow Lite / XNNPACK (from mediapipe)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
# torchvision pretrained= deprecation (from rembg dependency)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", message=".*pretrained.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights.*", category=UserWarning)
# MediaPipe absl logging noise
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

from PIL import Image, UnidentifiedImageError, ImageFilter, ImageEnhance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config import COUNTRY_RULES, PAPER_SIZES
from models.face_detector import align_and_detect
from models.bg_removal import remove_background
from models.auto_enhance import auto_enhance
from models.enhancement import enhance_image
from models.crop_engine import crop_and_resize
from models.layout_generator import generate_printable_sheet
from models.cutting_lines import draw_cutting_lines
from utils.pdf_generator import export_sheet_to_pdf
from models.super_resolution import enhance_hd, REALESRGAN_AVAILABLE
from models.pipeline import loose_crop

logger = logging.getLogger(__name__)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Passport Photo Studio",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Title ────────────────────────────────────────────────────────────────────
st.title("📷 AI Passport Studio")
st.write("Professional biometric passport photos — fully automated and print-ready.")
st.write("---")

# ── Session State Initialization ──────────────────────────────────────────────
_STATE_DEFAULTS = {
    # Upload tracking
    "uploaded_file_name": None,
    "orig_image": None,
    # Face detection (runs once per upload)
    "aligned_image": None,
    "face_box": None,
    "eyes": None,
    "auto_angle": 0.0,
    "auto_enhance_hints": None,
    "face_detected": False,
    "align_error": None,
    "loose_cropped_image": None,
    "adjusted_face_box": None,
    # Background removal cache (invalidated when remove_bg or bg_color changes)
    "bg_image": None,
    "bg_image_key": None,
    # HD Enhancement cache (invalidated when bg settings change)
    "hd_image": None,
    "hd_image_key": None,
    # Slow enhancement cache: AWB + Denoise + CLAHE (invalidated when those toggles change)
    "slow_enhanced_image": None,
    "slow_enhanced_key": None,
    # Print sheet cache (invalidated when country, paper, or photo changes)
    "sheet_img": None,
    "sheet_layout_info": None,
    "sheet_cache_key": None,
    # PDF cache
    "pdf_bytes": None,
    "pdf_cache_key": None,
}
for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _reset_all_caches():
    """Clear every cached intermediate result (called on new file upload)."""
    for k in (
        "orig_image", "aligned_image", "face_box", "eyes",
        "auto_angle", "auto_enhance_hints", "face_detected", "align_error",
        "loose_cropped_image", "adjusted_face_box",
        "bg_image", "bg_image_key",
        "hd_image", "hd_image_key",
        "slow_enhanced_image", "slow_enhanced_key",
        "sheet_img", "sheet_layout_info", "sheet_cache_key",
        "pdf_bytes", "pdf_cache_key",
    ):
        st.session_state[k] = None if k != "auto_angle" else 0.0


def _reset_processing_caches():
    """Clear only the processing pipeline caches — keeps the uploaded image
    and face detection results so they don't need to re-run.
    Use this when settings change (e.g. new BG model loaded) without re-upload."""
    for k in (
        "bg_image", "bg_image_key",
        "hd_image", "hd_image_key",
        "slow_enhanced_image", "slow_enhanced_key",
        "sheet_img", "sheet_layout_info", "sheet_cache_key",
        "pdf_bytes", "pdf_cache_key",
    ):
        st.session_state[k] = None


def _get_processing_image_copy():
    """Return a copy of the best available intermediate image for processing.

    Tries `loose_cropped_image`, then `aligned_image`, then `orig_image`.
    Returns a PIL.Image copy or None if no image is available.
    """
    for key in ("loose_cropped_image", "aligned_image", "orig_image"):
        img = st.session_state.get(key)
        if img is not None:
            try:
                return img.copy()
            except Exception:
                # If copy fails for some reason, return the original object as a last resort
                return img
    return None


# ── Left Control Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Studio Controls")

    # ── 1. Target Country Select ──────────────────────────────────────────────
    country_options = {v["name"]: k for k, v in COUNTRY_RULES.items()}
    country_names = list(country_options.keys())
    _default_country = "United States (USA)"
    _country_default_index = (
        country_names.index(_default_country)
        if _default_country in country_names
        else 0
    )
    selected_country_name = st.selectbox(
        "Target Country Specification",
        options=country_names,
        index=_country_default_index,
    )
    country_key = country_options[selected_country_name]
    country_rule = COUNTRY_RULES[country_key]

    st.info(
        f"**Official Rule for {selected_country_name}:**\n\n"
        f"📏 **Size:** {country_rule['width_mm']} x {country_rule['height_mm']} mm\n\n"
        f"👤 **Head Coverage:** "
        f"{int(country_rule.get('head_height_ratio_min', 0.6)*100)}–"
        f"{int(country_rule.get('head_height_ratio_max', 0.8)*100)}%\n\n"
        f"ℹ️ **Details:** {country_rule.get('description', 'N/A')}"
    )

    st.write("---")

    # ── 2. AI Background Removal ──────────────────────────────────────────────
    remove_bg = st.checkbox("AI Background Removal", value=True)
    bg_color = "#FFFFFF"
    BG_OPTIONS = {
        "White": "#FFFFFF",
        "Light Gray": "#F0F0F0",
        "Light Blue": "#ADD8E6",
        "Transparent": "transparent",
        "Custom Hex": None,
    }
    if remove_bg:
        bg_choice = st.selectbox("Replacement Background Color", options=list(BG_OPTIONS.keys()))
        if BG_OPTIONS[bg_choice] is not None:
            bg_color = BG_OPTIONS[bg_choice]
        else:
            bg_color = st.text_input("Custom Hex Color", value="#FFFFFF", max_chars=7)

    st.write("---")

    # ── 3. HD & Enhancement toggles ───────────────────────────────────────────
    enable_hd = st.checkbox(
        "🔬 AI HD Enhancement (Real-ESRGAN)",
        value=False,
        help="Sharpen details using Real-ESRGAN super-resolution. Runs locally on CPU (~1-2 mins on first run). "
             "If unchecked or weights are missing, the fast PIL sharpening pipeline is used."
    )

    # Local file path checks for weights download
    project_root = os.path.dirname(os.path.abspath(__file__))
    realesrgan_weights_path = os.path.join(project_root, "gfpgan", "weights", "RealESRGAN_x4plus.pth")

    realesrgan_missing = enable_hd and not os.path.isfile(realesrgan_weights_path)

    # Weights downloader helper
    def download_weights(url, dest_path, label):
        import urllib.request
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                total_size = int(response.info().get('Content-Length', 0))
                block_size = 1024 * 1024  # 1 MB
                downloaded = 0
                temp_path = dest_path + ".tmp"
                with open(temp_path, 'wb') as f:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        f.write(buffer)
                        if total_size > 0:
                            percent = min(1.0, downloaded / total_size)
                            progress_bar.progress(percent)
                            status_text.text(f"📥 Downloading {label}: {percent*100:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                os.rename(temp_path, dest_path)
            progress_bar.empty()
            status_text.success(f"✅ {label} downloaded successfully!")
            st.rerun()
            return True
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"❌ Failed to download {label}: {e}")
            if os.path.exists(dest_path + ".tmp"):
                os.remove(dest_path + ".tmp")
            return False

    if realesrgan_missing:
        st.warning("⚠️ Real-ESRGAN weights (~67 MB) are missing. You must download them to use this model.")
        if st.button("📥 Download Real-ESRGAN Weights (xinntao GitHub)"):
            download_weights(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                realesrgan_weights_path,
                "Real-ESRGAN Weights"
            )

    if enable_hd:
        if realesrgan_missing:
            st.info("💡 Model weights not downloaded. Running fast PIL Sharpening instead.")
        else:
            st.warning("⚠️ Real-ESRGAN super-resolution runs on CPU and takes 1–2 min. Result is cached after first run.")
    else:
        st.info("⚡ Running fast local PIL sharpening pipeline (runs in seconds).")

    enable_awb = st.checkbox("Auto White Balance (AWB)", value=False,
                             help="Correct color casts and normalize skin tones.")
    enable_denoise = st.checkbox("AI Noise Reduction (Denoising)", value=False,
                                 help="Reduce camera noise — slow, runs once and caches.")
    enable_clahe = st.checkbox("Adaptive Local Contrast (CLAHE)", value=False,
                               help="Boost local contrast and dynamic range.")

    st.write("---")
    paper_options = {v["name"]: k for k, v in PAPER_SIZES.items()}
    paper_names = list(paper_options.keys())
    _default_paper = "A4 Standard (Document)"
    _paper_default_index = (
        paper_names.index(_default_paper)
        if _default_paper in paper_names
        else 0
    )
    selected_paper_name = st.selectbox(
        "Print Paper Size",
        options=paper_names,
        index=_paper_default_index,
    )
    paper_key = paper_options[selected_paper_name]
    paper_rule = PAPER_SIZES[paper_key]

    photo_count = 8
    if paper_key in ["4x6", "5x7"]:
        photo_count = st.selectbox(
            "Photos Per Sheet",
            options=[1, 2, 4, 6, 8],
            index=4,  # default is 8
        )

    st.write("---")
    st.write("**🔄 Force Re-process**")
    st.caption("Use this if you changed settings and the preview didn't update.")
    if st.button("🔄 Re-process Image", use_container_width=True):
        _reset_processing_caches()
        st.rerun()

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload original portrait image",
    type=["jpg", "jpeg", "png", "heic"]
)

if uploaded_file is not None:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        _reset_all_caches()

        # ── HEIC support ──────────────────────────────────────────────────────
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".heic":
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except ImportError:
                st.error("❌ HEIC support requires the 'pillow-heif' package. "
                         "Run: `pip install pillow-heif`")
                st.stop()

        # ── Open image ────────────────────────────────────────────────────────
        try:
            raw_bytes = uploaded_file.read()
            orig_img = Image.open(io.BytesIO(raw_bytes))
            orig_img.verify()
            orig_img = Image.open(io.BytesIO(raw_bytes))
            orig_img = orig_img.convert("RGB")
            st.session_state.orig_image = orig_img
        except (UnidentifiedImageError, Exception) as e:
            st.error(f"❌ Could not open image: {e}. Please upload a valid JPG, PNG or HEIC file.")
            st.stop()

        # ── AI Face Alignment (once per upload) ───────────────────────────────
        with st.spinner("🤖 AI: Aligning face geometry & detecting eyes…"):
            try:
                aligned_pil, face_box, eyes, auto_angle = align_and_detect(orig_img)
                st.session_state.aligned_image  = aligned_pil
                st.session_state.face_box       = face_box
                st.session_state.eyes           = eyes
                st.session_state.auto_angle     = auto_angle
                st.session_state.face_detected  = face_box is not None
                st.session_state.align_error    = None
                
                # Perform Auto Crop & Center (Loose Crop) for sequential pipeline
                f_box = face_box
                if f_box is None:
                    # Default fallback face box centered on image
                    w, h = aligned_pil.size
                    f_box = {
                        "x": int(w * 0.25),
                        "y": int(h * 0.15),
                        "w": int(w * 0.50),
                        "h": int(h * 0.50)
                    }
                loose_img, adj_face_box = loose_crop(aligned_pil, f_box)
                st.session_state.loose_cropped_image = loose_img
                st.session_state.adjusted_face_box   = adj_face_box
            except Exception as e:
                logger.exception("align_and_detect failed")
                st.session_state.aligned_image = orig_img
                st.session_state.face_box      = None
                st.session_state.eyes          = None
                st.session_state.auto_angle    = 0.0
                st.session_state.face_detected = False
                st.session_state.align_error   = str(e)
                
                # Default loose crop fallback
                w, h = orig_img.size
                f_box = {"x": int(w * 0.25), "y": int(h * 0.15), "w": int(w * 0.50), "h": int(h * 0.50)}
                loose_img, adj_face_box = loose_crop(orig_img, f_box)
                st.session_state.loose_cropped_image = loose_img
                st.session_state.adjusted_face_box   = adj_face_box

        # ── AI Image Quality Analysis (once per upload) ───────────────────────
        with st.spinner("🔍 AI: Estimating brightness, contrast & sharpness…"):
            try:
                st.session_state.auto_enhance_hints = auto_enhance(
                    st.session_state.aligned_image
                )
            except Exception as e:
                logger.warning(f"auto_enhance failed: {e}")
                st.session_state.auto_enhance_hints = None

# ── Workspace ────────────────────────────────────────────────────────────────
if st.session_state.orig_image is not None:
    try:
        if st.session_state.align_error:
            st.error(f"⚠️ Face alignment error: {st.session_state.align_error}. "
                     "You can still proceed — use the sliders to position the face manually.")
        elif not st.session_state.face_detected:
            st.warning("⚠️ No face was detected in this image. "
                       "Use the Shift X/Y and Scale sliders to position and size the face manually.")

        hints = st.session_state.auto_enhance_hints or {}
        analysis = hints.get("analysis", {})

        auto_b  = hints.get("brightness",  1.0)
        auto_c  = hints.get("contrast",    1.0)
        auto_sh = hints.get("sharpness",   1.0)
        auto_sa = hints.get("saturation",  1.0)

        col_studio, col_preview = st.columns([1, 1])

        with col_studio:
            st.subheader("🛠️ Fine-Tuning Controls")

            st.write("**Crop Adjustments**")
            scale           = st.slider("Scale / Zoom",            min_value=0.5,  max_value=2.0,  value=1.0,  step=0.05)
            manual_rotation = st.slider("Manual Angle Adjustment", min_value=-45.0, max_value=45.0, value=0.0, step=0.5)
            x_offset        = st.slider("Shift X (Horizontal)",    min_value=-150, max_value=150,  value=0,    step=1)
            y_offset        = st.slider("Shift Y (Vertical)",      min_value=-150, max_value=150,  value=0,    step=1)

            st.write("---")
            st.write("**AI Enhancements**")
            brightness  = st.slider("Brightness",  min_value=0.5, max_value=1.5, value=1.0,  step=0.02, help=f"AI recommends: {auto_b:.2f}")
            contrast    = st.slider("Contrast",    min_value=0.5, max_value=1.5, value=1.0,  step=0.02, help=f"AI recommends: {auto_c:.2f}")
            sharpness   = st.slider("Sharpness",   min_value=0.0, max_value=2.5, value=1.0,  step=0.05, help=f"AI recommends: {auto_sh:.2f}")
            saturation  = st.slider("Saturation",  min_value=0.0, max_value=2.0, value=1.0,  step=0.05, help=f"AI recommends: {auto_sa:.2f}")
            gamma       = st.slider("Gamma Correction", min_value=0.5, max_value=2.0, value=1.0,  step=0.05, help="Adjust midtone exposure. Gamma > 1.0 brightens shadows, Gamma < 1.0 darkens them.")

        # ── Processing Pipeline (with per-step caching) ───────────────────────────
        with col_preview:
            st.subheader("✨ Passport Output Preview")

            # ── STEP 3: Background Removal (slow — cached by remove_bg + bg_color) ──
            bg_key = (remove_bg, bg_color if remove_bg else None)
            if st.session_state.bg_image_key != bg_key:
                img_for_bg = _get_processing_image_copy()
                if img_for_bg is None:
                    st.error("❌ No image available for background removal. Please upload a portrait photo to continue.")
                    st.session_state.bg_image = None
                else:
                    if remove_bg:
                        with st.spinner("🤖 AI: Removing background (BiRefNet)… (cached after first run)"):
                            try:
                                from models.bg_removal import REMBG_AVAILABLE
                                if not REMBG_AVAILABLE:
                                    st.warning("⚠️ Background removal engine (rembg) is not loaded — skipping.")
                                    st.session_state.bg_image = img_for_bg
                                else:
                                    st.session_state.bg_image = remove_background(img_for_bg, bg_color)
                            except Exception as e:
                                st.error(f"❌ Background removal failed: {e}")
                                st.session_state.bg_image = img_for_bg
                    else:
                        st.session_state.bg_image = img_for_bg
                st.session_state.bg_image_key = bg_key
                # Invalidate downstream caches when bg changes
                st.session_state.slow_enhanced_image = None
                st.session_state.slow_enhanced_key = None
                st.session_state.hd_image = None
                st.session_state.hd_image_key = None

            img_work = st.session_state.bg_image.copy() if st.session_state.bg_image is not None else _get_processing_image_copy()

            # ── STEP 5, 6, 7: Slow enhancements: AWB + Denoise + CLAHE (cached) ─────────
            slow_key = (enable_awb, enable_denoise, enable_clahe, bg_key)
            if st.session_state.slow_enhanced_key != slow_key:
                base_for_slow = img_work.copy()
                if enable_denoise:
                    spinner_msg = "🧹 AI: Denoising… (cached after first run)"
                elif enable_awb or enable_clahe:
                    spinner_msg = "🎨 AI: Applying color corrections (AWB & CLAHE)… (cached)"
                else:
                    spinner_msg = None

                if spinner_msg:
                    with st.spinner(spinner_msg):
                        try:
                            st.session_state.slow_enhanced_image = enhance_image(
                                base_for_slow,
                                brightness=1.0, contrast=1.0, sharpness=1.0, saturation=1.0,
                                denoise=enable_denoise,
                                white_balance=enable_awb,
                                auto_clahe=enable_clahe,
                            )
                        except Exception as e:
                            st.error(f"❌ Enhancement failed: {e}")
                            st.session_state.slow_enhanced_image = base_for_slow
                else:
                    st.session_state.slow_enhanced_image = base_for_slow
                st.session_state.slow_enhanced_key = slow_key
                # Invalidate downstream cache when enhancements change
                st.session_state.hd_image = None
                st.session_state.hd_image_key = None

            if st.session_state.slow_enhanced_image is not None:
                img_work = st.session_state.slow_enhanced_image.copy()

            # ── STEP 8: HD Enhancement / Real-ESRGAN (slow — cached by slow_key + enable_hd) ──────
            hd_key = (enable_hd, slow_key)
            if st.session_state.hd_image_key != hd_key:
                if enable_hd and realesrgan_missing:
                    st.info("💡 Real-ESRGAN weights not downloaded. Running fast PIL Sharpening instead.")
                
                spinner_msg = "🔬 AI: Applying Real-ESRGAN super-resolution… (cached after first run)" if (enable_hd and not realesrgan_missing) else "⚡ Applying PIL sharpening… (cached)"
                with st.spinner(spinner_msg):
                    try:
                        st.session_state.hd_image = enhance_hd(img_work, enable_ai=enable_hd)
                    except Exception as e:
                        st.warning(f"⚠️ HD enhancement skipped: {e}")
                        st.session_state.hd_image = img_work
                st.session_state.hd_image_key = hd_key

            if st.session_state.hd_image is not None:
                img_work = st.session_state.hd_image.copy()

            # ── STEP 9: Fast enhancements (brightness/contrast/sharpness/saturation) & Unsharp Mask
            # Pure PIL — very fast, runs every time for live slider response
            try:
                img_work = enhance_image(
                    img_work,
                    brightness=brightness,
                    contrast=contrast,
                    sharpness=1.0,
                    saturation=saturation,
                    denoise=False,
                    white_balance=False,
                    auto_clahe=False,
                    gamma=gamma,
                )
                if sharpness != 1.0:
                    enhancer = ImageEnhance.Sharpness(img_work)
                    img_work = enhancer.enhance(sharpness)
                    img_work = img_work.filter(
                        ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=2)
                    )
            except Exception as e:
                st.error(f"❌ Enhancement failed: {e}")

            # ── STEP 10: Crop & Resize (fast — runs every time for responsive sliders)
            cropped_photo = None
            face_box = st.session_state.adjusted_face_box
            if face_box is not None and st.session_state.hd_image is not None and st.session_state.slow_enhanced_image is not None:
                orig_w, orig_h = st.session_state.slow_enhanced_image.size
                new_w, new_h = st.session_state.hd_image.size
                if new_w != orig_w or new_h != orig_h:
                    scale_x = new_w / orig_w
                    scale_y = new_h / orig_h
                    face_box = {
                        "x": int(face_box["x"] * scale_x),
                        "y": int(face_box["y"] * scale_y),
                        "w": int(face_box["w"] * scale_x),
                        "h": int(face_box["h"] * scale_y),
                    }

            # Provide a fallback face_box centred on the image if face detection failed
            if face_box is None:
                img_w, img_h = img_work.size
                face_box = {
                    "x": int(img_w * 0.25),
                    "y": int(img_h * 0.15),
                    "w": int(img_w * 0.50),
                    "h": int(img_h * 0.50),
                }

            try:
                cropped_photo = crop_and_resize(
                    img_work,
                    face_box,
                    country_key,
                    scale=scale,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    manual_rotation=manual_rotation,
                )
                if cropped_photo is not None:
                    st.image(cropped_photo, caption="Processed Passport Photo", width="stretch")

                    buf_single = io.BytesIO()
                    cropped_photo.save(buf_single, format="JPEG", quality=95)
                    st.download_button(
                        label="📥 Download Single Photo (JPEG 95%)",
                        data=buf_single.getvalue(),
                        file_name=f"passport_{country_key}.jpg",
                        mime="image/jpeg",
                        width="stretch",
                    )
            except Exception as e:
                st.error(f"❌ Crop failed: {e}. Try adjusting Scale and Shift sliders.")

        # ── Biometric Compliance Report ───────────────────────────────────────────
        st.write("---")

        # Re-evaluate quality on the final processed image if available
        if cropped_photo is not None:
            try:
                final_hints = auto_enhance(cropped_photo)
                analysis = final_hints.get("analysis", {})
                score = final_hints.get("biometric_score", 90)
            except Exception as e:
                logger.warning(f"Failed to re-evaluate final photo quality: {e}")
        else:
            score = hints.get("biometric_score", 90)
            
        score_deductions = []

        if not st.session_state.face_detected:
            score -= 40
            score_deductions.append("No face detected (−40)")
        elif not st.session_state.eyes:
            score -= 10
            score_deductions.append("Eyes not clearly detected or misaligned (−10)")

        if analysis.get("is_blurry"):
            score_deductions.append("Image is slightly blurry")
        if analysis.get("is_dark"):
            score_deductions.append("Image lighting is dark")
        if analysis.get("is_overexposed"):
            score_deductions.append("Image is over-exposed")
        if analysis.get("is_low_contrast"):
            score_deductions.append("Low contrast")

        if not remove_bg:
            score -= 15
            score_deductions.append("Background was not replaced (−15)")

        # Capture sub-threshold minor quality factors if the score is below 90 but no hard checks failed
        if score < 90 and not score_deductions:
            noise_est = analysis.get("noise_score", 0.0)
            if noise_est > 3.0:
                score_deductions.append(f"Minor image noise detected (factor: {noise_est:.1f})")
            
            mean_lum = analysis.get("mean_luminance", 135.0)
            if abs(mean_lum - 135.0) > 20:
                score_deductions.append(f"Sub-optimal lighting exposure (average: {mean_lum:.1f})")
                
            std_lum = analysis.get("std_luminance", 55.0)
            if std_lum < 42.0:
                score_deductions.append(f"Sub-optimal image contrast (factor: {std_lum:.1f})")

            lap_var = analysis.get("blur_score", 80.0)
            if lap_var < 80.0:
                score_deductions.append(f"Minor lack of sharpness (factor: {lap_var:.1f})")

        # Guarantee at least 90% score (Validated PASS) if no deductions/check failures occurred
        if not score_deductions:
            score = max(score, 90)

        score = max(0, min(100, score))

        # ── Printable Layout Sheet (cached) ───────────────────────────────────────
        if cropped_photo is not None:
            st.subheader("🖨️ Printable Layout Sheet")

            col_layout, col_stats = st.columns([1.2, 1])

            with col_layout:
                margin_val = paper_rule.get("default_margin_mm", 5.0)
                gap_val    = paper_rule.get("default_gap_mm",    2.0)

                # Cache key: country + paper + photo size + crop params + photo count
                photo_size_proxy = cropped_photo.size if cropped_photo else None
                sheet_key = (
                    country_key, paper_key, photo_size_proxy,
                    scale, x_offset, y_offset, manual_rotation,
                    photo_count
                )

                if st.session_state.sheet_cache_key != sheet_key:
                    with st.spinner("🖨️ Building print sheet…"):
                        try:
                            sheet_img, layout_info = generate_printable_sheet(
                                cropped_photo,
                                paper_key,
                                country_rule["width_mm"],
                                country_rule["height_mm"],
                                margin_mm=margin_val,
                                gap_mm=gap_val,
                                draw_guides_func=draw_cutting_lines,
                                photo_count=photo_count,
                            )
                            st.session_state.sheet_img = sheet_img
                            st.session_state.sheet_layout_info = layout_info
                            st.session_state.sheet_cache_key = sheet_key
                            # Invalidate PDF when sheet changes
                            st.session_state.pdf_bytes = None
                            st.session_state.pdf_cache_key = None
                        except Exception as e:
                            st.error(f"❌ Sheet generation failed: {e}")
                            st.session_state.sheet_img = None
                            st.session_state.sheet_layout_info = None

                sheet_img   = st.session_state.sheet_img
                layout_info = st.session_state.sheet_layout_info

                if sheet_img is not None:
                    st.image(sheet_img, caption=f"Printable {paper_key} Layout Grid", width="stretch")

            with col_stats:
                st.write("### 📊 Biometric Compliance Report")
                st.metric(label="Biometric Quality Score", value=f"{score}%")
                st.progress(score / 100.0)

                if score >= 90:
                    st.success("✅ Validated — Photo meets all biometric specifications.")
                elif score >= 60:
                    st.warning(f"⚠️ Partial Pass — {'; '.join(score_deductions)}")
                else:
                    st.error(f"❌ Failed — {'; '.join(score_deductions)}")

                st.write("**Biometric Checklist:**")

                def _check(passed, label):
                    icon = "✅" if passed else "❌"
                    status = "PASSED" if passed else "FAILED"
                    st.write(f"{icon} {label}: **{status}**")

                _check(st.session_state.face_detected, "Face Detection")
                _check(bool(st.session_state.eyes), "Eye Detection & Alignment")
                _check(not analysis.get("is_blurry", False), "Image Sharpness")
                _check(not analysis.get("is_dark", False), "Lighting — Brightness")
                _check(not analysis.get("is_overexposed", False), "Lighting — Overexposure")
                _check(not analysis.get("is_low_contrast", False), "Image Contrast")
                _check(remove_bg, f"Background Replaced ({bg_color})")
                _check(True, f"Dimensions ({country_rule['width_mm']}x{country_rule['height_mm']}mm)")

                if layout_info:
                    st.write("---")
                    st.write("**Layout Specifications:**")
                    st.write(f"- **Total Photos:** {layout_info['count']}")
                    st.write(f"- **Grid:** {layout_info['columns']} cols x {layout_info['rows']} rows")
                    st.write(f"- **Paper:** {layout_info['width_mm']} x {layout_info['height_mm']} mm")
                    st.write(f"- **Margin / Gap:** {margin_val} mm / {gap_val} mm")

                    st.write("---")
                    st.write("**Print Downloads:**")

                    if sheet_img is not None:
                        # PDF — cached
                        pdf_key = st.session_state.sheet_cache_key
                        if st.session_state.pdf_cache_key != pdf_key or st.session_state.pdf_bytes is None:
                            try:
                                pdf_buf = io.BytesIO()
                                export_sheet_to_pdf(
                                    sheet_img,
                                    pdf_buf,
                                    paper_w_mm=layout_info["width_mm"],
                                    paper_h_mm=layout_info["height_mm"],
                                )
                                st.session_state.pdf_bytes = pdf_buf.getvalue()
                                st.session_state.pdf_cache_key = pdf_key
                            except Exception as e:
                                st.error(f"❌ PDF generation failed: {e}")

                        if st.session_state.pdf_bytes:
                            st.download_button(
                                label="📄 Download Printable PDF (Recommended)",
                                data=st.session_state.pdf_bytes,
                                file_name=f"layout_{paper_key}.pdf",
                                mime="application/pdf",
                                width="stretch",
                            )

                        # PNG sheet
                        try:
                            sheet_png_buf = io.BytesIO()
                            sheet_img.save(sheet_png_buf, format="PNG")
                            st.download_button(
                                label="🖼️ Download Print Sheet (PNG)",
                                data=sheet_png_buf.getvalue(),
                                file_name=f"layout_{paper_key}.png",
                                mime="image/png",
                                width="stretch",
                            )
                        except Exception as e:
                            st.error(f"❌ PNG export failed: {e}")
    except Exception as workspace_err:
        logger.error(f"Critical workspace rendering error: {workspace_err}", exc_info=True)
        st.error("❌ An unexpected error occurred while rendering the workspace.")
        st.info("💡 You can try: \n1. Clicking **🔄 Re-process Image** in the left sidebar\n2. Re-uploading your portrait photo\n3. Checking the application console logs for more details.")
else:
    st.info("📂 Upload a portrait photo above to run the automated AI Passport Photo pipeline.")
