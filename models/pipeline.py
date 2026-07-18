import logging
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)

def loose_crop(img: Image.Image, face_box: dict) -> tuple:
    """
    Step 2: Auto Crop & Center (Loose Crop)
    Crops a loose box centered around the face and including the neck and shoulders.
    This reduces the size of the image for background removal and upscaling,
    making the process faster and less memory-intensive.

    Returns:
        (cropped_img, adjusted_face_box)
    """
    try:
        w, h = img.size
        fx, fy, fw, fh = float(face_box["x"]), float(face_box["y"]), float(face_box["w"]), float(face_box["h"])

        # Calculate loose crop boundaries relative to face dimensions
        # We extend the box: 0.8 * fw on left/right, 0.9 * fh on top, and 2.5 * fh on bottom
        x1 = int(fx - fw * 0.8)
        y1 = int(fy - fh * 0.9)
        x2 = int(fx + fw * 1.8)
        y2 = int(fy + fh * 2.5)

        # Pad if the crop box falls outside original image boundaries
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        # Coordinates clamped to image boundaries
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x2)
        src_y2 = min(h, y2)

        # Crop the valid region
        crop_area = img.crop((src_x1, src_y1, src_x2, src_y2))

        # Create a new padded canvas (using pure white background default)
        target_w = (src_x2 - src_x1) + pad_left + pad_right
        target_h = (src_y2 - src_y1) + pad_top + pad_bottom
        padded_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        padded_img.paste(crop_area, (pad_left, pad_top))

        # Adjust face box coordinates relative to the new cropped image
        adjusted_face_box = {
            "x": int(fx - x1),
            "y": int(fy - y1),
            "w": int(fw),
            "h": int(fh)
        }
        return padded_img, adjusted_face_box
    except Exception as e:
        logger.error(f"Error in loose_crop: {e}", exc_info=True)
        # Return fallback original image and coordinates
        return img, face_box


def run_passport_pipeline(
    pil_image: Image.Image,
    country_code: str,
    remove_bg: bool = True,
    bg_color_hex: str = "#FFFFFF",
    enable_hd: bool = False,
    white_balance: bool = False,
    denoise: bool = False,
    auto_clahe: bool = False,
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpness: float = 1.0,
    saturation: float = 1.0,
    scale: float = 1.0,
    x_offset: int = 0,
    y_offset: int = 0,
    manual_rotation: float = 0.0,
    face_box_override: dict = None,
    gamma: float = 1.0
) -> Image.Image:
    """
    Executes the 10-step structured pipeline sequentially:
    1. Face Detection & Leveling (Done outside or first step)
    2. Auto Crop & Center (Loose Crop)
    3. BiRefNet (Background Removal)
    4. Edge Refinement (Hair Mask Cleanup)
    5. OpenCV Enhancement / White Balance / CLAHE (Color corrections)
    6. Real-ESRGAN x2 (or PIL HD upscaler)
    7. Unsharp Mask (Sharpening)
    8. Passport Size Crop & Resize (Final scale & 300 DPI)
    """
    from models.face_detector import align_and_detect
    from models.bg_removal import remove_background
    from models.enhancement import enhance_image
    from models.super_resolution import enhance_hd
    from models.crop_engine import crop_and_resize

    # ── Step 1: Face Detection & Leveling ──────────────────────────
    aligned_img, face_box, eyes, auto_angle = align_and_detect(pil_image)
    if face_box_override:
        face_box = face_box_override

    # ── Step 2: Auto Crop & Center (Loose Crop) ────────────────────
    cropped_loose, adjusted_face_box = loose_crop(aligned_img, face_box)
    img_work = cropped_loose

    # ── Step 3: BiRefNet Background Removal ────────────────────────
    if remove_bg:
        img_work = remove_background(img_work, bg_color_hex)
        # Note: Step 4 (Edge Refinement) is performed internally within remove_background
        # via the _clean_alpha_mask function (feathering, thresholding, alpha matting).

    # ── Step 5, 6, 7: OpenCV Enhancement, White Balance, CLAHE ────
    # These operations are coordinated via the enhance_image module
    img_work = enhance_image(
        img_work,
        brightness=brightness,
        contrast=contrast,
        sharpness=1.0,  # Apply sharpening at Step 9 instead
        saturation=saturation,
        denoise=denoise,
        white_balance=white_balance,
        auto_clahe=auto_clahe,
        gamma=gamma
    )

    # ── Step 8: Real-ESRGAN x2 / HD Enhancement ───────────────────
    orig_w, orig_h = img_work.size
    img_work = enhance_hd(img_work, enable_ai=enable_hd)
    new_w, new_h = img_work.size
    if new_w != orig_w or new_h != orig_h:
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        adjusted_face_box = {
            "x": int(adjusted_face_box["x"] * scale_x),
            "y": int(adjusted_face_box["y"] * scale_y),
            "w": int(adjusted_face_box["w"] * scale_x),
            "h": int(adjusted_face_box["h"] * scale_y)
        }

    # ── Step 9: Unsharp Mask ───────────────────────────────────────
    if sharpness != 1.0:
        try:
            # Multi-pass sharpening
            enhancer = ImageEnhance.Sharpness(img_work)
            img_work = enhancer.enhance(sharpness)

            # Apply unsharp mask filter for crisp edges
            img_work = img_work.filter(
                ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=2)
            )
        except Exception as e:
            logger.warning(f"Unsharp Mask failed: {e}")

    # ── Step 10: Passport Size Crop ────────────────────────────────
    final_passport = crop_and_resize(
        img_work,
        adjusted_face_box,
        country_code,
        scale=scale,
        x_offset=x_offset,
        y_offset=y_offset,
        manual_rotation=manual_rotation
    )

    return final_passport
