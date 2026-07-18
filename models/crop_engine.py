from PIL import Image
import logging
from config import COUNTRY_RULES, mm_to_px

logger = logging.getLogger(__name__)

def calculate_base_crop(face_box, target_w_mm, target_h_mm, head_ratio):
    """
    Calculate the base crop box coordinates on the image based on face box and country rules.
    Returns C_x, C_y, C_w, C_h.
    """
    try:
        if not face_box or not all(k in face_box for k in ("x", "y", "w", "h")):
            raise ValueError(f"Invalid face_box: {face_box}")

        fx, fy, fw, fh = float(face_box["x"]), float(face_box["y"]), float(face_box["w"]), float(face_box["h"])

        # 1. Estimate head dimensions
        h_head = fh * 1.35

        # Crown (top of head) is estimated to be 30% of face height above the face box top.
        y_crown = fy - int(fh * 0.30)

        # Horizontal center of the face
        x_center = fx + fw / 2.0

        # Target aspect ratio
        if target_h_mm <= 0:
            logger.warning(f"Invalid target height: {target_h_mm}, using default aspect ratio 1.0")
            target_ar = 1.0
        else:
            target_ar = target_w_mm / target_h_mm

        # 2. Calculate crop size
        if head_ratio <= 0:
            logger.warning(f"Invalid head ratio: {head_ratio}, using default 0.7")
            head_ratio = 0.7

        C_h = h_head / head_ratio
        C_w = C_h * target_ar

        # 3. Calculate crop position
        C_x = x_center - C_w / 2.0

        # Align top crown with a default top margin (e.g. 12% of crop height)
        top_margin = C_h * 0.12
        C_y = y_crown - top_margin

        return C_x, C_y, C_w, C_h
    except Exception as e:
        logger.error(f"Error in calculate_base_crop: {e}", exc_info=True)
        # Safe fallback: return some arbitrary coordinates based on a 0,0,100,100 box
        return 0, 0, 100, 100


def crop_and_resize(pil_img, face_box, country_code, scale=1.0, x_offset=0, y_offset=0, manual_rotation=0.0):
    """
    Crops the image according to the rules of the selected country and user adjustments,
    and resizes it to the official target dimensions (at 300 DPI).
    """
    if pil_img is None:
        logger.error("crop_and_resize received None pil_img")
        return None

    try:
        # Validate inputs
        if scale <= 0:
            logger.warning(f"Invalid crop scale {scale}, setting to 1.0")
            scale = 1.0

        # Get country rules
        rule = COUNTRY_RULES.get(country_code.lower())
        if not rule:
            logger.warning(f"Country code '{country_code}' not supported. Using standard rules.")
            rule = COUNTRY_RULES.get("united states (usa)")

        target_w_mm = rule["width_mm"]
        target_h_mm = rule["height_mm"]

        # Average of min/max head ratio
        head_ratio = (rule["head_height_ratio_min"] + rule["head_height_ratio_max"]) / 2.0

        # Fallback face box if none exists or is corrupt
        if not face_box or not all(k in face_box for k in ("x", "y", "w", "h")):
            img_w, img_h = pil_img.size
            face_box = {
                "x": int(img_w * 0.25),
                "y": int(img_h * 0.15),
                "w": int(img_w * 0.50),
                "h": int(img_h * 0.50),
            }
            logger.info("Using default fallback face_box for cropping")

        # 1. Rotate the image if manual rotation is specified
        if abs(manual_rotation) > 0.01:
            try:
                # Rotate around the center of the face box
                fx, fy, fw, fh = face_box["x"], face_box["y"], face_box["w"], face_box["h"]
                face_center = (fx + fw // 2, fy + fh // 2)

                # Convert to numpy to rotate using OpenCV (better quality)
                from models.face_detector import pil_to_cv2, cv2_to_pil, rotate_image

                cv_img = pil_to_cv2(pil_img)
                rotated_cv, _ = rotate_image(cv_img, manual_rotation, face_center)
                if rotated_cv is not None:
                    pil_img = cv2_to_pil(rotated_cv)
            except Exception as rot_e:
                logger.error(f"Rotation failed in crop_engine: {rot_e}. Proceeding without rotation.")

        # 2. Calculate base crop dimensions
        C_x, C_y, C_w, C_h = calculate_base_crop(face_box, target_w_mm, target_h_mm, head_ratio)

        # 3. Apply manual adjustments (scale, offsets)
        C_w_adj = C_w / scale
        C_h_adj = C_h / scale

        C_x_adj = C_x + x_offset + (C_w - C_w_adj) / 2.0
        C_y_adj = C_y + y_offset + (C_h - C_h_adj) / 2.0

        # 4. Crop image with boundaries check
        img_w, img_h = pil_img.size

        # Define bounding coordinates
        x1 = int(round(C_x_adj))
        y1 = int(round(C_y_adj))
        x2 = int(round(C_x_adj + C_w_adj))
        y2 = int(round(C_y_adj + C_h_adj))

        # Guarantee dimensions are positive
        if x2 <= x1:
            x2 = x1 + 10
        if y2 <= y1:
            y2 = y1 + 10

        # Padding handling
        bg_hex = rule.get("bg_color_hex", "#FFFFFF")
        from PIL import ImageColor
        try:
            bg_rgb = ImageColor.getcolor(bg_hex, "RGB")
        except Exception:
            bg_rgb = (255, 255, 255)

        try:
            cropped_img = Image.new("RGB", (x2 - x1, y2 - y1), bg_rgb)
        except Exception as img_e:
            logger.error(f"Failed to create new cropped canvas: {img_e}, using fallback canvas size")
            cropped_img = Image.new("RGB", (200, 200), (255, 255, 255))
            x1, y1, x2, y2 = 0, 0, 200, 200

        # Paste source image crop area
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(img_w, x2)
        src_y2 = min(img_h, y2)

        paste_x1 = src_x1 - x1
        paste_y1 = src_y1 - y1

        if src_x2 > src_x1 and src_y2 > src_y1:
            try:
                source_crop = pil_img.crop((src_x1, src_y1, src_x2, src_y2))
                cropped_img.paste(source_crop, (paste_x1, paste_y1))
            except Exception as paste_e:
                logger.error(f"Failed to crop or paste source image area: {paste_e}")

        # 5. Resize to target print size at 300 DPI
        try:
            target_w_px = mm_to_px(target_w_mm)
            target_h_px = mm_to_px(target_h_mm)
        except Exception:
            target_w_px, target_h_px = 600, 600

        if target_w_px <= 0:
            target_w_px = 100
        if target_h_px <= 0:
            target_h_px = 100

        resized_img = cropped_img.resize((target_w_px, target_h_px), Image.Resampling.LANCZOS)
        return resized_img
    except Exception as e:
        logger.error(f"Error in crop_and_resize: {e}", exc_info=True)
        # Return a simple generic fallback resized image
        fallback_img = Image.new("RGB", (600, 600), (255, 255, 255))
        return fallback_img
