import logging
import math
import os
import warnings
from PIL import Image

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", message=".*pretrained.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights.*", category=UserWarning)

# Set OpenCV to run headlessly without showing windows
os.environ["QT_QPA_PLATFORM"] = "offscreen"

logger = logging.getLogger(__name__)

# Initialize RetinaFace
_retinaface_detector = None

def get_retinaface_detector():
    """Lazily initialize the RetinaFace detector instance."""
    global _retinaface_detector
    if _retinaface_detector is None:
        from facexlib.detection import init_detection_model
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _retinaface_detector = init_detection_model('retinaface_resnet50', half=False, device=device)
        _retinaface_detector.eval()
    return _retinaface_detector

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV BGR image."""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert OpenCV BGR image to PIL Image."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def rotate_image(image, angle, center=None):
    """Rotate image by an angle in degrees around a center point."""
    try:
        if image is None or not hasattr(image, 'shape') or len(image.shape) < 2:
            raise ValueError("Invalid image object passed to rotate_image")

        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate bounding box dimensions of the rotated image to avoid clipping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, M
    except Exception as e:
        logger.error(f"Error rotating image: {e}", exc_info=True)
        # Return original image and identity matrix
        h, w = image.shape[:2] if (image is not None and hasattr(image, 'shape')) else (0, 0)
        M = np.eye(2, 3, dtype=np.float32)
        return image, M

def detect_face_and_eyes(cv_img):
    """
    Detects face bounding box and eye pupils using RetinaFace.
    Returns:
        face_box (x, y, w, h) or None
        eyes [(x, y, w, h), ...] or None
    """
    try:
        if cv_img is None or not hasattr(cv_img, 'shape') or len(cv_img.shape) < 2:
            logger.warning("Invalid cv_img passed to detect_face_and_eyes")
            return None, None

        h, w = cv_img.shape[:2]
        if w < 10 or h < 10:
            logger.warning("Image too small to perform face detection")
            return None, None

        detector = get_retinaface_detector()
        if detector is None:
            logger.error("RetinaFace detector is not initialized")
            return None, None

        import torch
        with torch.no_grad():
            results = detector.detect_faces(cv_img, conf_threshold=0.5)

        if results is None or len(results) == 0:
            return None, None

        best_face = results[0]
        x_min, y_min, x_max, y_max = best_face[0:4]

        # Extract eye landmarks:
        # Left eye (relative to person, i.e., image left): index 5, 6
        # Right eye (relative to person, i.e., image right): index 7, 8
        le_x, le_y = best_face[5], best_face[6]
        re_x, re_y = best_face[7], best_face[8]

        # Biometric crown estimation:
        # Average y of eyes
        eye_y = (le_y + re_y) / 2.0
        chin_y = y_max
        # Crown is roughly 0.95 * (chin - eyes) above the eyes
        crown_y = eye_y - (chin_y - eye_y) * 0.95

        face_h = chin_y - crown_y
        face_w = face_h  # Square biometric box

        center_x = (x_min + x_max) / 2.0
        fx = center_x - face_w / 2.0
        fy = crown_y

        # Clip values to image bounds
        fx_int = max(0, int(fx))
        fy_int = max(0, int(fy))
        fw_int = min(w - fx_int, int(face_w))
        fh_int = min(h - fy_int, int(face_h))

        # Avoid zero or negative dimension bounding box
        if fw_int <= 0 or fh_int <= 0:
            logger.warning("Detected face box has non-positive width/height")
            return None, None

        face_box = (fx_int, fy_int, fw_int, fh_int)

        # Create eye boxes centered around the pupils
        eye_size = int(fw_int * 0.12)
        if eye_size <= 0:
            eye_size = 1
        eyes = [
            (int(le_x - eye_size // 2), int(le_y - eye_size // 2), eye_size, eye_size),
            (int(re_x - eye_size // 2), int(re_y - eye_size // 2), eye_size, eye_size)
        ]

        return face_box, sorted(eyes, key=lambda e: e[0])
    except Exception as e:
        logger.error(f"Error in detect_face_and_eyes: {e}", exc_info=True)
        return None, None

def align_and_detect(pil_img):
    """
    Performs face detection, aligns (rotates) the image based on eye coordinates,
    and returns the aligned image and the face bounding box in that aligned image.
    """
    try:
        if pil_img is None:
            raise ValueError("Input PIL image is None")

        cv_img = pil_to_cv2(pil_img)
        if cv_img is None or not hasattr(cv_img, 'shape'):
            raise ValueError("Failed to convert PIL image to CV2")

        orig_h, orig_w = cv_img.shape[:2]

        # 1. Detect face and eyes on the original image
        face_box, eyes = detect_face_and_eyes(cv_img)

        angle = 0.0
        aligned_cv = cv_img.copy()

        if face_box is not None and eyes is not None and len(eyes) >= 2:
            try:
                eye1, eye2 = eyes[0], eyes[1]

                # Calculate eye centers
                p1 = (eye1[0] + eye1[2] / 2.0, eye1[1] + eye1[3] / 2.0)
                p2 = (eye2[0] + eye2[2] / 2.0, eye2[1] + eye2[3] / 2.0)

                # Calculate angle of rotation
                d_x = p2[0] - p1[0]
                d_y = p2[1] - p1[1]

                if abs(d_x) > 0:
                    angle = math.degrees(math.atan2(d_y, d_x))
                    eye_center = (int((p1[0] + p2[0]) / 2.0), int((p1[1] + p2[1]) / 2.0))

                    # Rotate image
                    rotated_cv, M = rotate_image(cv_img, angle, eye_center)
                    if rotated_cv is not None:
                        aligned_cv = rotated_cv

                    # 2. Re-detect face on the aligned image
                    aligned_face_box, aligned_eyes = detect_face_and_eyes(aligned_cv)
                    if aligned_face_box is not None:
                        face_box = aligned_face_box
                        eyes = aligned_eyes
            except Exception as inner_e:
                logger.error(f"Error during alignment calculations: {inner_e}", exc_info=True)

        # Fallback if no face/landmarks detected
        if face_box is None:
            fw = int(orig_w * 0.5)
            fh = int(orig_h * 0.6)
            fx = (orig_w - fw) // 2
            fy = int(orig_h * 0.15)
            face_box = (fx, fy, fw, fh)
            eyes = []

        face_data = {
            "x": int(face_box[0]),
            "y": int(face_box[1]),
            "w": int(face_box[2]),
            "h": int(face_box[3])
        }

        eyes_data = []
        if eyes:
            for eye in eyes:
                eyes_data.append({
                    "x": int(eye[0]),
                    "y": int(eye[1]),
                    "w": int(eye[2]),
                    "h": int(eye[3])
                })

        aligned_pil = cv2_to_pil(aligned_cv)
        return aligned_pil, face_data, eyes_data, angle
    except Exception as e:
        logger.error(f"Error in align_and_detect: {e}", exc_info=True)
        # Absolute fallback return
        orig_w, orig_h = pil_img.size if pil_img else (300, 400)
        face_data = {
            "x": int(orig_w * 0.25),
            "y": int(orig_h * 0.15),
            "w": int(orig_w * 0.5),
            "h": int(orig_h * 0.6)
        }
        return pil_img, face_data, [], 0.0
