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

# Initialize MediaPipe Face Mesh
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe Face Mesh is available.")
except ImportError:
    logger.info("MediaPipe not installed — Face Mesh fallback will be unavailable.")

def estimate_head_pose(landmarks, w, h):
    """
    Estimate head pose (yaw, pitch, roll) in degrees using solvePnP.
    landmarks dict must contain: left_eye, right_eye, nose_tip, left_mouth, right_mouth, chin
    """
    try:
        # 3D coordinates of generic face model
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye
            (225.0, 170.0, -135.0),      # Right eye
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)
        
        le = landmarks["left_eye"]
        re = landmarks["right_eye"]
        nt = landmarks["nose_tip"]
        lm = landmarks["left_mouth"]
        rm = landmarks["right_mouth"]
        ch = landmarks["chin"]
        
        image_points = np.array([
            nt, ch, le, re, lm, rm
        ], dtype=np.float32)
        
        # Camera model matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
            
        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        
        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])
        
        return yaw, pitch, roll
    except Exception as e:
        logger.warning(f"solvePnP head pose estimation skipped/failed: {e}")
        try:
            # Simplistic 2D fallback
            le = landmarks["left_eye"]
            re = landmarks["right_eye"]
            nt = landmarks["nose_tip"]
            dx = re[0] - le[0]
            dy = re[1] - le[1]
            roll = math.degrees(math.atan2(dy, dx)) if dx != 0 else 0.0
            eye_w = max(1, re[0] - le[0])
            yaw = ((nt[0] - le[0]) / eye_w - 0.5) * 90.0
            return yaw, 0.0, roll
        except Exception:
            return 0.0, 0.0, 0.0

def detect_face_mediapipe(cv_img):
    """
    Detect face and key landmarks using MediaPipe Face Mesh fallback.
    """
    if not MEDIAPIPE_AVAILABLE:
        return None, None, None
    try:
        h, w = cv_img.shape[:2]
        mp_face_mesh = mp.solutions.face_mesh
        
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb_img)
            
            if not results.multi_face_landmarks:
                return None, None, None
                
            landmarks = results.multi_face_landmarks[0]
            
            # Key landmark indices (refined mesh):
            # Left pupil: 468, Right pupil: 473, Nose tip: 4, Chin: 152, Left mouth corner: 61, Right mouth corner: 291
            pt_left = landmarks.landmark[468]
            pt_right = landmarks.landmark[473]
            pt_nose = landmarks.landmark[4]
            pt_chin = landmarks.landmark[152]
            pt_lm = landmarks.landmark[61]
            pt_rm = landmarks.landmark[291]
            
            le = (int(pt_left.x * w), int(pt_left.y * h))
            re = (int(pt_right.x * w), int(pt_right.y * h))
            nt = (int(pt_nose.x * w), int(pt_nose.y * h))
            ch = (int(pt_chin.x * w), int(pt_chin.y * h))
            lm = (int(pt_lm.x * w), int(pt_lm.y * h))
            rm = (int(pt_rm.x * w), int(pt_rm.y * h))
            
            # Compute face bounding box
            xs = [lmk.x * w for lmk in landmarks.landmark]
            ys = [lmk.y * h for lmk in landmarks.landmark]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            
            face_w = x_max - x_min
            face_h = y_max - y_min
            
            # Adjust face box to center around landmarks with standard proportions
            eye_y = (le[1] + re[1]) / 2.0
            crown_y = eye_y - (ch[1] - eye_y) * 0.95
            
            adjusted_h = int(ch[1] - crown_y)
            adjusted_w = adjusted_h
            adjusted_x = int((le[0] + re[0]) / 2.0 - adjusted_w / 2)
            adjusted_y = int(crown_y)
            
            adjusted_x = max(0, min(w - 10, adjusted_x))
            adjusted_y = max(0, min(h - 10, adjusted_y))
            adjusted_w = max(10, min(w - adjusted_x, adjusted_w))
            adjusted_h = max(10, min(h - adjusted_y, adjusted_h))
            
            face_box = (adjusted_x, adjusted_y, adjusted_w, adjusted_h)
            
            eye_size = int(adjusted_w * 0.12)
            eyes = [
                (le[0] - eye_size // 2, le[1] - eye_size // 2, eye_size, eye_size),
                (re[0] - eye_size // 2, re[1] - eye_size // 2, eye_size, eye_size)
            ]
            
            pts = {
                "left_eye": le,
                "right_eye": re,
                "nose_tip": nt,
                "chin": ch,
                "left_mouth": lm,
                "right_mouth": rm
            }
            
            return face_box, eyes, pts
    except Exception as e:
        logger.warning(f"MediaPipe fallback face detection failed: {e}")
        return None, None, None

def detect_face_and_eyes(cv_img):
    """
    Detect face bounding box, pupil coordinates, and face landmarks.
    Tries RetinaFace first, falls back to MediaPipe.
    Returns:
        face_box (x, y, w, h) or None
        eyes [(x, y, w, h), ...] or None
        landmarks (dict of points) or None
    """
    try:
        if cv_img is None or not hasattr(cv_img, 'shape') or len(cv_img.shape) < 2:
            return None, None, None

        h, w = cv_img.shape[:2]
        if w < 10 or h < 10:
            return None, None, None

        # ── 1. RetinaFace Detection ───────────────────────────────────────
        try:
            detector = get_retinaface_detector()
            if detector is not None:
                import torch
                with torch.no_grad():
                    results = detector.detect_faces(cv_img, conf_threshold=0.5)
                
                if results is not None and len(results) > 0:
                    best_face = results[0]
                    x_min, y_min, x_max, y_max = best_face[0:4]
                    
                    le_x, le_y = int(best_face[5]), int(best_face[6])
                    re_x, re_y = int(best_face[7]), int(best_face[8])
                    nt_x, nt_y = int(best_face[9]), int(best_face[10])
                    lm_x, lm_y = int(best_face[11]), int(best_face[12])
                    rm_x, rm_y = int(best_face[13]), int(best_face[14])
                    
                    # Estimate chin position (approx. 1.2x of eye-to-mouth height from mouth corners)
                    eye_y = (le_y + re_y) / 2.0
                    mouth_y = (lm_y + rm_y) / 2.0
                    chin_y = int(mouth_y + (mouth_y - eye_y) * 0.8)
                    chin_x = int((lm_x + rm_x) / 2.0)
                    
                    # Biometric crown position
                    crown_y = eye_y - (chin_y - eye_y) * 0.95
                    face_h = chin_y - crown_y
                    face_w = face_h
                    
                    center_x = (x_min + x_max) / 2.0
                    fx = center_x - face_w / 2.0
                    fy = crown_y
                    
                    fx_int = max(0, int(fx))
                    fy_int = max(0, int(fy))
                    fw_int = min(w - fx_int, int(face_w))
                    fh_int = min(h - fy_int, int(face_h))
                    
                    if fw_int > 0 and fh_int > 0:
                        face_box = (fx_int, fy_int, fw_int, fh_int)
                        eye_size = max(1, int(fw_int * 0.12))
                        
                        eyes = [
                            (le_x - eye_size // 2, le_y - eye_size // 2, eye_size, eye_size),
                            (re_x - eye_size // 2, re_y - eye_size // 2, eye_size, eye_size)
                        ]
                        
                        landmarks = {
                            "left_eye": (le_x, le_y),
                            "right_eye": (re_x, re_y),
                            "nose_tip": (nt_x, nt_y),
                            "left_mouth": (lm_x, lm_y),
                            "right_mouth": (rm_x, rm_y),
                            "chin": (chin_x, chin_y)
                        }
                        
                        return face_box, sorted(eyes, key=lambda e: e[0]), landmarks
        except Exception as ret_err:
            logger.warning(f"RetinaFace detection error (will retry with MediaPipe): {ret_err}")

        # ── 2. MediaPipe Fallback ──────────────────────────────────────────
        return detect_face_mediapipe(cv_img)
        
    except Exception as e:
        logger.error(f"Error in detect_face_and_eyes: {e}", exc_info=True)
        return None, None, None

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
