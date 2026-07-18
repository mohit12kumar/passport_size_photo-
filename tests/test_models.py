import os
import tempfile
from PIL import Image
import pytest

from models.crop_engine import crop_and_resize, calculate_base_crop
from models.layout_generator import generate_printable_sheet
from models.enhancement import enhance_image
from models.cutting_lines import draw_cutting_lines
from models.face_detector import align_and_detect
from models.bg_removal import remove_background
from utils.pdf_generator import export_sheet_to_pdf

def test_calculate_base_crop():
    # Test coordinate calculation logic
    face_box = {"x": 100, "y": 100, "w": 200, "h": 200}
    # For US, dimensions are 50.8 x 50.8 mm (aspect ratio 1.0)
    C_x, C_y, C_w, C_h = calculate_base_crop(face_box, 50.8, 50.8, 0.6)
    assert C_w == pytest.approx(C_h)
    assert C_w > 0
    assert C_x < 100  # Should expand outwards from face box
    assert C_y < 100

def test_crop_and_resize(dummy_image):
    face_box = {"x": 200, "y": 150, "w": 200, "h": 250}

    # Test cropping using the US passport specification (must return a 600x600 px image at 300 DPI)
    cropped = crop_and_resize(dummy_image, face_box, "usa")
    assert cropped is not None
    assert isinstance(cropped, Image.Image)
    assert cropped.size == (600, 600)

    # Test cropping using the Canada passport specification (must return a 590x826 px image)
    cropped_ca = crop_and_resize(dummy_image, face_box, "canada")
    assert cropped_ca is not None
    assert cropped_ca.size == (590, 826)

def test_enhance_image(dummy_image):
    # Test color enhancement pipeline
    enhanced = enhance_image(
        dummy_image,
        brightness=1.2,
        contrast=1.1,
        sharpness=1.5,
        saturation=1.3,
        denoise=False,
        white_balance=False,
        auto_clahe=False,
        gamma=1.0
    )
    assert enhanced is not None
    assert enhanced.size == dummy_image.size

    # Test with OpenCV adjustments enabled
    enhanced_cv = enhance_image(
        dummy_image,
        brightness=1.0,
        contrast=1.0,
        sharpness=1.0,
        saturation=1.0,
        denoise=True,
        white_balance=True,
        auto_clahe=True,
        gamma=1.2
    )
    assert enhanced_cv is not None
    assert enhanced_cv.size == dummy_image.size

def test_generate_printable_sheet(dummy_image):
    # Create a small cropped passport photo (e.g. 600x600 px)
    passport_img = Image.new("RGB", (600, 600), (255, 255, 255))

    # Generate A4 printable sheet (photo width 50.8mm, height 50.8mm)
    sheet_img, layout_info = generate_printable_sheet(
        passport_img,
        paper_size_key="A4",
        photo_w_mm=50.8,
        photo_h_mm=50.8,
        margin_mm=8.0,
        gap_mm=2.5,
        draw_guides_func=draw_cutting_lines
    )

    assert sheet_img is not None
    assert isinstance(sheet_img, Image.Image)
    assert layout_info["count"] > 0
    assert layout_info["columns"] > 0
    assert layout_info["rows"] > 0
    # A4 size in 300 DPI is approx 2480 x 3508 px
    assert sheet_img.width > 2000
    assert sheet_img.height > 2000

def test_export_sheet_to_pdf():
    sheet_img = Image.new("RGB", (2480, 3508), (255, 255, 255))
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "test_output.pdf")
        export_sheet_to_pdf(sheet_img, pdf_path, 210.0, 297.0) # A4 dimensions in mm
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

def test_align_and_detect(real_or_dummy_image):
    # Tests face detection. On dummy image it will hit fallback coordinates.
    # On real image with a face, it should run detection.
    aligned_img, face_box, eyes_data, angle = align_and_detect(real_or_dummy_image)
    assert aligned_img is not None
    assert face_box is not None
    assert "x" in face_box and "y" in face_box
    assert "w" in face_box and "h" in face_box
    assert isinstance(eyes_data, list)
    assert isinstance(angle, (int, float))

def test_remove_background(dummy_image, monkeypatch):
    import models.bg_removal
    # Mock session retrieval to prevent loading the heavy birefnet general model in unit tests
    monkeypatch.setattr(models.bg_removal, "_get_session", lambda: "mock_session")
    monkeypatch.setattr(models.bg_removal, "_get_session_by_name", lambda name: "mock_session")

    if models.bg_removal.REMBG_AVAILABLE:
        import rembg
        monkeypatch.setattr(rembg, "remove", lambda img, **kwargs: img.convert("RGBA"))

    # Test background removal, which will either process or gracefully fallback
    cleaned = remove_background(dummy_image, bg_color_hex="#FFFFFF")
    assert cleaned is not None
    assert cleaned.size == dummy_image.size

    # Test transparency output
    cleaned_trans = remove_background(dummy_image, bg_color_hex="transparent")
    assert cleaned_trans is not None
    assert cleaned_trans.mode in ("RGBA", "RGB")


def test_evaluate_biometric_compliance(dummy_image):
    from models.auto_enhance import evaluate_biometric_compliance
    
    # 1. Test with face_box missing (should fail face_detected)
    res = evaluate_biometric_compliance(dummy_image, face_box=None, eyes=[], auto_angle=0.0, country_code="usa")
    assert res["passed"] is False
    assert res["checks"]["face_detected"]["passed"] is False
    assert res["score"] < 100

    # 2. Test with a mock face_box and eyes, but uncropped image (is_processed=False)
    face_box = {"x": 200, "y": 150, "w": 200, "h": 250}
    eyes = [
        {"x": 250, "y": 230, "w": 20, "h": 20},
        {"x": 330, "y": 230, "w": 20, "h": 20}
    ]
    res = evaluate_biometric_compliance(dummy_image, face_box, eyes, auto_angle=2.0, country_code="usa", is_processed=False)
    assert res["checks"]["face_detected"]["passed"] is True
    assert res["checks"]["eyes_aligned"]["passed"] is True
    assert res["checks"]["tilt"]["status"] == "PASS"

    # 3. Test warning on roll angle (tilt warning)
    res_tilt = evaluate_biometric_compliance(dummy_image, face_box, eyes, auto_angle=12.0, country_code="usa", is_processed=False)
    assert res_tilt["checks"]["tilt"]["status"] == "WARN"

    # 4. Test warning on off-centering
    bad_center_face = {"x": 50, "y": 150, "w": 200, "h": 250} # shifted far left
    res_center = evaluate_biometric_compliance(dummy_image, bad_center_face, eyes, auto_angle=0.0, country_code="usa", is_processed=False)
    assert res_center["checks"]["centering"]["status"] == "WARN"


def test_is_background_compliant():
    from models.bg_removal import is_background_compliant
    # Create a solid white PIL image
    white_img = Image.new("RGB", (300, 300), (255, 255, 255))
    assert is_background_compliant(white_img, "#FFFFFF") is True
    assert is_background_compliant(white_img, "#000000") is False


def test_generate_compliance_pdf(tmp_path):
    from utils.pdf_generator import generate_compliance_pdf
    compliance_data = {
        "passed": True,
        "score": 95,
        "checks": {
            "face_detected": {"status": "PASS", "message": "Face detected"},
            "eyes_aligned": {"status": "PASS", "message": "Eyes aligned"},
            "tilt": {"status": "PASS", "message": "Head tilt is straight"},
        }
    }
    output_pdf = tmp_path / "test_compliance.pdf"
    res_path = generate_compliance_pdf(
        compliance_data=compliance_data,
        output_pdf_path=str(output_pdf),
        country_name="USA",
        doc_type="Passport",
        quality_score=95
    )
    assert os.path.exists(res_path)
    assert output_pdf.stat().st_size > 0


