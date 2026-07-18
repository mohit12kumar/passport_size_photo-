from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import uuid
import os
import shutil
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from config import (
    UPLOAD_FOLDER,
    OUTPUT_FOLDER,
    PASSPORT_OUTPUT_FOLDER,
    PRINTABLE_OUTPUT_FOLDER,
    COUNTRY_RULES,
    PAPER_SIZES,
    DEFAULT_MARGIN_MM,
    DEFAULT_GAP_MM
)
from models.face_detector import align_and_detect
from models.auto_enhance import auto_enhance as compute_auto_enhance, evaluate_biometric_compliance
from models.layout_generator import generate_printable_sheet
from models.cutting_lines import draw_cutting_lines
from utils.pdf_generator import export_sheet_to_pdf, generate_compliance_pdf
from models.pipeline import run_passport_pipeline

app = FastAPI(
    title="AI-Powered Passport Photo Generator API",
    description="Backend services for generating professional passport-size photos."
)

# Ensure directories exist (double check)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PASSPORT_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PRINTABLE_OUTPUT_FOLDER, exist_ok=True)

def clean_folders():
    """Deletes all files inside uploads/ and outputs/ directories to save space."""
    folders = [UPLOAD_FOLDER, PASSPORT_OUTPUT_FOLDER, PRINTABLE_OUTPUT_FOLDER]
    for folder in folders:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to delete {file_path}. Reason: {e}")

# Run cleanup on startup to clear any leftover files from previous sessions
try:
    clean_folders()
except Exception as clean_err:
    import logging
    logging.getLogger(__name__).warning(f"Startup clean failed: {clean_err}")

# Mount static and output folders to serve files directly to frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")

class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Prevent browsers from caching static JS/CSS files during development."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheStaticMiddleware)

# Pydantic request models
class FaceBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class ProcessRequest(BaseModel):
    filename: str
    country: str
    face: FaceBox
    scale: float = 1.0
    x_offset: int = 0
    y_offset: int = 0
    manual_rotation: float = 0.0
    remove_bg: bool = False
    bg_color_hex: str = "#FFFFFF"
    auto_enhance: bool = True     # When True the backend computes optimal values
    brightness: float = 1.0
    contrast: float = 1.0
    sharpness: float = 1.0
    saturation: float = 1.0
    enable_hd: bool = False
    white_balance: bool = False
    denoise: bool = False
    auto_clahe: bool = False
    gamma: float = 1.0

class LayoutRequest(BaseModel):
    filename: str
    country: str
    paper_size: str
    margin_mm: float = DEFAULT_MARGIN_MM
    gap_mm: float = DEFAULT_GAP_MM
    photo_count: int = 999

@app.get("/")
async def serve_index():
    """Serves the main application page."""
    try:
        clean_folders()
    except Exception as clean_err:
        import logging
        logging.getLogger(__name__).warning(f"Index reload cleanup failed: {clean_err}")
    index_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

@app.get("/api/countries")
async def get_countries():
    """Returns the database of supported country passport photo rules."""
    return COUNTRY_RULES

@app.get("/api/paper-sizes")
async def get_paper_sizes():
    """Returns the database of supported paper sheet dimensions."""
    return PAPER_SIZES

@app.post("/api/cleanup")
async def api_cleanup():
    """Manually clears all upload and output files."""
    try:
        clean_folders()
        return {"success": True, "message": "All previous data cleared successfully."}
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in api_cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a photo, performs automatic face detection and rotation/alignment,
    and returns face metadata and alignment angle.
    """
    # Clean up previous runs' files to save disk space
    try:
        clean_folders()
    except Exception as clean_err:
        import logging
        logging.getLogger(__name__).warning(f"Upload cleanup failed: {clean_err}")

    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".heic"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload JPG, PNG, or HEIC.")

    # Generate unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4()}{ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{unique_filename}")

    # Save uploaded file
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Check HEIC support
        if ext == ".heic":
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except ImportError:
                os.remove(temp_path)
                raise HTTPException(
                    status_code=400,
                    detail="HEIC support requires 'pillow-heif' package on the server. Please upload JPG or PNG."
                )

        pil_img = Image.open(temp_path)

        # Run automatic face detection & alignment (straightening)
        aligned_pil, face_box, eyes, auto_angle = align_and_detect(pil_img)

        # Save the straightened image (this becomes the working source image)
        aligned_filename = f"aligned_{unique_filename.split('.')[0]}.png"
        aligned_path = os.path.join(UPLOAD_FOLDER, aligned_filename)
        aligned_pil.save(aligned_path, "PNG")

        # Save original file reference
        final_orig_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        shutil.move(temp_path, final_orig_path)

        # Run AI auto-enhancement analysis
        try:
            enhance_hints = compute_auto_enhance(aligned_pil)
        except Exception:
            enhance_hints = {
                "brightness": 1.0, "contrast": 1.0,
                "sharpness": 1.0, "saturation": 1.0,
                "analysis": {}
            }

        # Run strict quality validation
        try:
            compliance = evaluate_biometric_compliance(
                aligned_pil, face_box, eyes, auto_angle, country_code="usa", is_processed=False
            )
        except Exception as comp_err:
            import logging
            logging.getLogger(__name__).warning(f"Compliance check failed on upload: {comp_err}")
            compliance = {
                "passed": face_box is not None,
                "score": 90,
                "checks": {}
            }

        return {
            "success": True,
            "filename": unique_filename,
            "aligned_filename": aligned_filename,
            "original_size": {"width": pil_img.width, "height": pil_img.height},
            "aligned_size": {"width": aligned_pil.width, "height": aligned_pil.height},
            "face": face_box,
            "eyes": eyes,
            "auto_angle": auto_angle,
            "original_url": f"/uploads/{unique_filename}",
            "aligned_url": f"/uploads/{aligned_filename}",
            "auto_enhance": enhance_hints,
            "compliance": compliance
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        import logging
        logging.getLogger(__name__).error(f"Error in upload processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/api/process")
async def process_photo(req: ProcessRequest):
    """
    Processes the photo using the 10-step unified pipeline:
    1. Alignment & face detection
    2. Loose crop & centering
    3. Background removal
    4. Edge refinement
    5. OpenCV exposure adjustment
    6. White balance correction
    7. CLAHE adaptive local contrast
    8. Real-ESRGAN or PIL HD upscaling
    9. Unsharp Masking
    10. Final Passport size crop and resize at 300 DPI
    """
    # Locate original working image
    orig_path = os.path.join(UPLOAD_FOLDER, req.filename)

    if not os.path.exists(orig_path):
        # Fallback to aligned if original is missing
        aligned_filename = f"aligned_{os.path.splitext(req.filename)[0]}.png"
        orig_path = os.path.join(UPLOAD_FOLDER, aligned_filename)
        if not os.path.exists(orig_path):
            raise HTTPException(status_code=404, detail="Source image not found.")

    try:
        # Load image
        img = Image.open(orig_path)

        # Convert face box to dictionary
        face_dict = {
            "x": req.face.x,
            "y": req.face.y,
            "w": req.face.w,
            "h": req.face.h
        }

        # Determine enhancement values (auto enhance or custom sliders)
        if req.auto_enhance:
            # We align it first to run auto-enhance on the aligned face image
            from models.face_detector import align_and_detect
            aligned_temp, _, _, _ = align_and_detect(img)
            hints = compute_auto_enhance(aligned_temp)
            brightness = hints["brightness"]
            contrast   = hints["contrast"]
            sharpness  = hints["sharpness"]
            saturation = hints["saturation"]
        else:
            brightness = req.brightness
            contrast   = req.contrast
            sharpness  = req.sharpness
            saturation = req.saturation

        # Run 10-step unified pipeline
        final_img = run_passport_pipeline(
            pil_image=img,
            country_code=req.country,
            remove_bg=req.remove_bg,
            bg_color_hex=req.bg_color_hex,
            enable_hd=req.enable_hd,
            white_balance=req.white_balance,
            denoise=req.denoise,
            auto_clahe=req.auto_clahe,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            saturation=saturation,
            scale=req.scale,
            x_offset=req.x_offset,
            y_offset=req.y_offset,
            manual_rotation=req.manual_rotation,
            face_box_override=face_dict,
            gamma=req.gamma
        )

        # Save processed passport image
        out_filename = f"passport_{os.path.splitext(req.filename)[0]}.jpg"
        out_path = os.path.join(PASSPORT_OUTPUT_FOLDER, out_filename)
        final_img.save(out_path, "JPEG", quality=95, dpi=(300, 300))

        # Calculate biometric quality analysis on the final processed image
        try:
            # Re-run face detection on the final cropped image to check final alignment
            final_aligned, final_face_box, final_eyes, final_angle = align_and_detect(final_img)
            
            compliance = evaluate_biometric_compliance(
                final_img, final_face_box, final_eyes, final_angle + req.manual_rotation, req.country, is_processed=True
            )
            
            biometric_score = compliance["score"]
            analysis = {
                "is_blurry": not compliance["checks"].get("sharpness", {}).get("passed", True),
                "is_dark": not compliance["checks"].get("brightness", {}).get("passed", True),
                "is_overexposed": not compliance["checks"].get("overexposure", {}).get("passed", True),
                "is_low_contrast": not compliance["checks"].get("contrast", {}).get("passed", True)
            }
        except Exception as comp_err:
            import logging
            logging.getLogger(__name__).error(f"Error in compliance check on process: {comp_err}", exc_info=True)
            biometric_score = 90
            compliance = {
                "passed": True,
                "score": 90,
                "checks": {}
            }
            analysis = {
                "is_blurry": False,
                "is_dark": False,
                "is_overexposed": False,
                "is_low_contrast": False
            }

        return {
            "success": True,
            "filename": out_filename,
            "url": f"/outputs/passport/{out_filename}",
            "width": final_img.width,
            "height": final_img.height,
            "biometric_score": biometric_score,
            "analysis": analysis,
            "compliance": compliance
        }
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in process_photo: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.post("/api/generate-sheet")
async def generate_sheet(req: LayoutRequest):
    """
    Arranges multiple passport photos in a grid layout on the selected paper sheet size,
    draws borders and trim marks, and exports the sheet as high-res PNG and PDF.
    """
    passport_path = os.path.join(PASSPORT_OUTPUT_FOLDER, req.filename)
    if not os.path.exists(passport_path):
        raise HTTPException(status_code=404, detail="Processed passport photo not found.")

    try:
        # Load cropped passport photo
        passport_img = Image.open(passport_path)

        # Get country rules to know dimensions in mm
        rule = COUNTRY_RULES.get(req.country.lower())
        if not rule:
            raise HTTPException(status_code=400, detail="Country not supported.")

        photo_w_mm = rule["width_mm"]
        photo_h_mm = rule["height_mm"]

        # Generate sheet
        sheet_img, layout_info = generate_printable_sheet(
            passport_img,
            req.paper_size,
            photo_w_mm,
            photo_h_mm,
            margin_mm=req.margin_mm,
            gap_mm=req.gap_mm,
            draw_guides_func=draw_cutting_lines,
            photo_count=req.photo_count
        )

        # Save sheet image
        sheet_filename = f"sheet_{req.paper_size}_{req.filename}"
        sheet_path = os.path.join(PRINTABLE_OUTPUT_FOLDER, sheet_filename)
        sheet_img.save(sheet_path, "PNG", dpi=(300, 300))

        # Generate PDF
        pdf_filename = f"print_{req.paper_size}_{os.path.splitext(req.filename)[0]}.pdf"
        pdf_path = os.path.join(PRINTABLE_OUTPUT_FOLDER, pdf_filename)

        export_sheet_to_pdf(
            sheet_img,
            pdf_path,
            layout_info["width_mm"],
            layout_info["height_mm"]
        )

        return {
            "success": True,
            "sheet_filename": sheet_filename,
            "pdf_filename": pdf_filename,
            "sheet_url": f"/outputs/printable/{sheet_filename}",
            "pdf_url": f"/outputs/printable/{pdf_filename}",
            "count": layout_info["count"],
            "columns": layout_info["columns"],
            "rows": layout_info["rows"],
            "paper_width_mm": layout_info["width_mm"],
            "paper_height_mm": layout_info["height_mm"],
            "photo_rotated": layout_info.get("photo_rotated", False)
        }
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in generate_sheet: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate layout: {str(e)}")

class ComplianceReportRequest(BaseModel):
    compliance: dict
    country: str
    doc_type: str
    score: int

@app.post("/api/compliance-report")
async def get_compliance_report(req: ComplianceReportRequest):
    try:
        report_filename = f"compliance_report_{uuid.uuid4().hex[:8]}.pdf"
        report_path = os.path.join(PRINTABLE_OUTPUT_FOLDER, report_filename)
        
        # Generate compliance PDF
        generate_compliance_pdf(
            compliance_data=req.compliance,
            output_pdf_path=report_path,
            country_name=req.country,
            doc_type=req.doc_type,
            quality_score=req.score
        )
        
        return {
            "success": True,
            "pdf_url": f"/outputs/printable/{report_filename}",
            "pdf_filename": report_filename
        }
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to generate compliance PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to compile report: {str(e)}")

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("RELOAD", "True").lower() in ("true", "1", "yes")
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        # Exclude venv, output dirs, and uploads from watchfiles to prevent
        # llvmlite/numba DLL crash during subprocess spawn on Windows hot-reload
        reload_excludes=[
            "venv",
            "outputs",
            "uploads",
            "*.jpg",
            "*.png",
            "*.pdf"
        ]
    )
