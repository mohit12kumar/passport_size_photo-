from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import uuid
import os
import shutil
from PIL import Image

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
from models.bg_removal import remove_background
from models.auto_enhance import auto_enhance as compute_auto_enhance
from models.enhancement import enhance_image
from models.crop_engine import crop_and_resize
from models.layout_generator import generate_printable_sheet
from models.cutting_lines import draw_cutting_lines
from utils.pdf_generator import export_sheet_to_pdf
from models.pipeline import run_passport_pipeline

app = FastAPI(
    title="AI-Powered Passport Photo Generator API",
    description="Backend services for generating professional passport-size photos."
)

# Ensure directories exist (double check)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PASSPORT_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PRINTABLE_OUTPUT_FOLDER, exist_ok=True)

# Mount static and output folders to serve files directly to frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

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

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a photo, performs automatic face detection and rotation/alignment,
    and returns face metadata and alignment angle.
    """
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
            "auto_enhance": enhance_hints
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
            hints = compute_auto_enhance(final_img)
            biometric_score = hints.get("biometric_score", 90)
            analysis = hints.get("analysis", {})
        except Exception:
            biometric_score = 90
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
            "analysis": analysis
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
