import os
import json

# Load environment variables from .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload and output folders (configurable via environment variables)
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", os.path.join(BASE_DIR, "outputs"))
PASSPORT_OUTPUT_FOLDER = os.environ.get("PASSPORT_OUTPUT_FOLDER", os.path.join(OUTPUT_FOLDER, "passport"))
PRINTABLE_OUTPUT_FOLDER = os.environ.get("PRINTABLE_OUTPUT_FOLDER", os.path.join(OUTPUT_FOLDER, "printable"))

# Ensure directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, PASSPORT_OUTPUT_FOLDER, PRINTABLE_OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Standard print DPI (Dots Per Inch) (configurable via environment variables)
DPI = int(os.environ.get("DPI", 300))

# Pre-fixed printable layout defaults
DEFAULT_MARGIN_MM = float(os.environ.get("DEFAULT_MARGIN_MM", 8.0))
DEFAULT_GAP_MM = float(os.environ.get("DEFAULT_GAP_MM", 2.5))

# rembg background removal model configuration
REMBG_MODEL = os.environ.get("REMBG_MODEL", "birefnet-general-lite")

# Conversion helper: mm to pixels at 300 DPI
def mm_to_px(mm, dpi=DPI):
    return int((mm / 25.4) * dpi)

# Conversion helper: inches to pixels
def inch_to_px(inches, dpi=DPI):
    return int(inches * dpi)

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Country Rules Database Loader
# ─────────────────────────────────────────────────────────────────────────────
COUNTRY_RULES = {}
try:
    json_path = os.path.join(BASE_DIR, "models", "country_rules.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            db = json.load(f)
        for c_key, data in db.items():
            # Gather aliases
            aliases = [c_key, data["code"]]
            if data["code"] == "us":
                aliases.append("usa")
            elif data["code"] == "gb":
                aliases.append("uk")
            elif data["code"] == "ca":
                aliases.append("can")

            # Determine displays names
            display_name = data["name"]
            if data["code"] == "us":
                display_name = "United States (USA)"
            elif data["code"] == "gb":
                display_name = "United Kingdom (UK)"
            elif data["code"] == "de":
                display_name = "Germany (Biometric)"

            for alias in aliases:
                passport_profile = data["passport"].copy()
                passport_profile["name"] = display_name
                COUNTRY_RULES[alias] = passport_profile
                
                # Document-type profiles
                for doc_type in ["passport", "visa", "id_card", "residence_permit", "driving_license"]:
                    doc_key = f"{alias}_{doc_type}"
                    doc_profile = data[doc_type].copy()
                    doc_profile["name"] = f"{display_name} - {doc_type.replace('_', ' ').title()}"
                    COUNTRY_RULES[doc_key] = doc_profile
    else:
        # Fallback profile if database is missing
        COUNTRY_RULES["usa"] = {
            "name": "United States (USA)",
            "width_mm": 50.8,
            "height_mm": 50.8,
            "dpi": 300,
            "pixel_width": 600,
            "pixel_height": 600,
            "head_height_ratio_min": 0.50,
            "head_height_ratio_max": 0.69,
            "eye_height_ratio_min": 0.56,
            "eye_height_ratio_max": 0.69,
            "top_margin_mm": 5.0,
            "bg_color": "White / Off-White",
            "bg_color_hex": "#FFFFFF",
            "expression": "Neutral",
            "glasses": "Not Allowed",
            "head_cover": "Allowed",
            "shadows": "Not Allowed",
            "smile": "Neutral",
            "rotation_max_deg": 5,
            "description": "Fallback US Profile"
        }
except Exception as e:
    import logging
    logging.getLogger(__name__).error(f"Failed to dynamically load country_rules database: {e}")

# Standard Paper Sizes
# Dimensions in millimeters (mm)
PAPER_SIZES = {
    "4x6": {
        "name": "4 x 6 inches (Photo Paper)",
        "width_mm": 102.0,
        "height_mm": 152.0,
        "default_margin_mm": 3.0,
        "default_gap_mm": 2.0
    },
    "5x7": {
        "name": "5 x 7 inches (Photo Paper)",
        "width_mm": 127.0,
        "height_mm": 178.0,
        "default_margin_mm": 4.0,
        "default_gap_mm": 2.0
    },
    "A4": {
        "name": "A4 Standard (Document)",
        "width_mm": 210.0,
        "height_mm": 297.0,
        "default_margin_mm": 5.0,
        "default_gap_mm": 2.0
    },
    "Letter": {
        "name": "Letter Standard",
        "width_mm": 215.9,
        "height_mm": 279.4,
        "default_margin_mm": 5.0,
        "default_gap_mm": 2.0
    },
    "Legal": {
        "name": "Legal Standard",
        "width_mm": 215.9,
        "height_mm": 355.6,
        "default_margin_mm": 5.0,
        "default_gap_mm": 2.0
    }
}
