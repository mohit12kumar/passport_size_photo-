import io
import os
import pytest
from PIL import Image

# Import configurations to know where files are written
from config import UPLOAD_FOLDER, PASSPORT_OUTPUT_FOLDER, PRINTABLE_OUTPUT_FOLDER

def test_get_countries(client):
    response = client.get("/api/countries")
    assert response.status_code == 200
    data = response.json()
    assert "usa" in data
    assert data["usa"]["name"] == "United States (USA)"
    assert "width_mm" in data["usa"]
    assert "height_mm" in data["usa"]

def test_get_paper_sizes(client):
    response = client.get("/api/paper-sizes")
    assert response.status_code == 200
    data = response.json()
    assert "A4" in data
    assert "Letter" in data
    assert data["A4"]["width_mm"] == 210.0
    assert data["A4"]["height_mm"] == 297.0

def test_api_full_pipeline_flow(client):
    # 1. Create a dummy test image in bytes
    img = Image.new("RGB", (600, 600), color=(240, 240, 240))
    # We don't draw anything complex, just standard RGB
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # Files created list for cleanup
    created_files = []
    
    try:
        # 2. Upload file
        upload_response = client.post(
            "/api/upload",
            files={"file": ("test_upload_image.jpg", img_byte_arr, "image/jpeg")}
        )
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        
        assert upload_data["success"] is True
        assert "filename" in upload_data
        assert "aligned_filename" in upload_data
        assert "face" in upload_data
        
        filename = upload_data["filename"]
        aligned_filename = upload_data["aligned_filename"]
        face = upload_data["face"]
        
        # Track files to delete later
        created_files.append(os.path.join(UPLOAD_FOLDER, filename))
        created_files.append(os.path.join(UPLOAD_FOLDER, aligned_filename))
        
        # 3. Process photo
        process_payload = {
            "filename": filename,
            "country": "usa",
            "face": face,
            "scale": 1.0,
            "x_offset": 0,
            "y_offset": 0,
            "manual_rotation": 0.0,
            "remove_bg": False,
            "bg_color_hex": "#FFFFFF",
            "auto_enhance": False,
            "brightness": 1.0,
            "contrast": 1.0,
            "sharpness": 1.0,
            "saturation": 1.0
        }
        process_response = client.post("/api/process", json=process_payload)
        assert process_response.status_code == 200
        process_data = process_response.json()
        
        assert process_data["success"] is True
        assert "filename" in process_data
        assert "url" in process_data
        
        processed_filename = process_data["filename"]
        created_files.append(os.path.join(PASSPORT_OUTPUT_FOLDER, processed_filename))
        
        # 4. Generate printable grid sheet
        layout_payload = {
            "filename": processed_filename,
            "country": "usa",
            "paper_size": "A4",
            "margin_mm": 8.0,
            "gap_mm": 2.5
        }
        sheet_response = client.post("/api/generate-sheet", json=layout_payload)
        assert sheet_response.status_code == 200
        sheet_data = sheet_response.json()
        
        assert sheet_data["success"] is True
        assert "sheet_filename" in sheet_data
        assert "pdf_filename" in sheet_data
        assert "sheet_url" in sheet_data
        assert "pdf_url" in sheet_data
        
        sheet_filename = sheet_data["sheet_filename"]
        pdf_filename = sheet_data["pdf_filename"]
        created_files.append(os.path.join(PRINTABLE_OUTPUT_FOLDER, sheet_filename))
        created_files.append(os.path.join(PRINTABLE_OUTPUT_FOLDER, pdf_filename))
        
    finally:
        # Cleanup files to prevent workspace pollution
        for filepath in created_files:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as cleanup_err:
                    print(f"Failed to remove test artifact {filepath}: {cleanup_err}")
