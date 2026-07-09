from reportlab.pdfgen import canvas
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

def mm_to_points(mm):
    """Convert millimeters to ReportLab PDF points (1/72 inch)."""
    try:
        return (float(mm) / 25.4) * 72.0
    except Exception as e:
        logger.error(f"Error converting mm to points: {e}")
        return 0.0

def export_sheet_to_pdf(sheet_pil_image, output_pdf_path, paper_w_mm, paper_h_mm):
    """
    Exports the generated printable sheet PIL image to a high-quality PDF.
    
    The PDF page size will exactly match the paper size in points,
    and the image will be drawn at maximum resolution to fit the page.
    
    output_pdf_path can be a file path string or a file-like object (e.g. io.BytesIO).
    """
    if sheet_pil_image is None:
        raise ValueError("sheet_pil_image is None")

    # 1. Convert paper dimensions to PDF points
    page_w = mm_to_points(paper_w_mm)
    page_h = mm_to_points(paper_h_mm)
    
    if page_w <= 0 or page_h <= 0:
        raise ValueError(f"Invalid paper dimensions in points: {page_w}x{page_h}")
    
    # 2. Save PIL image to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_img_path = os.path.join(temp_dir, f"temp_print_sheet_{os.getpid()}.jpg")
    
    try:
        # Save high quality JPEG (JPEG is natively supported by ReportLab's drawImage
        # and directly embedded, preventing memory usage issues)
        sheet_pil_image.save(temp_img_path, "JPEG", quality=95, dpi=(300, 300))
        
        # 3. Create ReportLab canvas (accepts path string or file-like object)
        c = canvas.Canvas(output_pdf_path, pagesize=(page_w, page_h))
        
        # 4. Draw image to cover the entire page
        c.drawImage(temp_img_path, 0, 0, width=page_w, height=page_h)
        
        # 5. Save the PDF
        c.showPage()
        c.save()
        
    except Exception as e:
        logger.error(f"Failed to export sheet to PDF: {e}", exc_info=True)
        raise RuntimeError(f"PDF generation failed: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception as clean_e:
                logger.warning(f"Failed to remove temp file {temp_img_path}: {clean_e}")
                
    return output_pdf_path
