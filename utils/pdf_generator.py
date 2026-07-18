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


def generate_compliance_pdf(compliance_data, output_pdf_path, country_name, doc_type, quality_score):
    """
    Generates a professional compliance certificate PDF detailing quality metrics
    and official biometric check statuses.
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from datetime import datetime

    try:
        # Page size: Letter/A4
        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=(612.0, 792.0),
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            name='TitleStyle',
            fontName='Helvetica-Bold',
            fontSize=24,
            leading=28,
            textColor=colors.HexColor('#1e293b'),
            alignment=0,
            spaceAfter=5
        )
        
        header_style = ParagraphStyle(
            name='HeaderStyle',
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#0f766e'),
            spaceBefore=15,
            spaceAfter=8
        )
        
        body_style = ParagraphStyle(
            name='BodyStyle',
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#475569')
        )
        
        bold_body = ParagraphStyle(
            name='BoldBody',
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#1e293b')
        )
        
        story = []
        
        # 1. Header Section
        story.append(Paragraph("AI Passport Photo Studio", title_style))
        story.append(Paragraph(f"<b>Biometric Compliance Certificate</b> — Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
        story.append(Spacer(1, 10))
        story.append(Table([[Paragraph("", ParagraphStyle('line', borderPadding=0, spaceBefore=0, spaceAfter=0, borderStyle='solid', borderWidth=1, borderColor=colors.HexColor('#cbd5e1')))]], colWidths=[532.0]))
        story.append(Spacer(1, 15))
        
        # 2. General Specs & Result Box
        status_text = "PASSED" if compliance_data.get("passed", False) else "WARNING / FAILED"
        status_color = '#10b981' if compliance_data.get("passed", False) else '#ef4444'
        
        specs_data = [
            [Paragraph("<b>Target Country / Spec:</b>", body_style), Paragraph(country_name.upper(), bold_body)],
            [Paragraph("<b>Document Type:</b>", body_style), Paragraph(doc_type.upper(), bold_body)],
            [Paragraph("<b>Biometric Validation Status:</b>", body_style), 
             Paragraph(f"<b>{status_text}</b>", ParagraphStyle('status', fontName='Helvetica-Bold', fontSize=10, leading=14, textColor=colors.HexColor(status_color)))],
            [Paragraph("<b>Overall Quality Score:</b>", body_style), Paragraph(f"<b>{quality_score}%</b>", bold_body)]
        ]
        
        specs_table = Table(specs_data, colWidths=[200.0, 332.0])
        specs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(specs_table)
        story.append(Spacer(1, 15))
        
        # 3. Validation Checklist Table
        story.append(Paragraph("Biometric Checklist Details", header_style))
        
        checklist_headers = [
            Paragraph("<b>Biometric Check Item</b>", bold_body), 
            Paragraph("<b>Status</b>", bold_body), 
            Paragraph("<b>Details / Guidance</b>", bold_body)
        ]
        
        table_rows = [checklist_headers]
        
        checks = compliance_data.get("checks", {})
        for key, value in checks.items():
            if not isinstance(value, dict):
                continue
            status = value.get("status", "PASS")
            msg = value.get("message", "Rule compliant.")
            
            label = key.replace("_", " ").title()
            
            st_color = '#10b981'
            if status == "WARN":
                st_color = '#f59e0b'
            elif status == "FAIL":
                st_color = '#ef4444'
                
            status_para = Paragraph(f"<b>{status}</b>", ParagraphStyle('sc', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor(st_color)))
            
            table_rows.append([
                Paragraph(label, body_style),
                status_para,
                Paragraph(msg, body_style)
            ])
            
        check_table = Table(table_rows, colWidths=[150.0, 80.0, 302.0])
        check_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(check_table)
        story.append(Spacer(1, 15))
        
        # 4. Footer Note
        story.append(Paragraph("<b>Notice:</b> This document acts as a compliance test certificate. Printed scaling and paper settings must exactly match target metrics to preserve standard mm dimensions.", ParagraphStyle('note', fontName='Helvetica-Oblique', fontSize=8, leading=10, textColor=colors.HexColor('#64748b'))))
        
        doc.build(story)
        return output_pdf_path
        
    except Exception as e:
        logger.error(f"Failed to generate compliance report PDF: {e}", exc_info=True)
        raise e
