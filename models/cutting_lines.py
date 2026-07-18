"""
Professional Cutting Guide Renderer
=====================================
Draws crosshair-style cutting marks at every corner and midpoint of every
passport photo on the printable sheet.

Design:
  • Short registration corner lines extending outward from each photo corner
  • Thin dashed lines running across the full sheet at every row/column cut line
  • No lines are drawn inside the photo itself — only in the gap / margin area
"""

from PIL import ImageDraw
import logging
from config import mm_to_px

logger = logging.getLogger(__name__)


def draw_cutting_lines(sheet_pil_image, positions_mm, photo_w_mm, photo_h_mm,
                       paper_w_mm, paper_h_mm, margin_mm, gap_mm):
    """
    Draws professional crosshair cutting guides on the sheet.

    Renders:
      1. Full-width horizontal dashed lines at every row boundary.
      2. Full-height vertical dashed lines at every column boundary.
      3. Corner registration marks (L-shaped brackets) at each photo corner.

    All guides are drawn only in the margin/gap area — never on top of
    the photo itself — so the output looks like a real photo lab print.
    """
    if sheet_pil_image is None:
        logger.error("draw_cutting_lines received None sheet_pil_image")
        return sheet_pil_image

    try:
        draw = ImageDraw.Draw(sheet_pil_image)

        GUIDE_COLOR   = (200, 200, 200)   # Light gray
        DASH_COLOR    = (200, 200, 200)   # Light gray
        LINE_WIDTH    = 3                 # 3 pixels is ~0.254 mm at 300 DPI
        MARK_LEN_PX   = mm_to_px(3.0)     # 3 mm corner mark arm length
        DASH_ON       = mm_to_px(2.0)     # dash on-length
        DASH_OFF      = mm_to_px(1.5)     # dash off-length

        sheet_w_px = mm_to_px(paper_w_mm)
        sheet_h_px = mm_to_px(paper_h_mm)

        if sheet_w_px <= 0 or sheet_h_px <= 0:
            logger.warning("Invalid sheet dimensions for drawing cutting lines.")
            return sheet_pil_image

        # ── Collect unique X and Y cut-line positions ─────────────────────────────
        cut_xs = set()   # vertical cut lines (left & right edge of every photo)
        cut_ys = set()   # horizontal cut lines (top & bottom edge of every photo)

        for x_mm, y_mm in positions_mm:
            cut_xs.add(mm_to_px(x_mm))
            cut_xs.add(mm_to_px(x_mm + photo_w_mm))
            cut_ys.add(mm_to_px(y_mm))
            cut_ys.add(mm_to_px(y_mm + photo_h_mm))

        # ── Draw full-sheet dashed lines ──────────────────────────────────────────
        def _draw_dashed_hline(y):
            """Horizontal dashed line across the full sheet width."""
            try:
                x = 0
                on = True
                while x < sheet_w_px:
                    x_end = min(x + (DASH_ON if on else DASH_OFF), sheet_w_px)
                    if x_end <= x:
                        break # Prevent infinite loop if DASH_ON/OFF are non-positive
                    if on:
                        draw.line([(x, y), (x_end, y)], fill=DASH_COLOR, width=LINE_WIDTH)
                    x = x_end
                    on = not on
            except Exception as e:
                logger.error(f"Error drawing dashed hline at y={y}: {e}")

        def _draw_dashed_vline(x):
            """Vertical dashed line across the full sheet height."""
            try:
                y = 0
                on = True
                while y < sheet_h_px:
                    y_end = min(y + (DASH_ON if on else DASH_OFF), sheet_h_px)
                    if y_end <= y:
                        break # Prevent infinite loop if DASH_ON/OFF are non-positive
                    if on:
                        draw.line([(x, y), (x, y_end)], fill=DASH_COLOR, width=LINE_WIDTH)
                    y = y_end
                    on = not on
            except Exception as e:
                logger.error(f"Error drawing dashed vline at x={x}: {e}")

        for cy in cut_ys:
            _draw_dashed_hline(cy)
        for cx in cut_xs:
            _draw_dashed_vline(cx)

        # ── Draw L-shaped corner registration marks at each photo ─────────────────
        for x_mm, y_mm in positions_mm:
            try:
                x1 = mm_to_px(x_mm)
                y1 = mm_to_px(y_mm)
                x2 = mm_to_px(x_mm + photo_w_mm)
                y2 = mm_to_px(y_mm + photo_h_mm)

                corners = [
                    # (corner_x, corner_y, h_dir, v_dir)
                    (x1, y1, -1, -1),   # top-left
                    (x2, y1, +1, -1),   # top-right
                    (x1, y2, -1, +1),   # bottom-left
                    (x2, y2, +1, +1),   # bottom-right
                ]

                for cx, cy, hd, vd in corners:
                    # Horizontal arm of the L
                    hx1, hx2 = (cx, cx + hd * MARK_LEN_PX) if hd > 0 else (cx + hd * MARK_LEN_PX, cx)
                    draw.line([(hx1, cy), (hx2, cy)], fill=GUIDE_COLOR, width=LINE_WIDTH)

                    # Vertical arm of the L
                    vy1, vy2 = (cy, cy + vd * MARK_LEN_PX) if vd > 0 else (cy + vd * MARK_LEN_PX, cy)
                    draw.line([(cx, vy1), (cx, vy2)], fill=GUIDE_COLOR, width=LINE_WIDTH)
            except Exception as corner_e:
                logger.error(f"Error drawing corner marks for position ({x_mm}, {y_mm}): {corner_e}")

        return sheet_pil_image
    except Exception as e:
        logger.error(f"Error drawing cutting guides: {e}", exc_info=True)
        return sheet_pil_image
