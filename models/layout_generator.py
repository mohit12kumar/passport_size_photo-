from PIL import Image
import logging
from config import PAPER_SIZES, mm_to_px, DEFAULT_MARGIN_MM, DEFAULT_GAP_MM

logger = logging.getLogger(__name__)

def _build_grid(paper_w, paper_h, photo_w, photo_h, margin, gap):
    """
    Given fixed paper and photo dimensions (all in mm), compute the grid
    that fits the most photos and return the layout details.

    Returns:
        (count, cols, rows, positions, grid_photo_w, grid_photo_h)
    """
    try:
        avail_w = paper_w - 2 * margin
        avail_h = paper_h - 2 * margin

        if avail_w <= 0 or avail_h <= 0:
            return 0, 0, 0, [], photo_w, photo_h

        w_divisor = photo_w + gap
        h_divisor = photo_h + gap
        if w_divisor <= 0 or h_divisor <= 0:
            logger.warning("Photo width/height plus gap is non-positive. Cannot build grid.")
            return 0, 0, 0, [], photo_w, photo_h

        cols = max(0, int((avail_w + gap) // w_divisor))
        rows = max(0, int((avail_h + gap) // h_divisor))
        count = cols * rows

        positions = []
        if count > 0:
            grid_w = cols * photo_w + (cols - 1) * gap
            grid_h = rows * photo_h + (rows - 1) * gap
            start_x = margin + (avail_w - grid_w) / 2.0
            start_y = margin + (avail_h - grid_h) / 2.0
            for r in range(rows):
                for c in range(cols):
                    positions.append((
                        start_x + c * (photo_w + gap),
                        start_y + r * (photo_h + gap)
                    ))

        return count, cols, rows, positions, photo_w, photo_h
    except Exception as e:
        logger.error(f"Error in _build_grid: {e}", exc_info=True)
        return 0, 0, 0, [], photo_w, photo_h


def calculate_grid_positions(paper_w_mm, paper_h_mm, photo_w_mm, photo_h_mm,
                             margin_mm=None, gap_mm=None):
    """
    Finds the best arrangement of passport photos on a sheet.
    Tries portrait and landscape paper orientations — photos are always upright.

    Returns:
        positions_mm, cols, rows, paper_w, paper_h, photo_w, photo_h, photo_rotated
    """
    try:
        margin_mm = margin_mm if margin_mm is not None else DEFAULT_MARGIN_MM
        gap_mm    = gap_mm    if gap_mm    is not None else DEFAULT_GAP_MM
        best = {"count": -1}

        # Only try combos where the photo is upright (rotated=False)
        combos = [
            (paper_w_mm, paper_h_mm, photo_w_mm, photo_h_mm, False),   # portrait
            (paper_h_mm, paper_w_mm, photo_w_mm, photo_h_mm, False),   # landscape
        ]

        for pw, ph, fw, fh, rotated in combos:
            count, cols, rows, positions, fw_used, fh_used = _build_grid(
                pw, ph, fw, fh, margin_mm, gap_mm
            )
            if count > best["count"]:
                best = {
                    "count": count, "cols": cols, "rows": rows,
                    "positions": positions,
                    "paper_w": pw, "paper_h": ph,
                    "photo_w": fw_used, "photo_h": fh_used,
                    "rotated": rotated
                }

        if best["count"] == -1:
            return [], 0, 0, paper_w_mm, paper_h_mm, photo_w_mm, photo_h_mm, False

        return (
            best["positions"],
            best["cols"],
            best["rows"],
            best["paper_w"],
            best["paper_h"],
            best["photo_w"],
            best["photo_h"],
            best["rotated"]
        )
    except Exception as e:
        logger.error(f"Error in calculate_grid_positions: {e}", exc_info=True)
        return [], 0, 0, paper_w_mm, paper_h_mm, photo_w_mm, photo_h_mm, False


def calculate_forced_grid_positions(paper_w_mm, paper_h_mm, cols, rows, margin_mm, gap_mm, photo_aspect_ratio):
    """
    Builds a fixed cols×rows grid scaled to fill the paper with given margin and gap.
    The photo size is derived from the available cell dimensions, preserving aspect ratio.
    """
    try:
        if cols <= 0:
            cols = 1
        if rows <= 0:
            rows = 1
        if photo_aspect_ratio <= 0:
            photo_aspect_ratio = 1.0

        avail_w = paper_w_mm - 2 * margin_mm
        avail_h = paper_h_mm - 2 * margin_mm

        # Cell dimensions that exactly fill the forced grid
        cell_w = (avail_w - (cols - 1) * gap_mm) / cols
        cell_h = (avail_h - (rows - 1) * gap_mm) / rows

        # Shrink cell to preserve photo aspect ratio (fit inside cell, no rotation)
        if cell_h > 0 and cell_w / cell_h > photo_aspect_ratio:
            cell_w = cell_h * photo_aspect_ratio
        else:
            if photo_aspect_ratio > 0:
                cell_h = cell_w / photo_aspect_ratio

        # Re-centre the grid on the paper
        grid_w = cols * cell_w + (cols - 1) * gap_mm
        grid_h = rows * cell_h + (rows - 1) * gap_mm
        start_x = margin_mm + (avail_w - grid_w) / 2.0
        start_y = margin_mm + (avail_h - grid_h) / 2.0

        positions = [
            (start_x + ci * (cell_w + gap_mm), start_y + ri * (cell_h + gap_mm))
            for ri in range(rows) for ci in range(cols)
        ]

        return positions, cell_w, cell_h, False
    except Exception as e:
        logger.error(f"Error in calculate_forced_grid_positions: {e}", exc_info=True)
        return [], 10.0, 10.0, False


def generate_photo_layout(paper_width, paper_height, photo_width, photo_height, photo_count, gap=2.0, min_margin=1.0):
    """
    Dynamically calculate the best centered layout grid for a given photo count.

    Args:
        paper_width: Paper width in mm (e.g. 102.0 for 4x6 portrait)
        paper_height: Paper height in mm (e.g. 152.0 for 4x6 portrait)
        photo_width: Photo width in mm
        photo_height: Photo height in mm
        photo_count: Selected number of photos (1, 2, 4, 6)
        gap: Gap between photos in mm
        min_margin: Minimum margin from page edges in mm

    Returns:
        A dict containing layout details or None if it doesn't fit.
    """
    try:
        shapes_by_count = {
            1: [(1, 1)],
            2: [(1, 2), (2, 1)],
            4: [(2, 2)],
            6: [(2, 3), (3, 2)]
        }

        if photo_count not in shapes_by_count:
            return None

        candidates = []
        min_paper_dim = min(paper_width, paper_height)
        max_paper_dim = max(paper_width, paper_height)

        orientations = [
            (min_paper_dim, max_paper_dim), # Portrait
            (max_paper_dim, min_paper_dim)  # Landscape
        ]

        for pw, ph in orientations:
            for c, r in shapes_by_count[photo_count]:
                grid_w = c * photo_width + (c - 1) * gap
                grid_h = r * photo_height + (r - 1) * gap

                mx = (pw - grid_w) / 2.0
                my = (ph - grid_h) / 2.0

                if grid_w <= pw and grid_h <= ph and mx >= min_margin and my >= min_margin:
                    candidates.append({
                        "paper_w": pw,
                        "paper_h": ph,
                        "columns": c,
                        "rows": r,
                        "margin_x": mx,
                        "margin_y": my,
                        "balance": abs(mx - my)
                    })

        if not candidates:
            return None

        # Select the layout candidate with the most balanced margins
        best = min(candidates, key=lambda x: x["balance"])

        positions = []
        start_x = best["margin_x"]
        start_y = best["margin_y"]
        cols = best["columns"]
        rows = best["rows"]

        for r in range(rows):
            for c in range(cols):
                positions.append((
                    start_x + c * (photo_width + gap),
                    start_y + r * (photo_height + gap)
                ))

        return {
            "positions": positions,
            "columns": cols,
            "rows": rows,
            "paper_w": best["paper_w"],
            "paper_h": best["paper_h"],
            "margin_x": start_x,
            "margin_y": start_y
        }
    except Exception as e:
        logger.error(f"Error in generate_photo_layout: {e}", exc_info=True)
        return None


def generate_printable_sheet(passport_pil_image, paper_size_key, photo_w_mm, photo_h_mm,
                              margin_mm=None, gap_mm=None, draw_guides_func=None, photo_count=None):
    """
    Arranges multiple passport photos in a grid on a blank canvas.
    """
    if passport_pil_image is None:
        raise ValueError("passport_pil_image is None")

    try:
        paper = PAPER_SIZES.get(paper_size_key)
        if not paper:
            raise ValueError(f"Paper size '{paper_size_key}' not supported.")

        # Ignore custom count for unsupported paper sizes to enforce full capacity printing
        if paper_size_key not in ["4x6", "5x7"]:
            photo_count = None

        paper_w_mm = float(paper["width_mm"])
        paper_h_mm = float(paper["height_mm"])

        # Resolve margin/gap: caller-supplied → paper config default → global default
        if margin_mm is None:
            margin_mm = float(paper.get("default_margin_mm", DEFAULT_MARGIN_MM))
        if gap_mm is None:
            gap_mm = float(paper.get("default_gap_mm", DEFAULT_GAP_MM))

        forced_cols = paper.get("forced_cols")
        forced_rows = paper.get("forced_rows")
        photo_rotated = False

        # Check if we should use the custom photo count layout engine (1, 2, 4, 6 photos on 4x6/5x7)
        use_custom_layout = False
        if paper_size_key in ["4x6", "5x7"] and photo_count in [1, 2, 4, 6]:
            use_custom_layout = True

        if use_custom_layout:
            positions_mm = []
            cols = 0
            rows = 0
            actual_w_mm = paper_w_mm
            actual_h_mm = paper_h_mm
            actual_photo_w_mm = photo_w_mm
            actual_photo_h_mm = photo_h_mm
            photo_rotated = False
            used_margin = margin_mm
            used_gap = gap_mm

            layout_result = generate_photo_layout(
                paper_width=paper_w_mm,
                paper_height=paper_h_mm,
                photo_width=photo_w_mm,
                photo_height=photo_h_mm,
                photo_count=photo_count,
                gap=used_gap,
                min_margin=1.0
            )

            if layout_result:
                positions_mm = layout_result["positions"]
                cols = layout_result["columns"]
                rows = layout_result["rows"]
                actual_w_mm = layout_result["paper_w"]
                actual_h_mm = layout_result["paper_h"]
                used_margin = min(layout_result["margin_x"], layout_result["margin_y"])
            else:
                use_custom_layout = False

        if not use_custom_layout:
            if forced_cols and forced_rows:
                eff_margin = float(paper.get("forced_margin_mm", margin_mm))
                eff_gap    = float(paper.get("forced_gap_mm",    gap_mm))
                if photo_h_mm <= 0:
                    aspect = 1.0
                else:
                    aspect = photo_w_mm / photo_h_mm

                positions_mm, actual_photo_w_mm, actual_photo_h_mm, photo_rotated = \
                    calculate_forced_grid_positions(
                        paper_w_mm, paper_h_mm,
                        forced_cols, forced_rows,
                        eff_margin, eff_gap,
                        aspect
                    )
                cols = forced_cols
                rows = forced_rows
                actual_w_mm = paper_w_mm
                actual_h_mm = paper_h_mm
                used_margin = eff_margin
                used_gap    = eff_gap
            else:
                # Auto-fit grid: try all orientation combos
                (positions_mm, cols, rows,
                 actual_w_mm, actual_h_mm,
                 actual_photo_w_mm, actual_photo_h_mm,
                 photo_rotated) = calculate_grid_positions(
                    paper_w_mm, paper_h_mm, photo_w_mm, photo_h_mm, margin_mm, gap_mm
                )
                used_margin = margin_mm
                used_gap    = gap_mm

        if not positions_mm:
            logger.warning("No grid positions calculated. Photo might be larger than paper size.")
            # Return a blank sheet with a warning
            blank_w = mm_to_px(actual_w_mm)
            blank_h = mm_to_px(actual_h_mm)
            blank_sheet = Image.new("RGB", (blank_w, blank_h), (255, 255, 255))
            return blank_sheet, {
                "width_mm":     actual_w_mm,
                "height_mm":    actual_h_mm,
                "columns":      0,
                "rows":         0,
                "count":        0,
                "photo_rotated": False,
                "positions":    []
            }

        # ── Build the sheet canvas ──────────────────────────────────────────────
        sheet_w_px = mm_to_px(actual_w_mm)
        sheet_h_px = mm_to_px(actual_h_mm)

        # Guard against zero or huge dimensions
        if sheet_w_px <= 0:
            sheet_w_px = 100
        if sheet_h_px <= 0:
            sheet_h_px = 100

        sheet = Image.new("RGB", (sheet_w_px, sheet_h_px), (255, 255, 255))

        # Prepare the passport photo at the right pixel size, rotating if needed
        photo_w_px = mm_to_px(actual_photo_w_mm)
        photo_h_px = mm_to_px(actual_photo_h_mm)

        if photo_rotated:
            src = passport_pil_image.rotate(-90, expand=True)
        else:
            src = passport_pil_image

        BORDER_MM = 0.5
        border_px = mm_to_px(BORDER_MM)
        content_w_px = max(1, photo_w_px - 2 * border_px)
        content_h_px = max(1, photo_h_px - 2 * border_px)

        passport_resized = src.resize((content_w_px, content_h_px), Image.Resampling.LANCZOS)

        # Paste photos into the grid (only up to photo_count if specified)
        num_to_paste = len(positions_mm)
        if photo_count is not None:
            try:
                num_to_paste = min(int(photo_count), len(positions_mm))
            except (ValueError, TypeError):
                pass

        for idx in range(num_to_paste):
            x_mm, y_mm = positions_mm[idx]
            cell_x_px = mm_to_px(x_mm)
            cell_y_px = mm_to_px(y_mm)

            # Make sure we paste inside boundaries
            if cell_x_px + border_px + content_w_px <= sheet_w_px and cell_y_px + border_px + content_h_px <= sheet_h_px:
                sheet.paste(passport_resized, (cell_x_px + border_px, cell_y_px + border_px))

        # Draw cutting guides if requested (always using the full grid positions for alignment guides)
        if draw_guides_func and positions_mm:
            try:
                sheet = draw_guides_func(
                    sheet, positions_mm,
                    actual_photo_w_mm, actual_photo_h_mm,
                    actual_w_mm, actual_h_mm,
                    used_margin, used_gap
                )
            except Exception as guide_e:
                logger.error(f"Failed to draw cutting guides: {guide_e}", exc_info=True)

        return sheet, {
            "width_mm":     actual_w_mm,
            "height_mm":    actual_h_mm,
            "columns":      cols,
            "rows":         rows,
            "count":        num_to_paste, # Return the actual number of photos printed
            "photo_rotated": photo_rotated,
            "positions":    [(mm_to_px(x), mm_to_px(y)) for x, y in positions_mm]
        }
    except Exception as e:
        logger.error(f"Error in generate_printable_sheet: {e}", exc_info=True)
        # Safe fallback: create a simple blank template sheet
        fallback_w = mm_to_px(100.0)
        fallback_h = mm_to_px(150.0)
        fallback_sheet = Image.new("RGB", (fallback_w, fallback_h), (240, 240, 240))
        return fallback_sheet, {
            "width_mm":     100.0,
            "height_mm":    150.0,
            "columns":      0,
            "rows":         0,
            "count":        0,
            "photo_rotated": False,
            "positions":    []
        }
