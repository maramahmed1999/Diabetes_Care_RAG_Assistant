"""
Phase 1 — Text extraction.

Pulls text out of the PDFs while respecting reading order (handles
multi-column layouts), exactly as the original notebook does.
"""

import pymupdf


def fix_pdf_ligatures(text: str) -> str:
    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb00": "ff",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u00a0": " ",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def extract_lines(page) -> list[dict]:
    """
    Extract every text line with its position on the page.
    """
    data = page.get_text("dict")

    lines = []

    for block in data["blocks"]:

        # Ignore image blocks
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):

            text = "".join(
                span["text"]
                for span in line.get("spans", [])
            ).strip()

            if not text:
                continue

            x0, y0, x1, y1 = line["bbox"]

            lines.append({
                "text": fix_pdf_ligatures(text),
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            })

    return lines


def extract_blocks(page) -> list[dict]:
    raw_blocks = page.get_text("blocks")

    blocks = []

    for block in raw_blocks:

        x0, y0, x1, y1, text = block[:5]

        text = text.strip()

        if not text:
            continue

        blocks.append({
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "text": text,
            "width": float(x1 - x0),
            "height": float(y1 - y0),
        })

    return blocks


def detect_columns(lines, tolerance: int = 20) -> list[float]:
    """
    Detect text columns based on the x-position of text lines.
    """
    x_positions = sorted(line["x0"] for line in lines)

    columns = []

    for x in x_positions:

        if not columns:
            columns.append([x])
            continue

        # if x is close to an existing column, group it in
        if abs(x - columns[-1][-1]) <= tolerance:
            columns[-1].append(x)
        else:
            columns.append([x])

    # average x per column
    column_centers = [
        sum(column) / len(column)
        for column in columns
    ]

    return column_centers


def assign_columns(lines, tolerance: int = 30) -> list[float]:
    """
    Group lines into columns according to their x position.
    """
    column_centers = []

    for line in lines:

        x = line["x0"]

        assigned = False

        for i, center in enumerate(column_centers):

            if abs(x - center) <= tolerance:
                # update center
                column_centers[i] = (center + x) / 2
                assigned = True
                break

        if not assigned:
            column_centers.append(x)

    # sort columns left to right
    column_centers.sort()

    return column_centers


def order_lines(lines) -> list[dict]:
    """
    Order lines from top to bottom inside each column,
    then move from left column to right column.
    """
    column_centers = assign_columns(lines)

    ordered_columns = []

    for center in column_centers:

        column = [
            line
            for line in lines
            if abs(line["x0"] - center) <= 30
        ]

        column.sort(key=lambda x: x["y0"])

        ordered_columns.append(column)

    # left -> middle -> right
    ordered_lines = []

    for column in ordered_columns:
        ordered_lines.extend(column)

    return ordered_lines


def extract_ordered_page(pdf_path, page_number: int) -> str:
    doc = pymupdf.open(pdf_path)

    page = doc[page_number - 1]

    lines = extract_lines(page)

    ordered_lines = order_lines(lines)

    text = "\n".join(line["text"] for line in ordered_lines)

    doc.close()

    return text


def inspect_page(pdf_path, page_number: int) -> None:
    """
    Debug helper: prints every extracted line with its coordinates.
    Kept from the original notebook for troubleshooting layouts.
    """
    doc = pymupdf.open(pdf_path)

    page = doc[page_number - 1]

    lines = extract_lines(page)

    print(f"Page size: {page.rect.width:.1f} x {page.rect.height:.1f}")
    print(f"Number of lines: {len(lines)}")
    print("=" * 100)

    for i, line in enumerate(lines):
        print(
            f"{i:03d} | "
            f"x={line['x0']:6.1f} | "
            f"y={line['y0']:6.1f} | "
            f"{line['text']}"
        )

    doc.close()
