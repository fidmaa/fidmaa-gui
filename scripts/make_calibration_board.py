"""Generate a printable ChArUco calibration board (A4 PDF).

The board is photographed held flat next to the face; its known geometry
gives the true camera-to-board distance and scale independently of the
iPhone depth map, so depth-map errors can be measured.

Board geometry (keep in sync with the analysis script):
    7 x 9 squares, square 25 mm, ArUco marker 18 mm, DICT_5X5_100.

Usage:
    uv run python scripts/make_calibration_board.py [output.pdf]
"""

import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

SQUARES_X = 7
SQUARES_Y = 9
SQUARE_MM = 25.0
MARKER_MM = 18.0
DICTIONARY = cv2.aruco.DICT_5X5_100

PX_PER_MM = 20  # 508 dpi: 25 mm squares are exactly 500 px
DPI = PX_PER_MM * 25.4
PAGE_MM = (210, 297)
BOARD_ORIGIN_MM = (17.5, 32.0)
CHECK_LINE_MM = 100.0


def mm(value):
    return round(value * PX_PER_MM)


def make_board_image():
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_MM / 1000, MARKER_MM / 1000, dictionary
    )
    size = (mm(SQUARES_X * SQUARE_MM), mm(SQUARES_Y * SQUARE_MM))
    return Image.fromarray(board.generateImage(size, marginSize=0))


def main(output):
    page = Image.new("L", (mm(PAGE_MM[0]), mm(PAGE_MM[1])), 255)
    page.paste(make_board_image(), (mm(BOARD_ORIGIN_MM[0]), mm(BOARD_ORIGIN_MM[1])))

    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", mm(3.5))
    except OSError:
        font = ImageFont.load_default()
    header = [
        "FIDMAA - plansza kalibracyjna (ChArUco 7x9, pole 25 mm, znacznik 18 mm, DICT_5X5_100)",
        "Drukuj w skali 100% / 'Rzeczywisty rozmiar' - NIE 'Dopasuj do strony'.",
        "Sprawdz linijka: kazde pole = 25,0 mm, linia na dole = 100,0 mm.",
    ]
    for index, line in enumerate(header):
        draw.text((mm(17.5), mm(8 + index * 6)), line, fill=0, font=font)

    line_y = mm(BOARD_ORIGIN_MM[1] + SQUARES_Y * SQUARE_MM + 12)
    x0, x1 = mm(17.5), mm(17.5 + CHECK_LINE_MM)
    draw.line((x0, line_y, x1, line_y), fill=0, width=mm(0.4))
    for x in (x0, x1):
        draw.line((x, line_y - mm(3), x, line_y + mm(3)), fill=0, width=mm(0.4))
    draw.text((x1 + mm(3), line_y - mm(2)), "100 mm", fill=0, font=font)

    page.save(output, "PDF", resolution=DPI)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fidmaa_calibration_board.pdf"))
