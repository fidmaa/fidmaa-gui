#!/usr/bin/env python
"""Throwaway probe: how do MediaPipe FaceMesh landmarks look in the lip region?

Purpose: eyeball whether FaceMesh carries enough signal to separate ULBT
(upper lip bite test) class I / II / III.

ULBT grading is defined by where the *lower incisor edge* lands on the upper
lip:

    class I   - lower incisors bite the upper lip ABOVE the vermilion line
    class II  - lower incisors bite the upper lip BELOW the vermilion line
    class III - lower incisors cannot reach the upper lip at all

FaceMesh gives lip contours but NO teeth landmarks, so this script also
overlays the teeth matte from portrait_analyser (when the input is a HEIC with
iOS portrait data) - the teeth matte is what actually carries the incisor edge.

This is the research counterpart of `portrait_analyser.ulbt`, which ships the
conclusion: FaceMesh's "upper lip" band is not upper lip, so the measurement
must key off pixel colour rather than model labels. Use `measure-ulbt` for the
number; use this to look at why.

Usage:
    uv run python scripts/ulbt_facemesh_probe.py IMAGE [-o OUT.png] [--margin 0.10]

    # with the FaRL semantic-parsing panel (not a project dependency):
    uv run --with pyfacer --with torch --with torchvision \\
        python scripts/ulbt_facemesh_probe.py IMAGE --parse

IMAGE may be an iPhone HEIC portrait or any plain JPG/PNG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# --- FaceMesh lip contour groups (canonical 468/478-point topology) ---------
# Four rings, outside-in from the top. The boundary between UPPER_LIP_OUTER and
# UPPER_LIP_INNER is the upper vermilion border - the ULBT decision line.
UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LOWER_LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

LIP_GROUPS = [
    ("upper lip OUTER (vermilion border)", UPPER_LIP_OUTER, "#e6194b"),
    ("upper lip INNER (wet line)", UPPER_LIP_INNER, "#f58231"),
    ("lower lip INNER (wet line)", LOWER_LIP_INNER, "#3cb44b"),
    ("lower lip OUTER", LOWER_LIP_OUTER, "#4363d8"),
]

ALL_LIP_INDICES = sorted(set(UPPER_LIP_OUTER + UPPER_LIP_INNER + LOWER_LIP_INNER + LOWER_LIP_OUTER))

# Named single landmarks that matter for ULBT reasoning.
KEY_POINTS = {
    0: "labiale superius (top of upper vermilion)",
    13: "upper lip inner centre",
    14: "lower lip inner centre",
    17: "labiale inferius (lower lip outer centre)",
    61: "mouth corner L",
    291: "mouth corner R",
    152: "menton (chin tip)",
    1: "nose tip",
}

# Blendshapes worth printing - jaw protrusion is the mechanical basis of ULBT.
BLENDSHAPES_OF_INTEREST = (
    "jawForward",
    "jawOpen",
    "jawLeft",
    "jawRight",
    "mouthClose",
    "mouthPucker",
    "mouthFunnel",
    "mouthShrugUpper",
    "mouthShrugLower",
    "mouthRollUpper",
    "mouthRollLower",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
)


def load_photo(path: Path):
    """Return (RGB PIL image, teethmap or None).

    HEIC goes through portrait_analyser so we also get the iOS teeth matte;
    anything else is loaded straight through Pillow.
    """
    if path.suffix.lower() in (".heic", ".heif"):
        from portrait_analyser.ios import load_image

        portrait = load_image(str(path))
        if portrait is None:
            raise SystemExit(f"portrait_analyser could not load {path}")
        return portrait.photo.convert("RGB"), portrait.teethmap

    return Image.open(path).convert("RGB"), None


def detect_landmarks(rgb: np.ndarray):
    """Run FaceLandmarker. Returns (landmarks, blendshapes or None)."""
    import mediapipe as mp
    from portrait_analyser.pose import _get_face_mesh_model_path

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_get_face_mesh_model_path()),
        num_faces=1,
        output_face_blendshapes=True,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise SystemExit("FaceMesh found no face in this image.")

    blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
    return result.face_landmarks[0], blendshapes


def to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    """Normalised landmarks -> (N, 2) array of pixel coordinates."""
    return np.array([(lm.x * width, lm.y * height) for lm in landmarks])


def lip_roi(points: np.ndarray, width: int, height: int, margin: float):
    """Bounding box around the lips, expanded by `margin` (fraction of size)."""
    lips = points[ALL_LIP_INDICES]
    x0, y0 = lips.min(axis=0)
    x1, y1 = lips.max(axis=0)
    pad_x = (x1 - x0) * margin
    pad_y = (y1 - y0) * margin
    return (
        max(0.0, x0 - pad_x),
        max(0.0, y0 - pad_y),
        min(float(width), x1 + pad_x),
        min(float(height), y1 + pad_y),
    )


def points_inside(points: np.ndarray, box) -> list[int]:
    x0, y0, x1, y1 = box
    xs, ys = points[:, 0], points[:, 1]
    inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return np.nonzero(inside)[0].tolist()


def draw_contours(ax, points: np.ndarray) -> None:
    """Colour-coded lip rings with the ULBT-relevant points called out."""
    for label, indices, colour in LIP_GROUPS:
        ring = points[indices]
        ax.plot(ring[:, 0], ring[:, 1], "-", color=colour, lw=1.6, label=label, zorder=3)
        ax.plot(ring[:, 0], ring[:, 1], "o", color=colour, ms=4, zorder=4)

    for idx, name in KEY_POINTS.items():
        x, y = points[idx]
        ax.plot(x, y, "*", color="yellow", ms=14, mec="black", mew=0.6, zorder=5)
        ax.annotate(
            f"{idx} {name.split('(')[0].strip()}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=5,
            color="yellow",
            zorder=6,
            path_effects=None,
        )


def metrics(points: np.ndarray) -> dict[str, float]:
    """Scale-invariant candidate measurements for ULBT discrimination."""
    corner_l, corner_r = points[61], points[291]
    mouth_width = float(np.hypot(*(corner_r - corner_l)))

    y_upper_vermilion = float(points[0][1])
    y_upper_inner = float(points[13][1])
    y_lower_inner = float(points[14][1])
    y_lower_outer = float(points[17][1])

    return {
        "mouth_width_px": mouth_width,
        # How far the lower lip's top edge rides up over the upper vermilion
        # border. Positive => lower lip is ABOVE the vermilion line (class I
        # territory); negative => it sits below (class II/III territory).
        "lower_edge_over_vermilion / width": (y_upper_vermilion - y_lower_inner) / mouth_width,
        "lip_gap (14-13) / width": (y_lower_inner - y_upper_inner) / mouth_width,
        "upper_lip_thickness (13-0) / width": (y_upper_inner - y_upper_vermilion) / mouth_width,
        "lower_lip_thickness (17-14) / width": (y_lower_outer - y_lower_inner) / mouth_width,
        "chin_below_lower_lip (152-17) / width": (points[152][1] - y_lower_outer) / mouth_width,
    }


def polygon_mask(points: np.ndarray, outer_idx, inner_idx, size) -> np.ndarray:
    """Boolean mask of the band between two landmark rings (both run L->R)."""
    from PIL import ImageDraw

    poly = [tuple(points[i]) for i in outer_idx]
    poly += [tuple(points[i]) for i in reversed(inner_idx)]
    canvas = Image.new("L", size, 0)
    ImageDraw.Draw(canvas).polygon(poly, fill=255)
    return np.asarray(canvas) > 0


def skin_reference_mask(points: np.ndarray, size, mouth_width: float) -> np.ndarray:
    """Skin strip directly above the upper-lip outer ring (same illumination).

    Starts a little above the ring so it cannot straddle the vermilion edge.
    """
    ring = points[UPPER_LIP_OUTER]
    near = ring - np.array([0.0, 0.04 * mouth_width])
    far = ring - np.array([0.0, 0.16 * mouth_width])
    poly = [tuple(p) for p in near] + [tuple(p) for p in far[::-1]]
    from PIL import ImageDraw

    canvas = Image.new("L", size, 0)
    ImageDraw.Draw(canvas).polygon(poly, fill=255)
    return np.asarray(canvas) > 0


def classify_upper_lip_band(rgb: np.ndarray, points: np.ndarray, teeth: np.ndarray | None):
    """Is FaceMesh's "upper lip" band actually skin, vermilion, or teeth?

    Builds reference colour distributions from regions we are sure about
    (skin strip above the ring; the lower lip, which is unbitten vermilion in
    a ULBT photo; teeth from the iOS matte), then assigns every pixel of the
    upper-lip band to the nearest reference in CIELAB chromaticity (a*, b*).
    L* is dropped so shading does not drive the decision.
    """
    import cv2

    height, width = rgb.shape[:2]
    size = (width, height)
    mouth_width = float(np.hypot(*(points[291] - points[61])))

    band = polygon_mask(points, UPPER_LIP_OUTER, UPPER_LIP_INNER, size)
    lower_lip = polygon_mask(points, LOWER_LIP_INNER, LOWER_LIP_OUTER, size)
    skin = skin_reference_mask(points, size, mouth_width)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = lab[:, :, 1:]

    refs = {
        "skin": skin,
        "lip": lower_lip,
    }
    if teeth is not None:
        teeth_ref = teeth > 200
        if teeth_ref.sum() > 50:
            refs["teeth"] = teeth_ref

    names = list(refs)
    centres = np.array([np.median(chroma[m], axis=0) for m in refs.values()])

    pixels = chroma[band]
    if pixels.size == 0:
        return None
    distances = np.linalg.norm(pixels[:, None, :] - centres[None, :, :], axis=2)
    assignment = distances.argmin(axis=1)

    shares = {name: float((assignment == i).mean()) for i, name in enumerate(names)}
    swatches = {name: tuple(int(v) for v in np.median(rgb[m], axis=0)) for name, m in refs.items()}

    label_map = np.full((height, width), -1, dtype=np.int8)
    label_map[band] = assignment
    return {
        "shares": shares,
        "swatches": swatches,
        "names": names,
        "label_map": label_map,
        "band_px": int(band.sum()),
        "ref_px": {name: int(m.sum()) for name, m in refs.items()},
    }


# Colours used to paint the band classification (skin / lip / teeth).
CLASS_COLOURS = {"skin": "#ff00ff", "lip": "#00ffff", "teeth": "#ffffff"}


def build_figure(rgb, points, box, teethmap, blendshapes, parsing, source: Path, out_path: Path):
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = box
    inside = points_inside(points, box)

    teeth = None
    if teethmap is not None:
        teeth = np.asarray(teethmap.convert("L").resize((width, height), Image.BILINEAR))
    colour = classify_upper_lip_band(rgb, points, teeth)

    # Portrait face on the left, the wide/short zoom crops stacked right.
    rows = 5 if parsing is not None else 4
    fig = plt.figure(figsize=(17, 4.3 * rows))
    gs = fig.add_gridspec(rows, 2, width_ratios=(1, 2.3), wspace=0.04, hspace=0.12)
    ax_face = fig.add_subplot(gs[:, 0])
    axes = [
        [ax_face, fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])],
    ]
    ax_colour = fig.add_subplot(gs[3, 1])
    ax_parse = fig.add_subplot(gs[4, 1]) if parsing is not None else None
    fig.suptitle(f"FaceMesh in the lip region — {source.name}", fontsize=13)

    # 1. whole face, ROI marked
    ax = axes[0][0]
    ax.imshow(rgb)
    ax.plot(points[:, 0], points[:, 1], ".", color="#00ffcc", ms=0.7, zorder=2)
    ax.add_patch(
        mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1.5, zorder=3)
    )
    ax.set_title(f"all {len(points)} landmarks; ROI boxed", fontsize=9)

    # 2. zoom, every landmark in ROI numbered
    ax = axes[0][1]
    ax.imshow(rgb)
    for idx in inside:
        x, y = points[idx]
        ax.plot(x, y, ".", color="#00ffcc", ms=3, zorder=3)
        ax.annotate(
            str(idx),
            (x, y),
            textcoords="offset points",
            xytext=(2, 2),
            fontsize=4.5,
            color="#00ffcc",
            zorder=4,
        )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title(f"ZOOM: all {len(inside)} landmarks in ROI, index-numbered", fontsize=9)

    # 3. zoom, lip rings colour-coded
    ax = axes[1][0]
    ax.imshow(rgb)
    draw_contours(ax, points)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)
    ax.set_title("ZOOM: lip rings — red line is the upper vermilion border", fontsize=9)

    # 4. zoom + teeth matte (the incisor edge FaceMesh does not give us)
    ax = axes[1][1]
    ax.imshow(rgb)
    if teeth is not None:
        ax.imshow(
            np.ma.masked_less(teeth, 40),
            cmap="autumn",
            alpha=0.55,
            interpolation="nearest",
            zorder=2,
        )
        title = "ZOOM: teeth matte (portrait_analyser) + vermilion border"
    else:
        title = "ZOOM: no teeth matte available for this input"
    ring = points[UPPER_LIP_OUTER]
    ax.plot(ring[:, 0], ring[:, 1], "-", color="#e6194b", lw=1.8, zorder=4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title(title, fontsize=9)

    # 5. is FaceMesh's "upper lip" band skin, vermilion, or teeth?
    ax_colour.imshow(rgb)
    if colour is None:
        ax_colour.set_title("ZOOM: upper-lip band is empty — no colour verdict", fontsize=9)
    else:
        for i, name in enumerate(colour["names"]):
            layer = np.ma.masked_where(colour["label_map"] != i, colour["label_map"])
            ax_colour.imshow(
                layer,
                cmap=matplotlib.colors.ListedColormap([CLASS_COLOURS[name]]),
                alpha=0.75,
                interpolation="nearest",
                zorder=2,
            )
        for label, indices, ring_colour in LIP_GROUPS[:2]:
            ring = points[indices]
            ax_colour.plot(ring[:, 0], ring[:, 1], "-", color=ring_colour, lw=1.4, zorder=4)
        verdict = "  ".join(f"{n} {colour['shares'][n]:.0%}" for n in colour["names"])
        ax_colour.set_title(
            f'ZOOM: what is inside FaceMesh\'s "upper lip" band? → {verdict}', fontsize=9
        )
    ax_colour.set_xlim(x0, x1)
    ax_colour.set_ylim(y1, y0)

    # 6. FaRL semantic parsing — an independent opinion on lip vs skin vs teeth
    parse_lines: list[str] = []
    if ax_parse is not None:
        from ulbt_face_parsing import CLASS_COLOURS as PARSE_COLOURS
        from ulbt_face_parsing import PRETTY, band_composition

        labels, names = parsing
        ax_parse.imshow(rgb)
        for name, hexcolour in PARSE_COLOURS.items():
            if name not in names:
                continue
            index = names.index(name)
            layer = np.ma.masked_where(labels != index, labels)
            ax_parse.imshow(
                layer,
                cmap=matplotlib.colors.ListedColormap([hexcolour]),
                alpha=0.5,
                interpolation="nearest",
                zorder=2,
            )
        ring = points[UPPER_LIP_OUTER]
        ax_parse.plot(ring[:, 0], ring[:, 1], "-", color="black", lw=2.2, zorder=4)
        ring = points[UPPER_LIP_INNER]
        ax_parse.plot(ring[:, 0], ring[:, 1], "--", color="black", lw=2.2, zorder=4)

        band = polygon_mask(points, UPPER_LIP_OUTER, UPPER_LIP_INNER, (width, height))
        composition = band_composition(labels, names, band)
        verdict = "  ".join(f"{PRETTY.get(n, n)} {v:.0%}" for n, v in list(composition.items())[:3])
        ax_parse.set_title(f"ZOOM: FaRL parsing — band contains → {verdict}", fontsize=9)
        ax_parse.set_xlim(x0, x1)
        ax_parse.set_ylim(y1, y0)

        parse_lines.append("")
        parse_lines.append("FaRL parsing inside FaceMesh's upper-lip band:")
        for name, share in composition.items():
            parse_lines.append(f"  {PRETTY.get(name, name):<22s} {share:6.1%}")
        parse_lines.append("FaRL class pixel counts (whole image):")
        for index, name in enumerate(names):
            count = int((labels == index).sum())
            if count and name in PARSE_COLOURS:
                parse_lines.append(f"  {PRETTY.get(name, name):<22s} {count}")

    for ax in [a for row in axes for a in row] + [ax_colour] + ([ax_parse] if ax_parse else []):
        ax.set_xticks([])
        ax.set_yticks([])

    # numbers panel, printed under the figure
    lines = [f"{k:42s} {v:+.4f}" for k, v in metrics(points).items()]
    if colour is not None:
        lines.append("")
        lines.append(f"upper-lip band pixels{'':21s} {colour['band_px']}")
        for name in colour["names"]:
            rgb_ref = colour["swatches"][name]
            lines.append(
                f"  band is {name:<10s} {colour['shares'][name]:6.1%}"
                f"   ref RGB {rgb_ref}  n={colour['ref_px'][name]}"
            )
    lines.extend(parse_lines)
    if blendshapes is not None:
        scores = {b.category_name: b.score for b in blendshapes}
        lines.append("")
        for name in BLENDSHAPES_OF_INTEREST:
            if name in scores:
                lines.append(f"{name:42s} {scores[name]:.4f}")
    fig.text(0.02, 0.30, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=7)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.35,
        help="ROI padding as a fraction of the lip bounding box (default 0.35; "
        "lower = tighter zoom)",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="add a FaRL semantic-parsing panel (needs: uv run --with pyfacer "
        "--with torch --with torchvision ...)",
    )
    args = parser.parse_args(argv)

    if not args.image.exists():
        raise SystemExit(f"no such file: {args.image}")

    out_path = args.output or Path.cwd() / f"{args.image.stem}_ulbt_facemesh.png"

    photo, teethmap = load_photo(args.image)
    rgb = np.ascontiguousarray(np.asarray(photo))
    landmarks, blendshapes = detect_landmarks(rgb)
    points = to_pixels(landmarks, photo.width, photo.height)
    box = lip_roi(points, photo.width, photo.height, args.margin)

    parsing = None
    if args.parse:
        sys.path.insert(0, str(Path(__file__).parent))
        from ulbt_face_parsing import parse_face

        parsing = parse_face(rgb)
        if parsing is None:
            print("WARNING: FaRL detected no face; skipping the parsing panel", file=sys.stderr)

    print(f"image      : {args.image}  ({photo.width}x{photo.height})")
    print(f"landmarks  : {len(landmarks)}")
    print(f"teeth matte: {'yes ' + str(teethmap.size) if teethmap is not None else 'no'}")
    print(f"lip ROI    : x {box[0]:.0f}..{box[2]:.0f}  y {box[1]:.0f}..{box[3]:.0f}")
    print()

    lines = build_figure(rgb, points, box, teethmap, blendshapes, parsing, args.image, out_path)
    print("\n".join(lines))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
