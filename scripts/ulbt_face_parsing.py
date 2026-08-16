"""FaRL face parsing for the ULBT probe — independent second opinion.

FaceMesh fits a fixed topology and has no idea whether it is looking at lip,
skin or teeth. A semantic segmentation model does classify each pixel, so it
can be asked directly: is FaceMesh's "upper lip" band actually upper lip?

Uses `pyfacer` (FaRL, trained on LaPa) which has explicit `upper lip`,
`lower lip` and `inner mouth` classes. Not a project dependency - run the
probe under:

    uv run --with pyfacer --with torch --with torchvision \\
        python scripts/ulbt_facemesh_probe.py IMAGE --parse
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def parse_face(rgb: np.ndarray, device: str = "cpu", max_dim: int = 1600):
    """Segment the face semantically.

    Returns (label_map, label_names) where label_map is (H, W) uint8 of class
    indices at the original resolution, or None if no face was detected.
    """
    import facer
    import torch

    height, width = rgb.shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale < 1.0:
        small = np.asarray(
            Image.fromarray(rgb).resize((int(width * scale), int(height * scale)), Image.BILINEAR)
        )
    else:
        small = rgb

    image = facer.hwc2bchw(torch.from_numpy(np.ascontiguousarray(small))).to(device=device)

    detector = facer.face_detector("retinaface/mobilenet", device=device)
    with torch.inference_mode():
        faces = detector(image)
    if len(faces.get("rects", [])) == 0:
        return None

    parser = facer.face_parser("farl/lapa/448", device=device)
    with torch.inference_mode():
        faces = parser(image, faces)

    logits = faces["seg"]["logits"]
    names = list(faces["seg"]["label_names"])
    labels = logits[0].argmax(dim=0).cpu().numpy().astype(np.uint8)

    if scale < 1.0:
        labels = np.asarray(Image.fromarray(labels).resize((width, height), Image.NEAREST))
    return labels, names


# LaPa class names, as returned by farl/lapa/448.
PRETTY = {
    "face": "skin",
    "ulip": "upper lip",
    "llip": "lower lip",
    "imouth": "inner mouth (teeth)",
    "nose": "nose",
}

# Classes worth painting; anything else (background, hair, brows, eyes) is left
# alone so the overlay stays readable in the mouth region.
CLASS_COLOURS = {
    "face": "#00ff88",
    "ulip": "#e6194b",
    "llip": "#4363d8",
    "imouth": "#ffe119",
}


def band_composition(labels: np.ndarray, names: list[str], band: np.ndarray) -> dict[str, float]:
    """What does the parser call the pixels inside a given mask?"""
    inside = labels[band]
    if inside.size == 0:
        return {}
    counts = np.bincount(inside, minlength=len(names))
    shares = {names[i]: float(counts[i] / inside.size) for i in range(len(names))}
    return {k: v for k, v in sorted(shares.items(), key=lambda kv: -kv[1]) if v > 0.001}
