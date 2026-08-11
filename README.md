# fidmaa

FIDMAA GUI is a desktop application for analyzing iPhone portrait photos (HEIC format) that contain TrueDepth camera data. It extracts depth maps to perform 3D measurements (distances, neck circumference estimation, angle calculations) on facial portraits. Built with PySide6 (Qt for Python).

## Build & Run

```bash
# Install dependencies
uv sync

# Run the application
uv run fidmaa_gui
# Or with a file argument:
uv run fidmaa_gui ~/path/to/photo.heic

# Build macOS app bundle
make all  # runs clean + pyinstaller + copy files
```

## Region-to-region measurements

The **Region measurement** dock provides a robust alternative to selecting two
individual depth pixels:

1. Choose **Draw A** and drag a rectangle over the first anatomical area.
2. Draw region B when the application switches to **Draw B**. Hold `Shift` to
   constrain a new region to a square.
3. Choose `Highest` (nearest to the camera), `Lowest` (farthest from the
   camera), or `Flattest` independently for both regions. A teeth mask can be
   applied when it is available.

Single-key shortcuts select the active tool without `Alt`: `H` for highest,
`L` for lowest, `F` for flattest, and `P` for the original pixel tool. The same
commands are available from the **Tools** menu.

The best 5–20% of pixels form each candidate pool. The application selects
5–10 spatially distributed endpoint pairs and reports mean ± sample standard
deviation for both straight 3D distance and median-filtered surface distance.
Rectangles can be moved or resized by dragging a corner.
