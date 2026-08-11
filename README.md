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

1. Choose **Draw A**, press at the intended centre, and drag to set the radius
   of the circular contact area.
2. Draw region B when the application switches to **Draw B**. Circles can be
   moved by dragging inside them and resized from the perimeter handle.
3. Choose `Highest` (nearest to the camera), `Lowest` (farthest from the
   camera), or `Flattest` independently for both regions. A teeth mask can be
   applied when it is available.

Single-key shortcuts select the active tool without `Alt`: `H` for highest,
`L` for lowest, `F` for flattest, and `P` for the original pixel tool. The same
commands are available from the **Tools** menu.

The best 5–20% of pixels form each candidate pool. The application selects
5–10 spatially distributed endpoint pairs and reports mean ± sample standard
deviation for both straight 3D distance and median-filtered surface distance.

The depth display can remain in raw grayscale or use locally stretched Viridis
colours. An optional contour mode outlines boundaries between median-smoothed
raw depth levels. These modes affect only visualization—the measurement engine
continues to use the original depth values.
