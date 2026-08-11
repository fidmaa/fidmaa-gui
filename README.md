# fidmaa-gui

[![Build](https://github.com/fidmaa/fidmaa-gui/actions/workflows/build.yml/badge.svg)](https://github.com/fidmaa/fidmaa-gui/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FIDMAA GUI is a desktop application for analyzing iPhone portrait photos (HEIC
format) that contain TrueDepth camera data. It extracts the depth map to perform
3D measurements — distances, neck circumference estimation, angle calculations —
on facial portraits. Built with PySide6 (Qt for Python).

## Requirements

- macOS (the app bundle and CI target macOS; other platforms are untested)
- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- An iPhone portrait photo in HEIC format, with the depth map preserved. Photos
  exported through most messaging apps have it stripped.

## Install & run

```bash
# Install dependencies
uv sync

# Run the application
uv run fidmaa_gui
# Or with a file argument:
uv run fidmaa_gui ~/path/to/photo.heic
```

## Build a macOS app bundle

```bash
make all  # runs clean + pyinstaller + copy files
```

The bundle lands in `dist/`, and `make zip-app` packs it into
`dist/fidmaa_gui.app.zip`.

This is a local-only step on purpose — the bundle is roughly 850 MB, so CI
neither builds it nor stores it. Note it is unsigned and un-notarized, so
Gatekeeper blocks it on any machine other than the one that built it unless
you clear the quarantine flag:

```bash
xattr -dr com.apple.quarantine dist/fidmaa_gui.app
```

## Region-to-region measurements

The **Region measurement** dock provides a robust alternative to selecting two
individual depth pixels:

1. Set the shared **Patch radius** (20 px by default), enable **Show measurement
   controls**, and click once to place region A. Click once more to place
   region B.
2. Existing circles are selected directly under the pointer: drag inside
   either patch to move it. Once both patches exist, dragging outside them
   cannot accidentally replace either region; use **Clear regions** to start
   over.
3. Choose `Area (uniform)`, `Highest`, `Lowest`, `Local peak`, `Local valley`,
   or `Flattest` independently for both regions. Highest and lowest use the
   absolute depth extrema in the patch. Local peak and valley remove the
   patch's dominant 3D tilt before ranking points, so mild patient rotation
   does not turn one side of the patch into the automatic winner. A teeth mask
   can be applied when it is available.

`Area (uniform)` is intended for broad, asymmetric, or divided anatomical
surfaces without one unambiguous landmark, such as a two-lobed mentum. It
ignores depth ranking, places one sample at the centre and distributes the
remaining samples uniformly across each disk. When both regions use Area, the
same relative disk locations are paired directly. Candidate-pool percentage is
therefore ignored for Area.

Single-key shortcuts select the active tool without `Alt`: `H` for highest,
`L` for lowest, `F` for flattest, and `P` for the original pixel tool. Local
peak and valley remain available in both selectors and the **Tools** menu.

Both circles, selected candidate pixels, and measurement vectors are repeated
in the zoomed photo and map views. Their compact value readouts occupy no more
than two lines at the bottom of each zoom view.

For ranked modes, the best 5–20% of pixels form each candidate pool (5% by
default). The application selects 5–30 spatially distributed endpoint pairs
and reports mean ± sample standard deviation for both straight 3D distance and
median-filtered surface distance.

The depth display can remain in raw grayscale or use locally stretched Viridis
colours. An optional contour mode outlines boundaries between median-smoothed
raw depth levels. **Contour step** controls the density from every raw level to
every eighth level and defaults to every fourth level. Contours are extracted
from a 3×3 median-filtered map, recomputed after zooming, and stay one display
pixel wide instead of scaling up with the source bitmap. These modes affect
only visualization—the measurement engine continues to use the original depth
values.

For a neutral-neck portrait, **Show neck: skin matte → stable depth** displays
the automatic neck-edge correction. Yellow rings mark the raw skin-matte
silhouette, cyan points mark the first stable depth samples found while walking
inward, and the green curve is the depth arc used for the circumference
measurement. The same overlay is rendered on the photo and zoomed maps.
Automatic row detection searches only in a FaceMesh-scaled anatomical band
below the lowest facial landmark and selects the first stable local minimum of
the median-smoothed neck-width profile, avoiding later shoulder/collar minima.
The neck output labels the sampled green curve as a 3D surface polyline and
reports its direct Euclidean endpoint chord separately. It also includes a
median-filtered straight-row surface vector at several sampling intervals.

## Related projects

- [portrait-analyser](https://github.com/fidmaa/portrait-analyser) — loads iOS
  portrait photos, extracts depth/skin/teeth maps, detects faces and landmarks
- [fidmaa-simple-viewer](https://github.com/fidmaa/fidmaa-simple-viewer) —
  renders FIDMAA data as a PyVista 3D surface

## Development

```bash
uv run pytest          # run the test suite
pre-commit install     # enable ruff + hygiene hooks on commit
```

## License

MIT — see [LICENSE](LICENSE).
