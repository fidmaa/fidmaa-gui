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

The bundle lands in `dist/`.

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
