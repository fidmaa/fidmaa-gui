# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

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

```bash
# Run the test suite
uv run pytest
```

## Architecture

- **Entrypoint**: `fidmaa_gui.entrypoints:run` launches `app.main()` which creates QApplication with a single MainWindow
- **MainWindow** (`app.py`): QMainWindow — loads HEIC images via `portrait_analyser` library, displays photo at 480x640, renders depth chart at 255x640, handles all click-based measurement logic. Contains a bottom QDockWidget with zoomed depth map, tabbed photo/skin/teeth views, and reconstruction chart.
- **UI file**: `form.ui` loaded at runtime via `UILoaderMixin` in `utils.py` using a custom `QUiLoader` that registers `QClickableLabel`
- **QClickableLabel** (`QClickableLabel.py`): Custom QLabel widget emitting click/drag signals

## Key Dependencies

- `portrait-analyser`: Loads iOS portrait photos, extracts depth/skin/teeth maps, detects faces
- `fidmaa-simple-viewer`: Converts FIDMAA data to PyVista 3D surfaces
- `pyvistaqt`: 3D visualization via PyVista in Qt
- `pyheif-iplweb`: HEIC file format support

## Conventions

- Formatting and linting: ruff (`ruff-check --fix` + `ruff-format`, line length 100),
  wired up via `.pre-commit-config.yaml`. Run `pre-commit install` once.
- The application works with fixed display sizes: main image 480x640, zoom views 480x320
- Depth values are converted from 0-255 pixel range to centimeters using iPhone TrueDepth EXIF calibration data
