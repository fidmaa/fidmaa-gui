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
