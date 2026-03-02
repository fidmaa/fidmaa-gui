#all:
#	@echo "Available targets: clean macos"

all: clean macos

clean:
	find . -name __pycache__ -type d -print0 | xargs -0 rm -rf
	find . -name \*~ -print0 | xargs -0 rm -f
	find . -name \*pyc -print0 | xargs -0 rm -f
	find . -name \*\\.log -print0 | xargs -0 rm -f
	rm -rf .tox build dist

macos: macos-build macos-copy-files

macos-build:
	pyinstaller fidmaa_gui.spec

VENV_PATH=$(shell python -c "import sys; print(sys.prefix)")

macos-copy-files:
	cp -R $(VENV_PATH)/lib/python3*/site-packages/cv2/data/ ./dist/fidmaa_gui.app/Contents/MacOS/cv2/data/
	mkdir -p dist/fidmaa_gui.app/Contents/MacOS/pyheif/data/
	cp -R $(VENV_PATH)/lib/python3*/site-packages/pyheif/data/ ./dist/fidmaa_gui.app/Contents/MacOS/pyheif/data/

local-dev:
	uv sync
	uv pip install -e '../portrait-analyser[pose]'
	@echo ""
	@echo "Local portrait-analyser installed. Use 'uv run --no-sync' to avoid reverting it."
	@echo "Example: uv run --no-sync fidmaa_gui"

zip-app:
	cd dist && zip -r fidmaa_gui.app.zip fidmaa_gui.app
