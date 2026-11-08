#all:
#	@echo "Available targets: clean macos"

all: clean macos

UNAME_S := $(shell uname -s)

.PHONY: prepare-developer-machine prepare-developer-machine-linux \
	prepare-developer-machine-macos dev-setup test run

# Installs the system packages uv cannot (apt on Debian/Ubuntu, Homebrew on
# macOS), then the Python environment and the pre-commit hooks.
prepare-developer-machine:
ifeq ($(UNAME_S),Darwin)
	$(MAKE) prepare-developer-machine-macos
else ifeq ($(UNAME_S),Linux)
	$(MAKE) prepare-developer-machine-linux
else
	$(error Unsupported OS: $(UNAME_S))
endif
	$(MAKE) dev-setup

prepare-developer-machine-linux:
	./scripts/prepare-dev-linux.sh

prepare-developer-machine-macos:
	./scripts/prepare-dev-macos.sh

dev-setup:
	uv sync
	uv run pre-commit install

test:
	uv run pytest

run:
	uv run fidmaa_gui

clean:
	find . -name __pycache__ -type d -print0 | xargs -0 rm -rf
	find . -name \*~ -print0 | xargs -0 rm -f
	find . -name \*pyc -print0 | xargs -0 rm -f
	find . -name \*\\.log -print0 | xargs -0 rm -f
	rm -rf .tox build dist

macos: macos-build macos-copy-files

macos-build:
	uv run pyinstaller fidmaa_gui.spec

# Bare `python` resolves to whatever is first on PATH, not this project's
# interpreter, so ask uv where the environment actually is.
VENV_PATH=$(shell uv run python -c "import sys; print(sys.prefix)")
# PyInstaller 6 collects packages into Contents/Frameworks; Contents/MacOS
# holds only the executable.
BUNDLE_LIBS=dist/fidmaa_gui.app/Contents/Frameworks

macos-copy-files:
	mkdir -p $(BUNDLE_LIBS)/cv2/data/
	cp -R $(VENV_PATH)/lib/python3*/site-packages/cv2/data/ $(BUNDLE_LIBS)/cv2/data/
	mkdir -p $(BUNDLE_LIBS)/pyheif/data/
	cp -R $(VENV_PATH)/lib/python3*/site-packages/pyheif/data/ $(BUNDLE_LIBS)/pyheif/data/

local-dev:
	uv sync
	uv pip install -e '../portrait-analyser[pose]'
	@echo ""
	@echo "Local portrait-analyser installed. Use 'uv run --no-sync' to avoid reverting it."
	@echo "Example: uv run --no-sync fidmaa_gui"

zip-app:
	cd dist && zip -r fidmaa_gui.app.zip fidmaa_gui.app
