#!/usr/bin/env bash
# Prepare a macOS machine for fidmaa-gui development (Homebrew required).
#
# pyheif-iplweb's compiled extension dlopen()s libheif at import time from the
# MacPorts path (/opt/local/lib) rather than wherever Homebrew puts it, so we
# install libheif with brew and symlink it into place. This mirrors what
# .github/workflows/build.yml does on CI.
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
    echo "error: Homebrew is required -- install it from https://brew.sh first." >&2
    exit 1
fi

brew install libheif uv

LIBHEIF="$(brew --prefix libheif)/lib/libheif.1.dylib"
TARGET=/opt/local/lib/libheif.1.dylib

if [ -e "$TARGET" ] && [ ! -L "$TARGET" ]; then
    # A real file there means MacPorts' own libheif -- leave it alone.
    echo "$TARGET exists (MacPorts?), leaving it untouched."
elif [ "$(readlink "$TARGET" 2>/dev/null || true)" != "$LIBHEIF" ]; then
    echo "Linking $LIBHEIF -> $TARGET (needs sudo)"
    sudo mkdir -p /opt/local/lib
    sudo ln -sf "$LIBHEIF" "$TARGET"
fi
