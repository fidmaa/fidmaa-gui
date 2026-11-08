#!/usr/bin/env bash
# Prepare a Debian/Ubuntu machine for fidmaa-gui development.
#
# uv installs every Python dependency (PySide6, VTK, OpenCV, pyheif, ...) as
# manylinux wheels, but those wheels still dlopen() a handful of system
# libraries -- mostly the X11/xcb, OpenGL/EGL and font stack Qt needs to open a
# window. uv cannot install those, so this script does it via apt.
#
# HEIC: pyheif >= 0.8.0 ships a wheel with its own libheif (+libde265 HEVC
# decoder), so libheif-dev is deliberately NOT needed here.
#
# Tested on Ubuntu 26.04. Should work on 22.04+ and Debian 12+.
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
    echo "error: this script supports apt-based distributions (Debian/Ubuntu) only." >&2
    echo "On other distributions install the equivalents of the packages listed in" >&2
    echo "$0, then uv (https://docs.astral.sh/uv/)." >&2
    exit 1
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

# Ubuntu 24.04 renamed libglib2.0-0 to libglib2.0-0t64 (64-bit time_t
# transition); pick whichever this release has.
pick() {
    for candidate in "$@"; do
        if LC_ALL=C apt-cache policy "$candidate" 2>/dev/null | grep -q 'Candidate: [^(]'; then
            echo "$candidate"
            return
        fi
    done
    echo "error: none of these packages is available: $*" >&2
    exit 1
}

$SUDO apt-get update

GLIB=$(pick libglib2.0-0t64 libglib2.0-0)

PACKAGES=(
    # tooling
    curl ca-certificates git
    # OpenGL / EGL -- Qt, VTK (pyvista), OpenCV and mediapipe all need it
    libgl1 libegl1
    "$GLIB"
    # Qt 6 xcb platform plugin (see https://doc.qt.io/qt-6/linux-requirements.html)
    libfontconfig1 libdbus-1-3 libxkbcommon0 libxkbcommon-x11-0
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0
    libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxcb-xfixes0
    libxrender1 libxi6 libsm6 libice6
    # Qt wayland platform plugin
    libwayland-client0 libwayland-cursor0 libwayland-egl1
)

$SUDO apt-get install -y --no-install-recommends "${PACKAGES[@]}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo
    echo "uv was installed into ~/.local/bin -- open a new shell (or run"
    echo "'source ~/.local/bin/env') before running 'make' again."
fi
