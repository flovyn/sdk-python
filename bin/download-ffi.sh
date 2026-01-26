#!/bin/bash
# Download FFI native libraries and Python bindings from sdk-rust releases
#
# Usage:
#   ./bin/download-ffi.sh <version> [platforms...]
#
# Examples:
#   ./bin/download-ffi.sh v0.1.5                           # Download all platforms
#   ./bin/download-ffi.sh v0.1.5 linux-x86_64              # Download only linux-x86_64
#   ./bin/download-ffi.sh v0.1.5 linux-x86_64 linux-aarch64

set -euo pipefail

VERSION="${1:-}"
shift || true

if [[ -z "$VERSION" ]]; then
    echo "Usage: $0 <version> [platforms...]"
    echo "Example: $0 v0.1.5 linux-x86_64"
    exit 1
fi

# Default platforms if none specified
if [[ $# -eq 0 ]]; then
    PLATFORMS=("linux-x86_64" "linux-aarch64" "macos-x86_64" "macos-aarch64")
else
    PLATFORMS=("$@")
fi

BASE_URL="https://github.com/flovyn/sdk-rust/releases/download/${VERSION}"
NATIVES_DIR="flovyn/_native"

echo "Downloading FFI version ${VERSION}..."
echo "Platforms: ${PLATFORMS[*]}"

# Create directories
mkdir -p "${NATIVES_DIR}"
mkdir -p tmp

# Download and extract native libraries
for platform in "${PLATFORMS[@]}"; do
    echo "Downloading native library for ${platform}..."
    curl -fsSL "${BASE_URL}/libflovyn_worker_ffi-${platform}.tar.gz" -o "tmp/libflovyn_worker_ffi-${platform}.tar.gz"
    tar -xzf "tmp/libflovyn_worker_ffi-${platform}.tar.gz" -C "${NATIVES_DIR}/"
done

# Create symlink for the current platform's library in the expected location
# UniFFI bindings expect the library directly in _native/, not in a subdirectory
CURRENT_PLATFORM=""
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  CURRENT_PLATFORM="linux-x86_64" ;;
    Linux-aarch64) CURRENT_PLATFORM="linux-aarch64" ;;
    Darwin-x86_64) CURRENT_PLATFORM="macos-x86_64" ;;
    Darwin-arm64)  CURRENT_PLATFORM="macos-aarch64" ;;
esac

if [[ -n "$CURRENT_PLATFORM" && -d "${NATIVES_DIR}/${CURRENT_PLATFORM}" ]]; then
    echo "Creating symlinks for current platform (${CURRENT_PLATFORM})..."
    for lib in "${NATIVES_DIR}/${CURRENT_PLATFORM}"/*; do
        if [[ -f "$lib" ]]; then
            libname=$(basename "$lib")
            ln -sf "${CURRENT_PLATFORM}/${libname}" "${NATIVES_DIR}/${libname}"
            echo "  Linked ${libname}"
        fi
    done
fi

# Download and extract Python bindings
echo "Downloading Python bindings..."
curl -fsSL "${BASE_URL}/flovyn-worker-ffi-bindings.tar.gz" -o "tmp/flovyn-worker-ffi-bindings.tar.gz"
mkdir -p tmp/bindings
tar -xzf tmp/flovyn-worker-ffi-bindings.tar.gz -C ./tmp/bindings
cp tmp/bindings/python/flovyn_worker_ffi.py "${NATIVES_DIR}/"

# Cleanup
rm -rf tmp

echo "Done."
