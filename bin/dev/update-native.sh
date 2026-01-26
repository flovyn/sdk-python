#!/bin/bash
#
# Updates the native FFI library and Python bindings from sdk-rust.
#
# Usage:
#   ./bin/dev/update-native.sh                    # Build from local sdk-rust for current platform
#   ./bin/dev/update-native.sh --download [VER]   # Download from GitHub release (default: latest)
#   ./bin/dev/update-native.sh --bindings         # Only regenerate Python bindings from local build
#
# Prerequisites:
#   For local build:
#     - Rust toolchain installed
#     - sdk-rust repository available at ../sdk-rust (or set SDK_RUST_PATH)
#
#   For download:
#     - GitHub CLI (gh) installed and authenticated
#
# Examples:
#   ./bin/dev/update-native.sh                    # Build from local sdk-rust
#   ./bin/dev/update-native.sh --download         # Download latest release
#   ./bin/dev/update-native.sh --download v0.1.0  # Download specific version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_PYTHON_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SDK_RUST_PATH="${SDK_RUST_PATH:-$SDK_PYTHON_ROOT/../sdk-rust}"
SDK_RUST_REPO="${SDK_RUST_REPO:-flovyn/sdk-rust}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect current platform
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)

    case "$os" in
        darwin)
            case "$arch" in
                arm64|aarch64) echo "macos-aarch64" ;;
                x86_64) echo "macos-x86_64" ;;
                *) log_error "Unsupported macOS architecture: $arch"; exit 1 ;;
            esac
            ;;
        linux)
            case "$arch" in
                aarch64) echo "linux-aarch64" ;;
                x86_64) echo "linux-x86_64" ;;
                *) log_error "Unsupported Linux architecture: $arch"; exit 1 ;;
            esac
            ;;
        *)
            log_error "Unsupported OS: $os"
            exit 1
            ;;
    esac
}

# Get library filename for platform
get_lib_name() {
    case "$1" in
        macos-*) echo "libflovyn_worker_ffi.dylib" ;;
        linux-*) echo "libflovyn_worker_ffi.so" ;;
        windows-*) echo "flovyn_worker_ffi.dll" ;;
        *) log_error "Unknown platform: $1"; exit 1 ;;
    esac
}

# Check if sdk-rust exists (for local builds)
check_sdk_rust() {
    if [[ ! -d "$SDK_RUST_PATH" ]]; then
        log_error "sdk-rust not found at: $SDK_RUST_PATH"
        log_error "Set SDK_RUST_PATH environment variable or use --download to fetch from GitHub releases"
        exit 1
    fi
    SDK_RUST_PATH="$(cd "$SDK_RUST_PATH" && pwd)"
    log_info "Using sdk-rust at: $SDK_RUST_PATH"
}

# Build for current platform (no cross-compilation needed)
build_current_platform() {
    check_sdk_rust

    local platform=$(detect_platform)
    local lib_name=$(get_lib_name "$platform")

    log_info "Building flovyn-worker-ffi for current platform ($platform)..."

    (cd "$SDK_RUST_PATH" && cargo build -p flovyn-worker-ffi --release)

    local src="$SDK_RUST_PATH/target/release/$lib_name"
    local dest_dir="$SDK_PYTHON_ROOT/flovyn/_native"

    mkdir -p "$dest_dir"
    cp "$src" "$dest_dir/"

    log_info "Copied $lib_name to $dest_dir"
}

# Generate Python bindings from local build
generate_bindings() {
    check_sdk_rust

    local lib_name
    local platform=$(detect_platform)
    lib_name=$(get_lib_name "$platform")

    local lib_path="$SDK_RUST_PATH/target/release/$lib_name"

    # Build if library doesn't exist
    if [[ ! -f "$lib_path" ]]; then
        log_info "Native library not found, building first..."
        (cd "$SDK_RUST_PATH" && cargo build -p flovyn-worker-ffi --release)
    fi

    log_info "Generating Python bindings..."

    local bindings_dir="$SDK_PYTHON_ROOT/flovyn/_native"
    mkdir -p "$bindings_dir"

    (cd "$SDK_RUST_PATH" && cargo run -p flovyn-worker-ffi --bin uniffi-bindgen -- \
        generate --library "$lib_path" \
        --language python \
        --out-dir "$bindings_dir")

    log_info "Generated bindings at: $bindings_dir/flovyn_worker_ffi.py"
}

# Download from GitHub releases
download_from_release() {
    local version="${1:-latest}"

    # Check if gh is installed
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) is not installed"
        log_error "Install it from: https://cli.github.com/"
        exit 1
    fi

    # Check if gh is authenticated
    if ! gh auth status &> /dev/null; then
        log_error "GitHub CLI is not authenticated"
        log_error "Run: gh auth login"
        exit 1
    fi

    # Detect current platform
    local current_platform=$(detect_platform)
    log_info "Detected platform: $current_platform"

    log_info "Downloading FFI artifacts from $SDK_RUST_REPO release: $version"

    local tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" EXIT

    # Download only the current platform's archive and bindings
    local archive_pattern="libflovyn_worker_ffi-${current_platform}.tar.gz"

    if [[ "$version" == "latest" ]]; then
        log_info "Fetching latest release..."
        gh release download \
            --repo "$SDK_RUST_REPO" \
            --pattern "$archive_pattern" \
            --pattern "flovyn-worker-ffi-bindings.tar.gz" \
            --dir "$tmp_dir" \
            2>&1 || {
                log_error "Failed to download release. Make sure releases exist in $SDK_RUST_REPO"
                exit 1
            }
    else
        log_info "Fetching release $version..."
        gh release download "$version" \
            --repo "$SDK_RUST_REPO" \
            --pattern "$archive_pattern" \
            --pattern "flovyn-worker-ffi-bindings.tar.gz" \
            --dir "$tmp_dir" \
            2>&1 || {
                log_error "Failed to download release $version from $SDK_RUST_REPO"
                exit 1
            }
    fi

    # Prepare directories
    local dest_dir="$SDK_PYTHON_ROOT/flovyn/_native"
    mkdir -p "$dest_dir"

    # Extract native library
    log_info "Extracting native library for $current_platform..."
    local archive="$tmp_dir/$archive_pattern"
    if [[ -f "$archive" ]]; then
        # Extract to temp directory first, then copy to _native
        mkdir -p "$tmp_dir/natives"
        tar -xzf "$archive" -C "$tmp_dir/natives"
        local lib_name=$(get_lib_name "$current_platform")
        cp "$tmp_dir/natives/$current_platform/$lib_name" "$dest_dir/"
        log_info "  Extracted $lib_name to $dest_dir"
    else
        log_error "Archive not found: $archive_pattern"
        exit 1
    fi

    # Extract bindings
    local bindings_archive="$tmp_dir/flovyn-worker-ffi-bindings.tar.gz"
    if [[ -f "$bindings_archive" ]]; then
        log_info "Extracting Python bindings..."
        mkdir -p "$tmp_dir/bindings"
        tar -xzf "$bindings_archive" -C "$tmp_dir/bindings"

        # Copy Python bindings
        if [[ -f "$tmp_dir/bindings/python/flovyn_worker_ffi.py" ]]; then
            cp "$tmp_dir/bindings/python/flovyn_worker_ffi.py" "$dest_dir/"
            log_info "  Copied Python bindings to $dest_dir"
        elif [[ -f "$tmp_dir/bindings/flovyn_worker_ffi.py" ]]; then
            cp "$tmp_dir/bindings/flovyn_worker_ffi.py" "$dest_dir/"
            log_info "  Copied Python bindings to $dest_dir"
        else
            log_warn "Python bindings not found in archive, generating locally..."
            generate_bindings
        fi
    else
        log_warn "Bindings archive not found, generating locally..."
        generate_bindings
    fi

    log_info "Download complete! (only $current_platform)"
}

# Show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Update native FFI libraries and Python bindings from sdk-rust.

Options:
  (none)              Build from local sdk-rust for current platform
  --download [VER]    Download from GitHub release for current platform (default: latest)
  --bindings          Only regenerate Python bindings from local build
  --help              Show this help message

Environment Variables:
  SDK_RUST_PATH   Path to local sdk-rust repository (default: ../sdk-rust)
  SDK_RUST_REPO   GitHub repository for releases (default: flovyn/sdk-rust)

Notes:
  - --download auto-detects your OS and downloads only the matching native library
  - Available release platforms: linux-x86_64, linux-aarch64, macos-x86_64, macos-aarch64, windows-x86_64

Examples:
  $0                          # Build from local sdk-rust
  $0 --download               # Download latest release (current platform only)
  $0 --download v0.1.0        # Download specific version
  $0 --bindings               # Regenerate bindings only
EOF
}

# Main
main() {
    local mode="${1:-local}"

    case "$mode" in
        --download|-d)
            local version="${2:-latest}"
            download_from_release "$version"
            ;;
        --bindings|-b)
            generate_bindings
            ;;
        --help|-h)
            show_help
            ;;
        local|*)
            build_current_platform
            generate_bindings
            ;;
    esac

    log_info "Done!"
}

main "$@"
