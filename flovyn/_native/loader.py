"""Native library loader for Flovyn FFI bindings."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

_native_module: ModuleType | None = None


def _get_library_name() -> str:
    """Get the native library filename for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize machine architecture names
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")

    # Map to library filename
    if system == "linux":
        return "libflovyn_worker_ffi.so"
    elif system == "darwin":
        return "libflovyn_worker_ffi.dylib"
    elif system == "windows":
        return "flovyn_worker_ffi.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}/{arch}")


def _get_platform_dir() -> str:
    """Get the platform-specific directory name."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize machine architecture names
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        arch = machine

    return f"{system}-{arch}"


def _find_native_library() -> Path:
    """Find the native library in package resources or development locations."""
    # Check in the _native directory (installed package)
    native_dir = Path(__file__).parent
    lib_name = _get_library_name()
    platform_dir = _get_platform_dir()

    # Check standard locations
    candidates = [
        native_dir / lib_name,
        native_dir / "lib" / lib_name,
        # Platform-specific subdirectory (from download script)
        native_dir / platform_dir / lib_name,
    ]

    # Check development locations (relative to sdk-python)
    sdk_python_root = native_dir.parent.parent
    sdk_rust_root = sdk_python_root.parent / "sdk-rust"

    # Check Rust target directories
    for profile in ["release", "debug"]:
        candidates.extend(
            [
                sdk_rust_root / "target" / profile / lib_name,
                sdk_python_root.parent / "target" / profile / lib_name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        f"Could not find native library '{lib_name}'. "
        f"Searched locations: {[str(c) for c in candidates]}"
    )


def _load_native_module() -> ModuleType:
    """Load the UniFFI-generated Python module."""
    # First, try to import the pre-installed module
    try:
        from flovyn._native import flovyn_worker_ffi

        return flovyn_worker_ffi
    except ImportError:
        pass

    # Find the native library path (validates it exists)
    _find_native_library()

    # The UniFFI-generated Python file should be alongside this loader
    bindings_path = Path(__file__).parent / "flovyn_worker_ffi.py"

    if not bindings_path.exists():
        raise RuntimeError(
            f"UniFFI Python bindings not found at {bindings_path}. "
            "Run 'cargo run -p flovyn-worker-ffi --bin uniffi-bindgen -- "
            "generate --library <lib_path> --language python --out-dir flovyn/_native'"
        )

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("flovyn_worker_ffi", bindings_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {bindings_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["flovyn._native.flovyn_worker_ffi"] = module
    spec.loader.exec_module(module)

    return module


def get_native_module() -> Any:
    """Get the native FFI module, loading it if necessary.

    Returns:
        The UniFFI-generated Python module with FFI bindings.

    Raises:
        RuntimeError: If the native library or bindings cannot be found.
    """
    global _native_module
    if _native_module is None:
        _native_module = _load_native_module()
    return _native_module


def is_native_available() -> bool:
    """Check if the native library is available.

    Returns:
        True if the native library can be loaded, False otherwise.
    """
    try:
        get_native_module()
        return True
    except RuntimeError:
        return False
