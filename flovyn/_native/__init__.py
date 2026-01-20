"""Native FFI bindings for Flovyn worker."""

from flovyn._native.loader import get_native_module

__all__ = ["get_native_module"]
