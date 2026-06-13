"""Small ctypes binding for NanoQuant's C ABI.

This intentionally binds the stable C surface, not C++ internals. Build the
project first, then point NANOQUANT_LIBRARY at libnanoquant_c when needed.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path


class _TensorInfo(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("rows", ctypes.c_uint64),
        ("cols", ctypes.c_uint64),
        ("data_offset", ctypes.c_uint64),
        ("data_bytes", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class TensorInfo:
    version: int
    rows: int
    cols: int
    data_offset: int
    data_bytes: int


def _default_library_path() -> Path:
    suffix = "dylib" if os.uname().sysname == "Darwin" else "so"
    return Path(__file__).resolve().parents[2] / "build" / f"libnanoquant_c.{suffix}"


def load_library(path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    library_path = Path(path or os.environ.get("NANOQUANT_LIBRARY", _default_library_path()))
    lib = ctypes.CDLL(str(library_path))
    lib.nq_version.restype = ctypes.c_char_p
    lib.nq_save_demo_tensor.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
    lib.nq_save_demo_tensor.restype = ctypes.c_int
    lib.nq_inspect_tensor.argtypes = [ctypes.c_char_p, ctypes.POINTER(_TensorInfo)]
    lib.nq_inspect_tensor.restype = ctypes.c_int
    lib.nq_last_error.restype = ctypes.c_char_p
    return lib


class NanoQuant:
    def __init__(self, library_path: str | os.PathLike[str] | None = None) -> None:
        self._lib = load_library(library_path)

    def version(self) -> str:
        return self._lib.nq_version().decode()

    def save_demo_tensor(self, path: str | os.PathLike[str], rows: int, cols: int, seed: int = 42) -> None:
        result = self._lib.nq_save_demo_tensor(os.fsencode(path), rows, cols, seed)
        if result != 0:
            raise RuntimeError(self._lib.nq_last_error().decode())

    def inspect_tensor(self, path: str | os.PathLike[str]) -> TensorInfo:
        info = _TensorInfo()
        result = self._lib.nq_inspect_tensor(os.fsencode(path), ctypes.byref(info))
        if result != 0:
            raise RuntimeError(self._lib.nq_last_error().decode())
        return TensorInfo(info.version, info.rows, info.cols, info.data_offset, info.data_bytes)
