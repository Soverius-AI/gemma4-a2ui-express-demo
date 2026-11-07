"""Hashing and dependency-free safetensors reading and writing.

Tensors are memory-mapped rather than loaded, so multi-gigabyte checkpoints are
processed with bounded memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

# safetensors dtype -> NumPy dtype. BF16 is read as raw bits (see bf16_to_fp16).
DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "BF16": "<u2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U64": "<u8",
    "U32": "<u4",
    "U16": "<u2",
    "U8": "u1",
    "BOOL": "?",
}
MAX_HEADER_BYTES = 100 * 1024 * 1024
HASH_BLOCK_BYTES = 8 * 1024 * 1024

Header = dict[str, Any]


class IdentityError(ValueError):
    """A file does not have its pinned digest."""


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def sha256(path: Path) -> str:
    return file_digest(path, "sha256")


def verify_sha256(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise IdentityError(f"Identity mismatch for {path}: expected {expected}, got {actual}")
    return actual


def dtype_size(dtype: str) -> int:
    return np.dtype(DTYPES[dtype]).itemsize


def tensor_nbytes(meta: dict[str, Any]) -> int:
    return math.prod(meta["shape"]) * dtype_size(meta["dtype"])


def read_header(path: Path) -> tuple[Header, int]:
    """Return a validated header and the absolute offset of the tensor data."""
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Missing safetensors header: {path}")
        (length,) = struct.unpack("<Q", prefix)
        if length > MAX_HEADER_BYTES or length % 8:
            raise ValueError(f"Invalid safetensors header length: {path}")
        header: Header = json.loads(stream.read(length))
    data_bytes = path.stat().st_size - 8 - length
    _validate_layout(header, data_bytes)
    return header, 8 + length


def _validate_layout(header: Header, data_bytes: int) -> None:
    """Require every byte of the data section to belong to exactly one tensor."""
    spans = []
    for name, meta in tensors(header):
        if meta["dtype"] not in DTYPES:
            raise ValueError(f"Unsupported dtype {meta['dtype']} for {name}")
        start, end = meta["data_offsets"]
        if start < 0 or end < start or end > data_bytes or end - start != tensor_nbytes(meta):
            raise ValueError(f"Invalid tensor bounds for {name}")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("Noncontiguous/overlapping tensor data")
        cursor = end
    if cursor != data_bytes:
        raise ValueError("Unaccounted tensor data")


def tensors(header: Header):
    """Yield (name, metadata) for each tensor, skipping __metadata__."""
    return ((name, meta) for name, meta in header.items() if name != "__metadata__")


def write_header(stream: BinaryIO, header: Header) -> int:
    """Write an 8-byte aligned header; return the offset where tensor data begins."""
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    stream.write(struct.pack("<Q", len(encoded)))
    stream.write(encoded)
    return 8 + len(encoded)


def load_tensor(path: Path, data_offset: int, meta: dict[str, Any]) -> np.memmap:
    return np.memmap(
        path,
        mode="r",
        dtype=DTYPES[meta["dtype"]],
        offset=data_offset + meta["data_offsets"][0],
        shape=tuple(meta["shape"]),
    )


def bf16_to_fp16(bits: np.ndarray) -> np.ndarray:
    """Widen BF16 bit patterns to F32, then round to F16 (ties to even)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float16)


def write_finite(stream: BinaryIO, array: np.ndarray) -> None:
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("Nonfinite/overflow in output tensor")
    stream.write(array.tobytes(order="C"))
