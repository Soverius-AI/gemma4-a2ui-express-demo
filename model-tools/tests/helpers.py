from __future__ import annotations

from pathlib import Path

import numpy as np

from a2ui_model_tools.checkpoint_io import write_header


def write_checkpoint(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """Write a safetensors file from {name: (dtype, values)}."""
    header, cursor = {}, 0
    for name, (dtype, values) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(values.shape),
            "data_offsets": [cursor, cursor + values.nbytes],
        }
        cursor += values.nbytes
    with path.open("wb") as stream:
        write_header(stream, header)
        for _, values in tensors.values():
            stream.write(values.tobytes())


def to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Exact BF16 bit patterns of F32 values that BF16 represents exactly."""
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
