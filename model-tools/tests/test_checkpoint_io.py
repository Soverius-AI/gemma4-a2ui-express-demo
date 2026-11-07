import io
import json

import numpy as np
import pytest
from safetensors.numpy import load_file

from a2ui_model_tools.checkpoint_io import (
    IdentityError,
    bf16_to_fp16,
    read_header,
    sha256,
    verify_sha256,
    write_finite,
    write_header,
)

from helpers import write_checkpoint


def test_header_is_padded_to_eight_bytes():
    stream = io.BytesIO()
    end = write_header(stream, {"a": {"shape": [1], "dtype": "I8", "data_offsets": [0, 1]}})
    assert end % 8 == 0
    assert int.from_bytes(stream.getvalue()[:8], "little") == end - 8
    assert json.loads(stream.getvalue()[8:])["a"]["dtype"] == "I8"


def test_written_checkpoints_load_with_safetensors(tmp_path):
    path = tmp_path / "model.safetensors"
    values = np.arange(6, dtype=np.float16).reshape(2, 3)
    write_checkpoint(path, {"w": ("F16", values), "i": ("I64", np.array([1, -2], dtype=np.int64))})
    loaded = load_file(path)
    np.testing.assert_array_equal(loaded["w"], values)
    assert loaded["i"].tolist() == [1, -2]
    header, offset = read_header(path)
    assert set(header) == {"w", "i"} and offset % 8 == 0


def test_rejects_unaccounted_tensor_bytes(tmp_path):
    path = tmp_path / "model.safetensors"
    write_checkpoint(path, {"w": ("F16", np.zeros(2, dtype=np.float16))})
    with path.open("ab") as stream:
        stream.write(b"\0\0")
    with pytest.raises(ValueError, match="Unaccounted"):
        read_header(path)


def test_bf16_rounds_through_f32_to_f16():
    bits = np.array([0x3F80, 0xC000, 0x3F81, 0x0001], dtype=np.uint16)
    expected = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
    np.testing.assert_array_equal(bf16_to_fp16(bits), expected)


def test_nonfinite_output_is_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        write_finite(io.BytesIO(), np.array([np.inf], dtype=np.float16))


def test_verify_sha256(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"pinned")
    assert verify_sha256(path, sha256(path)) == sha256(path)
    with pytest.raises(IdentityError, match="Identity mismatch"):
        verify_sha256(path, "0" * 64)
