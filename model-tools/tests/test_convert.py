import numpy as np
import pytest

from a2ui_model_tools import identities as pins
from a2ui_model_tools.checkpoint_io import IdentityError, verify_sha256
from a2ui_model_tools.convert import fetch_shard, quantize, target_groups, validate_shapes, verify_untouched

PREFIX = "model.language_model.layers.0.mlp."


def test_rounding_ties_to_even_and_packing():
    # Scale is exactly 1; half ties must go to even, not away from zero.
    values = np.array([-7, -6.5, -5.5, -0.5, 0.5, 1.5, 6.5, 7] * 4, dtype=np.float16)[None, :]
    packed, scale = quantize(values)
    np.testing.assert_array_equal(scale, np.ones((1, 1), dtype=np.float16))
    codes = [0, 0, 2, 6, 8, 8, 14, 14]
    expected = sum(code << (4 * i) for i, code in enumerate(codes))
    np.testing.assert_array_equal(packed, np.full((1, 4), expected, dtype=np.uint32))


def test_zero_groups_fail_closed():
    with pytest.raises(ValueError, match="Zero/nonfinite scale"):
        quantize(np.zeros((2, 32), dtype=np.float16))


def test_quantize_input_checks():
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize(np.ones((2, 31), dtype=np.float16))
    with pytest.raises(ValueError, match="F16"):
        quantize(np.ones((2, 32), dtype=np.float32))


def test_gate_is_fused_before_up():
    groups = target_groups(
        [{"hf_tensor": PREFIX + "up_proj.weight"}, {"hf_tensor": PREFIX + "gate_proj.weight"}]
    )
    assert groups["language_model.layers.0.mlp.gate_up_proj"] == [
        PREFIX + "gate_proj.weight",
        PREFIX + "up_proj.weight",
    ]
    with pytest.raises(ValueError, match="both gate and up"):
        target_groups([{"hf_tensor": PREFIX + "gate_proj.weight"}])


def test_mlc_shape_mismatch():
    name = "model.language_model.layers.0.per_layer_input_gate.weight"
    groups = target_groups([{"hf_tensor": name}])
    index = {
        "language_model.layers.0.per_layer_input_gate.weight": (
            {},
            {"shape": [2, 31], "dtype": "float16", "nbytes": 124},
        )
    }
    with pytest.raises(ValueError, match="MLC shape/dtype mismatch"):
        validate_shapes(
            groups,
            {name: {"shape": [2, 32], "dtype": "BF16"}},
            {name: {"shape": [2, 32], "dtype": "F16"}},
            index,
        )


def test_wrong_base_identity(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not the pinned model")
    with pytest.raises(IdentityError, match="Identity mismatch"):
        verify_sha256(tmp_path / "model.safetensors", pins.BASE_WEIGHTS_SHA256)


def test_corrupt_cached_shard_is_never_replaced(tmp_path):
    (tmp_path / "params_shard_0.bin").write_bytes(b"bad")
    shard = {"dataPath": "params_shard_0.bin", "nbytes": 4, "md5sum": "0" * 32}
    with pytest.raises(IdentityError, match="pinned identity"):
        fetch_shard(tmp_path, shard)
    assert (tmp_path / "params_shard_0.bin").read_bytes() == b"bad"


def test_unsafe_shard_name(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        fetch_shard(tmp_path, {"dataPath": "../escape.bin", "nbytes": 1, "md5sum": "0" * 32})


def test_untouched_ranges(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    before.write_bytes(b"aaaaBBBBcccc")
    shard = {"nbytes": 12, "records": [{"name": "t", "byteOffset": 4, "nbytes": 4}]}
    after.write_bytes(b"aaaaXXXXcccc")
    verify_untouched(before, after, shard, {"t"})
    after.write_bytes(b"aaaaXXXXcccd")
    with pytest.raises(ValueError, match="Unaffected shard bytes changed"):
        verify_untouched(before, after, shard, {"t"})
