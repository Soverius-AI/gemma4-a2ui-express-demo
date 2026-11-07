import json

import numpy as np
import pytest

from a2ui_model_tools.checkpoint_io import IdentityError, load_tensor, read_header, sha256
from a2ui_model_tools.merge import adapter_plan, hf_tensor_name, merge

from helpers import to_bf16_bits, write_checkpoint

MODULE = "language_model.model.layers.0.mlp.up_proj"
TARGET = "model.language_model.layers.0.mlp.up_proj.weight"
ORIGINAL = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
PRESERVED_FLOAT = np.array([1.234567, -4.7], dtype=np.float32)
PRESERVED_INT = np.array([0, -1, 2**60], dtype=np.int64)
# mlx_lm orientation: A [in, r], B [r, out].
A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
B = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)


@pytest.fixture
def base(tmp_path):
    directory = tmp_path / "base"
    directory.mkdir()
    bits = to_bf16_bits(ORIGINAL)
    write_checkpoint(
        directory / "model.safetensors",
        {
            TARGET: ("BF16", bits),
            "float": ("F32", PRESERVED_FLOAT),
            "integer": ("I64", PRESERVED_INT),
            "other": ("BF16", bits),
        },
    )
    (directory / "config.json").write_text("{}")
    return directory


def mlx_adapter(directory):
    directory.mkdir()
    write_checkpoint(
        directory / "adapters.safetensors",
        {
            MODULE + ".lora_a": ("F32", A),
            MODULE + ".lora_b": ("F32", B),
        },
    )
    (directory / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"rank": 2, "scale": 2.0}}))
    return directory


def peft_adapter(directory, dtype="F32"):
    """The same LoRA as mlx_adapter, saved the way PEFT/Unsloth saves it."""
    directory.mkdir()
    module = "base_model.model.model.language_model.layers.0.mlp.up_proj"
    cast = {"F32": lambda v: v, "BF16": to_bf16_bits}[dtype]
    write_checkpoint(
        directory / "adapter_model.safetensors",
        {
            module + ".lora_A.weight": (dtype, cast(np.ascontiguousarray(A.T))),
            module + ".lora_B.weight": (dtype, cast(np.ascontiguousarray(B.T))),
        },
    )
    (directory / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4, "revision": None})
    )
    return directory


def run_merge(base, adapter, output, **overrides):
    options = {
        "expected_base_sha256": sha256(base / "model.safetensors"),
        "expected_adapter_sha256": sha256(next(adapter.glob("*.safetensors"))),
        "base_model": "fixture",
        "base_revision": "fixture",
        "expected_modules": 1,
        "expected_rank": 2,
        "chunk_bytes": 32,
    } | overrides
    return merge(base, adapter, output, **options)


def read(output, key):
    header, offset = read_header(output / "model.safetensors")
    return load_tensor(output / "model.safetensors", offset, header[key])


def expected_merge(a=A, b=B):
    return (ORIGINAL.astype(np.float16).astype(np.float32) + 2.0 * (a @ b).T).astype(np.float16)


def bf16_rounded(values):
    return (to_bf16_bits(values).astype(np.uint32) << 16).view(np.float32)


def test_merge_orientation_scale_and_preservation(base, tmp_path):
    adapter, output = mlx_adapter(tmp_path / "adapter"), tmp_path / "output"
    result = run_merge(base, adapter, output)
    np.testing.assert_array_equal(read(output, TARGET), expected_merge())
    assert read(output, "float").tobytes() == PRESERVED_FLOAT.tobytes()
    assert read(output, "integer").tobytes() == PRESERVED_INT.tobytes()
    np.testing.assert_array_equal(read(output, "other"), ORIGINAL.astype(np.float16))
    assert result["adapted_logical_matrices"] == 1
    assert result["adapter_format"] == "mlx"
    assert not result["scale_divided_by_rank"]
    assert result["merged_sha256"] == sha256(output / "model.safetensors")
    assert (output / "config.json").read_text() == "{}"


def test_wrong_identity_writes_nothing(base, tmp_path):
    adapter, output = mlx_adapter(tmp_path / "adapter"), tmp_path / "output"
    with pytest.raises(IdentityError, match="Identity mismatch"):
        run_merge(base, adapter, output, expected_adapter_sha256="0" * 64)
    assert not output.exists()


def test_refuses_existing_output(base, tmp_path):
    adapter, output = mlx_adapter(tmp_path / "adapter"), tmp_path / "output"
    run_merge(base, adapter, output)
    with pytest.raises(FileExistsError):
        run_merge(base, adapter, output)


def test_rank_and_module_count_are_checked(base, tmp_path):
    adapter = mlx_adapter(tmp_path / "adapter")
    with pytest.raises(ValueError, match="rank/scale"):
        run_merge(base, adapter, tmp_path / "a", expected_scale=1.0)
    with pytest.raises(ValueError, match="logical modules"):
        run_merge(base, adapter, tmp_path / "b", expected_modules=2)


@pytest.mark.parametrize("dtype", ["F32", "BF16"])
def test_peft_adapter_merges_like_mlx(base, tmp_path, dtype):
    adapter, output = peft_adapter(tmp_path / "adapter", dtype), tmp_path / "output"
    result = run_merge(base, adapter, output)
    assert result["adapter_format"] == "peft"
    assert result["scale"] == 2.0  # lora_alpha / r
    a, b = (A, B) if dtype == "F32" else (bf16_rounded(A), bf16_rounded(B))
    np.testing.assert_array_equal(read(output, TARGET), expected_merge(a, b))


def test_every_adapter_key_is_accounted_for():
    with pytest.raises(ValueError, match="Unaccounted"):
        adapter_plan({"unexpected": {}}, {}, 8)
    with pytest.raises(ValueError, match="Unaccounted"):
        adapter_plan({"x.lora_A.weight": {}}, {}, 8, "peft")


def test_tensor_names():
    assert hf_tensor_name(MODULE, "mlx") == TARGET
    assert hf_tensor_name("base_model.model.model.language_model.layers.0.mlp.up_proj", "peft") == TARGET
