"""Merge a LoRA adapter into the original BF16 Hugging Face checkpoint.

Recipe: BF16 -> F16 -> F32, add scale * delta in F32, round to F16. The
intermediate F16 rounding of the base matches the published ONNX export, and
the scale is applied as stored (no division by rank). Unadapted BF16 tensors
become F16; every other unadapted tensor keeps its bytes.

Both mlx_lm adapters (``adapters.safetensors``, A [in, r], B [r, out]) and PEFT
adapters such as Unsloth's (``adapter_model.safetensors``, A [r, in],
B [out, r]) are supported.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np

from . import identities as pins
from .checkpoint_io import (
    Header,
    bf16_to_fp16,
    dtype_size,
    load_tensor,
    read_header,
    sha256,
    tensors,
    verify_sha256,
    write_finite,
    write_header,
)

log = logging.getLogger(__name__)

AdapterFormat = Literal["mlx", "peft"]
ADAPTER_WEIGHTS: dict[AdapterFormat, str] = {
    "mlx": "adapters.safetensors",
    "peft": "adapter_model.safetensors",
}
PEFT_PREFIX = "base_model.model."
# Tokenizer, template and config files copied unchanged beside the merged weights.
ASSETS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class LoraPair:
    module: str
    a_key: str
    b_key: str


@dataclass(frozen=True, slots=True)
class Adapter:
    format: AdapterFormat
    path: Path
    config: dict[str, Any]
    rank: int
    scale: float
    header: Header
    data_offset: int
    pairs: dict[str, LoraPair]  # by the base tensor each pair patches

    def factors(self, pair: LoraPair) -> tuple[np.ndarray, np.ndarray]:
        a, b = (
            _as_float32(load_tensor(self.path, self.data_offset, self.header[key]), key, self.header)
            for key in (pair.a_key, pair.b_key)
        )
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Nonfinite adapter: {pair.module}")
        return a, b

    def delta(self, pair: LoraPair) -> np.ndarray:
        """scale * B·A in the base tensor's [out, in] orientation, in F32."""
        a, b = self.factors(pair)
        if self.format == "mlx":
            delta = a @ b
            delta *= np.float32(self.scale)
            return delta.T
        delta = b @ a
        delta *= np.float32(self.scale)
        return delta


def _as_float32(array: np.ndarray, key: str, header: Header) -> np.ndarray:
    dtype = header[key]["dtype"]
    if dtype == "F32":
        return array
    if dtype == "F16":
        return array.astype(np.float32)
    # BF16 is the top half of an F32.
    return (array.astype(np.uint32) << 16).view(np.float32)


def detect_format(adapter_dir: Path) -> AdapterFormat:
    found = [fmt for fmt, name in ADAPTER_WEIGHTS.items() if (adapter_dir / name).is_file()]
    if len(found) != 1:
        raise ValueError(f"{adapter_dir} must contain exactly one of {', '.join(ADAPTER_WEIGHTS.values())}")
    return found[0]


def hf_tensor_name(module: str, fmt: AdapterFormat) -> str:
    """The base checkpoint tensor a LoRA module patches."""
    if fmt == "mlx":
        return module.replace("language_model.model.", "model.language_model.") + ".weight"
    return module.removeprefix(PEFT_PREFIX) + ".weight"


def adapter_hyperparameters(config: dict[str, Any], fmt: AdapterFormat) -> tuple[int, float]:
    if fmt == "mlx":
        lora = config.get("lora_parameters", config)
        return int(lora["rank"]), float(lora["scale"])
    if config.get("peft_type", "LORA") != "LORA" or config.get("use_dora"):
        raise ValueError("Only plain LoRA PEFT adapters are supported")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("Per-module rank/alpha patterns are not supported")
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    return rank, alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)


def adapter_plan(header: Header, base: Header, rank: int, fmt: AdapterFormat = "mlx") -> dict:
    """Pair every adapter tensor with the base tensor it patches, or fail."""
    a_suffix, b_suffix = (".lora_a", ".lora_b") if fmt == "mlx" else (".lora_A.weight", ".lora_B.weight")
    keys = {name for name, _ in tensors(header)}
    modules = sorted(key.removesuffix(a_suffix) for key in keys if key.endswith(a_suffix))
    expected = {module + suffix for module in modules for suffix in (a_suffix, b_suffix)}
    if not modules or expected != keys:
        raise ValueError("Unaccounted adapter keys / missing A or B")
    plan: dict[str, LoraPair] = {}
    for module in modules:
        pair = LoraPair(module, module + a_suffix, module + b_suffix)
        name = hf_tensor_name(module, fmt)
        if name not in base or name in plan:
            raise ValueError(f"Missing/duplicate base target: {name}")
        _check_pair_shape(
            header[pair.a_key], header[pair.b_key], base[name], rank=rank, fmt=fmt, module=module
        )
        plan[name] = pair
    return plan


def _check_pair_shape(a: dict, b: dict, target: dict, *, rank: int, fmt: AdapterFormat, module: str) -> None:
    allowed = {"F32"} if fmt == "mlx" else {"F32", "F16", "BF16"}
    delta_shape = None
    if len(a["shape"]) == 2 and len(b["shape"]) == 2:
        if fmt == "mlx":  # A [in, r], B [r, out]
            (inputs, a_rank), (b_rank, outputs) = a["shape"], b["shape"]
        else:  # A [r, in], B [out, r]
            (a_rank, inputs), (outputs, b_rank) = a["shape"], b["shape"]
        if a_rank == b_rank == rank:
            delta_shape = [outputs, inputs]
    if (
        a["dtype"] not in allowed
        or b["dtype"] not in allowed
        or delta_shape is None
        or target["shape"] != delta_shape
        or target["dtype"] != "BF16"
    ):
        raise ValueError(f"Incompatible adapter shape/dtype: {module}")


def load_adapter(
    adapter_dir: Path, base_header: Header, *, expected_sha256: str, base_revision: str
) -> tuple[Adapter, str]:
    fmt = detect_format(adapter_dir)
    path = adapter_dir / ADAPTER_WEIGHTS[fmt]
    digest = verify_sha256(path, expected_sha256)
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank, scale = adapter_hyperparameters(config, fmt)
    pinned_revision = config.get("base_model_commit_hash" if fmt == "mlx" else "revision")
    if pinned_revision not in (None, base_revision):
        raise ValueError("Adapter base revision mismatch")
    header, offset = read_header(path)
    plan = adapter_plan(header, base_header, rank, fmt)
    return Adapter(fmt, path, config, rank, scale, header, offset, plan), digest


def output_header(base_header: Header) -> Header:
    """The merged layout: BF16 tensors become F16, everything else keeps its dtype."""
    header: Header = {}
    if "__metadata__" in base_header:
        header["__metadata__"] = base_header["__metadata__"]
    cursor = 0
    for name, meta in tensors(base_header):
        dtype = "F16" if meta["dtype"] == "BF16" else meta["dtype"]
        size = math.prod(meta["shape"]) * dtype_size(dtype)
        header[name] = {"dtype": dtype, "shape": meta["shape"], "data_offsets": [cursor, cursor + size]}
        cursor += size
    return header


def _write_merged(stream: BinaryIO, source: np.ndarray, delta: np.ndarray, step: int) -> None:
    # Row slices avoid copying the whole (possibly transposed) delta.
    rows = max(1, step // source.shape[1])
    for start in range(0, source.shape[0], rows):
        values = bf16_to_fp16(source[start : start + rows]).astype(np.float32)
        values += delta[start : start + rows]
        with np.errstate(over="ignore"):
            write_finite(stream, values.astype(np.float16))


def _write_unadapted(stream: BinaryIO, source: np.ndarray, dtype: str, step: int) -> None:
    flat = source.reshape(-1)
    for start in range(0, flat.size, step):
        values = flat[start : start + step]
        write_finite(stream, bf16_to_fp16(values) if dtype == "BF16" else values)


def merge(
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    expected_base_sha256: str = pins.BASE_WEIGHTS_SHA256,
    expected_adapter_sha256: str = pins.ADAPTER_SHA256,
    base_model: str = pins.BASE.repo_id,
    base_revision: str = pins.BASE.revision,
    expected_modules: int = pins.ADAPTER_MODULES,
    expected_rank: int = pins.ADAPTER_RANK,
    expected_scale: float = pins.ADAPTER_SCALE,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    """Write ``output_dir/model.safetensors`` and ``merge-provenance.json``."""
    base_dir, adapter_dir, output_dir = Path(base_dir), Path(adapter_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output: {output_dir}")
    base_path = base_dir / "model.safetensors"
    log.info("Verifying base and adapter SHA-256")
    base_sha = verify_sha256(base_path, expected_base_sha256)
    base_header, base_offset = read_header(base_path)
    adapter, adapter_sha = load_adapter(
        adapter_dir, base_header, expected_sha256=expected_adapter_sha256, base_revision=base_revision
    )
    if adapter.rank != expected_rank or adapter.scale != expected_scale or not np.isfinite(adapter.scale):
        raise ValueError("Adapter rank/scale identity mismatch")
    if len(adapter.pairs) != expected_modules:
        raise ValueError(f"Expected {expected_modules} logical modules, found {len(adapter.pairs)}")

    output_dir.mkdir(parents=True)
    output = output_dir / "model.safetensors"
    step = max(1, chunk_bytes // 16)  # bounds temporaries regardless of tensor size
    completed = []
    with output.open("xb") as stream:
        write_header(stream, output_header(base_header))
        for index, (name, meta) in enumerate(tensors(base_header)):
            source = load_tensor(base_path, base_offset, meta)
            pair = adapter.pairs.get(name)
            if pair is None:
                _write_unadapted(stream, source, meta["dtype"], step)
            else:
                _write_merged(stream, source, adapter.delta(pair), step)
                completed.append({"module": pair.module, "hf_tensor": name, "shape": meta["shape"]})
            if index % 100 == 0 or pair is not None:
                log.debug("%d: %s", index, name)
    if len(completed) != len(adapter.pairs):
        raise ValueError("Unmerged adapter modules")

    log.info("Hashing merged checkpoint")
    provenance = {
        "format_version": 1,
        "created_unix": time.time(),
        "base_model": base_model,
        "base_revision": base_revision,
        "base_path": str(base_path.resolve()),
        "base_sha256": base_sha,
        "base_bytes": base_path.stat().st_size,
        "adapter_format": adapter.format,
        "adapter_path": str(adapter.path.resolve()),
        "adapter_sha256": adapter_sha,
        "adapter_bytes": adapter.path.stat().st_size,
        "adapter_config": adapter.config,
        "adapter_config_sha256": sha256(adapter_dir / "adapter_config.json"),
        "rank": adapter.rank,
        "scale": adapter.scale,
        "scale_divided_by_rank": False,
        "adapted_logical_matrices": len(completed),
        "adapter_tensor_count": 2 * len(completed),
        "modules": completed,
        "merged_file": output.name,
        "merged_sha256": sha256(output),
        "merged_bytes": output.stat().st_size,
        "assets_sha256": _copy_assets(base_dir, output_dir),
        "recipe": (
            "HF BF16 -> FP16 -> FP32; add scale * delta in FP32; round FP16. "
            "No rank division. Non-BF16 unadapted bytes unchanged."
        ),
        "completion": "complete",
    }
    with (output_dir / "merge-provenance.json").open("x") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    log.info("Merged %d logical modules; SHA-256 %s", len(completed), provenance["merged_sha256"])
    return provenance


def _copy_assets(base_dir: Path, output_dir: Path) -> dict[str, str]:
    copied = {}
    for name in ASSETS:
        source = base_dir / name
        if source.is_file():
            with source.open("rb") as src, (output_dir / name).open("xb") as dst:
                shutil.copyfileobj(src, dst)
            copied[name] = sha256(output_dir / name)
    return copied
