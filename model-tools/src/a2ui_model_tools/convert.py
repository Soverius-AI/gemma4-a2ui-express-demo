"""Convert a merged checkpoint to MLC q4f16_1 by patching the official MLC base.

Only the adapted matrices are requantized. Before anything is written, every
affected matrix of the *original* base must quantize to exactly the official
MLC bytes, which proves the quantizer matches the producer's. Every other byte
of the official shards is copied and then checked unchanged.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import identities as pins
from .checkpoint_io import (
    Header,
    bf16_to_fp16,
    file_digest,
    load_tensor,
    read_header,
    sha256,
    verify_sha256,
)
from .download import digest_verifier, fetch

log = logging.getLogger(__name__)

GROUP_SIZE = 32
CHUNK_ROWS = 256
COPY_BLOCK_BYTES = 4 * 1024 * 1024
RAW_SUFFIX = ".per_layer_input_gate"  # stored as plain F16, not quantized


def quantize(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q4f16_1: F16 at each arithmetic boundary, ties to even, low nibble first.

    Zero scales are rejected: the pinned producer divides 0/0 there, whose
    storage cast is backend-specific, so canonical bytes cannot be derived.
    """
    if weight.dtype != np.float16 or weight.ndim != 2 or weight.shape[1] % GROUP_SIZE:
        raise ValueError("Quantization requires F16 NK matrix with K divisible by 32")
    if not np.isfinite(weight).all():
        raise ValueError("Nonfinite quantization input")
    groups = weight.reshape(weight.shape[0], -1, GROUP_SIZE)
    scale = (np.max(np.abs(groups), axis=-1) / np.float16(7)).astype(np.float16)
    if not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError(
            "Zero/nonfinite scale requires verified producer-specific zero bytes; refusing to guess"
        )
    values = (groups / scale[..., None]).astype(np.float16)
    values = (values + np.float16(7)).astype(np.float16)
    codes = np.clip(np.rint(values), 0, 14).astype(np.uint32).reshape(weight.shape[0], -1, 8)
    packed = np.zeros(codes.shape[:2], dtype="<u4")
    for i in range(8):
        packed |= codes[:, :, i] << np.uint32(4 * i)
    return packed, scale.astype("<f2", copy=False)


def target_groups(modules: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map MLC parameter groups to the HF tensors they are built from.

    MLC fuses gate_proj and up_proj into gate_up_proj (gate rows first).
    """
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for entry in modules:
        name = entry["hf_tensor"]
        if name in seen or not name.startswith("model.language_model.") or not name.endswith(".weight"):
            raise ValueError("Duplicate/unsupported HF adapter target")
        seen.add(name)
        mapped = name.removeprefix("model.").removesuffix(".weight")
        if mapped.endswith((".mlp.gate_proj", ".mlp.up_proj")):
            mapped = mapped.rsplit(".", 1)[0] + ".gate_up_proj"
        groups.setdefault(mapped, []).append(name)
    for group, names in groups.items():
        if group.endswith(".mlp.gate_up_proj"):
            if len(names) != 2 or {n.rsplit(".", 2)[-2] for n in names} != {"gate_proj", "up_proj"}:
                raise ValueError("Fusion requires both gate and up adapter matrices")
            names.sort(key=lambda n: not n.endswith(".gate_proj.weight"))
        elif len(names) != 1:
            raise ValueError("Unexpected multiple target matrices")
    return groups


def group_arrays(group: str) -> tuple[str, ...]:
    return (".weight",) if group.endswith(RAW_SUFFIX) else (".q_weight", ".q_scale")


def encode(rows: np.ndarray, group: str) -> list[np.ndarray]:
    """The stored MLC arrays for F16 rows of ``group``."""
    return [rows] if group.endswith(RAW_SUFFIX) else list(quantize(rows))


def fetch_shard(cache: Path, shard: dict[str, Any]) -> Path:
    """Download one official shard, verified by the size and MD5 MLC records."""
    name = shard["dataPath"]
    if Path(name).name != name:
        raise ValueError("Unsafe shard name")
    target = cache / name
    verify = digest_verifier(shard["md5sum"], algorithm="md5", size=shard["nbytes"])
    fetch(pins.MLC_BASE.url(name), target, verify, timeout=120)
    return target


def index_tensors(tensor_cache: dict[str, Any]) -> dict[str, tuple[dict, dict]]:
    """Map each MLC tensor name to its (shard record, tensor record)."""
    result: dict[str, tuple[dict, dict]] = {}
    for shard in tensor_cache["records"]:
        for tensor in shard["records"]:
            if tensor["name"] in result:
                raise ValueError("Duplicate MLC tensor")
            if tensor["byteOffset"] < 0 or tensor["byteOffset"] + tensor["nbytes"] > shard["nbytes"]:
                raise ValueError("Invalid MLC tensor bounds")
            result[tensor["name"]] = (shard, tensor)
    return result


def validate_shapes(
    groups: dict[str, list[str]], base_header: Header, merged_header: Header, index: dict
) -> None:
    for group, names in groups.items():
        shapes = []
        for name in names:
            if name not in base_header or name not in merged_header:
                raise ValueError(f"Missing HF target {name}")
            base, merged = base_header[name], merged_header[name]
            if (
                base["shape"] != merged["shape"]
                or len(base["shape"]) != 2
                or base["dtype"] != "BF16"
                or merged["dtype"] != "F16"
            ):
                raise ValueError(f"HF shape/dtype mismatch: {name}")
            shapes.append(base["shape"])
        if len({shape[1] for shape in shapes}) != 1:
            raise ValueError("Fusion input-width mismatch")
        rows, width = sum(shape[0] for shape in shapes), shapes[0][1]
        if group.endswith(RAW_SUFFIX):
            expected = {group + ".weight": ([rows, width], "float16")}
        else:
            if width % GROUP_SIZE:
                raise ValueError("Invalid quantized width")
            expected = {
                group + ".q_weight": ([rows, width // 8], "uint32"),
                group + ".q_scale": ([rows, width // GROUP_SIZE], "float16"),
            }
        for name, (shape, dtype) in expected.items():
            if name not in index:
                raise ValueError(f"Missing MLC target {name}")
            tensor = index[name][1]
            if (
                tensor["shape"] != shape
                or tensor["dtype"] != dtype
                or tensor["nbytes"] != np.prod(shape) * np.dtype(dtype).itemsize
            ):
                raise ValueError(f"MLC shape/dtype mismatch: {name}")


@dataclass
class Checkpoint:
    path: Path
    header: Header
    offset: int

    @classmethod
    def open(cls, path: Path) -> Checkpoint:
        header, offset = read_header(path)
        return cls(path, header, offset)

    def tensor(self, name: str) -> np.memmap:
        return load_tensor(self.path, self.offset, self.header[name])


@dataclass
class Patch:
    """One stored MLC array range: where it lives and its original/replacement bytes."""

    target: str
    shard: dict[str, Any]
    tensor: dict[str, Any]
    offset: int
    original: bytes
    replacement: bytes | None


@dataclass
class Conversion:
    base: Checkpoint
    merged: Checkpoint
    groups: dict[str, list[str]]
    index: dict[str, tuple[dict, dict]]
    cache: Path
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def patches(self, *, with_replacements: bool) -> Iterator[Patch]:
        """Yield every affected array range in CHUNK_ROWS row blocks."""
        for number, (group, names) in enumerate(sorted(self.groups.items())):
            row_offset = 0
            for name in names:
                base = self.base.tensor(name)
                merged = self.merged.tensor(name) if with_replacements else None
                for start in range(0, base.shape[0], CHUNK_ROWS):
                    original = encode(bf16_to_fp16(base[start : start + CHUNK_ROWS]), group)
                    updates = (
                        encode(np.asarray(merged[start : start + CHUNK_ROWS]), group)
                        if merged is not None
                        else None
                    )
                    for position, suffix in enumerate(group_arrays(group)):
                        shard, tensor = self.index[group + suffix]
                        columns = tensor["shape"][1]
                        itemsize = np.dtype(tensor["dtype"]).itemsize
                        yield Patch(
                            target=group + suffix,
                            shard=shard,
                            tensor=tensor,
                            offset=tensor["byteOffset"] + (row_offset + start) * columns * itemsize,
                            original=np.ascontiguousarray(original[position]).tobytes(),
                            replacement=(
                                np.ascontiguousarray(updates[position]).tobytes()
                                if updates is not None
                                else None
                            ),
                        )
                row_offset += base.shape[0]
                del base, merged
            if number % 20 == 0:
                log.info(
                    "%s: %d/%d %s",
                    "Patch" if with_replacements else "Base parity",
                    number + 1,
                    len(self.groups),
                    group,
                )

    def official_bytes(self, patch: Patch) -> bytes:
        with (self.cache / patch.shard["dataPath"]).open("rb") as stream:
            stream.seek(patch.offset)
            return stream.read(len(patch.original))

    def check_base_parity(self) -> None:
        for patch in self.patches(with_replacements=False):
            if patch.original != self.official_bytes(patch):
                raise ValueError(
                    f"BASE PARITY FAILURE: {patch.target} at byte {patch.offset}; abort without normalization"
                )

    def write(self, output_dir: Path) -> None:
        for patch in self.patches(with_replacements=True):
            official = self.official_bytes(patch)
            if patch.original != official:
                raise ValueError(f"BASE PARITY FAILURE: {patch.target} at byte {patch.offset}")
            assert patch.replacement is not None
            if len(patch.replacement) != len(official):
                raise ValueError("Replacement size mismatch")
            with (output_dir / patch.shard["dataPath"]).open("r+b") as stream:
                stream.seek(patch.offset)
                stream.write(patch.replacement)
            item = self.stats.setdefault(
                patch.target,
                {
                    "name": patch.target,
                    "shard": patch.shard["dataPath"],
                    "shape": patch.tensor["shape"],
                    "dtype": patch.tensor["dtype"],
                    "baseline_parity": True,
                    "compared_bytes": 0,
                    "changed_bytes": 0,
                },
            )
            item["compared_bytes"] += len(official)
            item["changed_bytes"] += int(
                np.count_nonzero(
                    np.frombuffer(official, dtype=np.uint8)
                    != np.frombuffer(patch.replacement, dtype=np.uint8)
                )
            )


def verify_untouched(before: Path, after: Path, shard: dict[str, Any], patched: set[str]) -> None:
    """Every byte outside the patched tensors, padding included, must be unchanged."""
    ranges = sorted(
        (t["byteOffset"], t["byteOffset"] + t["nbytes"]) for t in shard["records"] if t["name"] in patched
    )
    cursor = 0
    with before.open("rb") as old, after.open("rb") as new:
        for start, end in [*ranges, (shard["nbytes"], shard["nbytes"])]:
            old.seek(cursor)
            new.seek(cursor)
            left = start - cursor
            if left < 0:
                raise ValueError("Overlapping modified tensors")
            while left:
                amount = min(left, COPY_BLOCK_BYTES)
                if old.read(amount) != new.read(amount):
                    raise ValueError("Unaffected shard bytes changed")
                left -= amount
            cursor = end


def _verify_inputs(
    base_dir: Path, merged_dir: Path, producer_dir: Path, expected_merged_sha256: str
) -> tuple[dict, dict]:
    provenance = json.loads((merged_dir / "merge-provenance.json").read_text())
    verify_sha256(base_dir / "model.safetensors", pins.BASE_WEIGHTS_SHA256)
    verify_sha256(merged_dir / "model.safetensors", expected_merged_sha256)
    if (
        provenance["merged_sha256"] != expected_merged_sha256
        or provenance["base_sha256"] != pins.BASE_WEIGHTS_SHA256
        or provenance["adapted_logical_matrices"] != pins.ADAPTER_MODULES
        or provenance["adapter_tensor_count"] != 2 * pins.ADAPTER_MODULES
    ):
        raise ValueError("Merge provenance identity/accounting mismatch")
    build = json.loads((producer_dir / "mlc-build-info.json").read_text())
    if (
        build["source_weights_sha256"] != pins.BASE_WEIGHTS_SHA256
        or build["mlc_llm"]["commit"] != pins.MLC_LLM_COMMIT
        or build["quantization"] != pins.QUANTIZATION
    ):
        raise ValueError("Official producer build identity mismatch")
    return provenance, build


def convert(
    base_dir: Path,
    merged_dir: Path,
    producer_dir: Path,
    output_dir: Path,
    *,
    expected_merged_sha256: str = pins.MERGED_WEIGHTS_SHA256,
) -> dict[str, Any]:
    """Write the patched MLC model and ``conversion-provenance.json`` to ``output_dir``."""
    base_dir, merged_dir, producer_dir, output_dir = map(
        Path, (base_dir, merged_dir, producer_dir, output_dir)
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output: {output_dir}")
    provenance, build = _verify_inputs(base_dir, merged_dir, producer_dir, expected_merged_sha256)
    tensor_cache = json.loads((producer_dir / "tensor-cache.json").read_text())
    index = index_tensors(tensor_cache)
    base = Checkpoint.open(base_dir / "model.safetensors")
    merged = Checkpoint.open(merged_dir / "model.safetensors")
    groups = target_groups(provenance["modules"])
    if len(groups) != pins.MLC_TARGET_GROUPS:
        raise ValueError(f"Expected {pins.MLC_TARGET_GROUPS} fused/ungrouped target groups")
    validate_shapes(groups, base.header, merged.header, index)

    cache = producer_dir / "base-mlc"
    cache.mkdir(exist_ok=True)
    shards = tensor_cache["records"]
    log.info("Downloading/verifying %d official shards", len(shards))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda shard: fetch_shard(cache, shard), shards))

    conversion = Conversion(base, merged, groups, index, cache)
    conversion.check_base_parity()  # all of it, before any output exists
    log.info("All affected base arrays have exact official byte parity")
    output_dir.mkdir(parents=True)
    for shard in shards:
        with (
            (cache / shard["dataPath"]).open("rb") as src,
            (output_dir / shard["dataPath"]).open("xb") as dst,
        ):
            shutil.copyfileobj(src, dst, COPY_BLOCK_BYTES)
    conversion.write(output_dir)
    if len(conversion.stats) != pins.MLC_PATCHED_ARRAYS:
        raise ValueError(f"Expected exactly {pins.MLC_PATCHED_ARRAYS} patched arrays")

    output_cache = copy.deepcopy(tensor_cache)
    shard_stats = []
    for shard, out_shard in zip(shards, output_cache["records"], strict=True):
        before, after = cache / shard["dataPath"], output_dir / shard["dataPath"]
        verify_untouched(before, after, shard, set(conversion.stats))
        out_shard["md5sum"] = file_digest(after, "md5")
        shard_stats.append(
            {
                "file": shard["dataPath"],
                "nbytes": shard["nbytes"],
                "base_md5": shard["md5sum"],
                "base_sha256": sha256(before),
                "output_md5": out_shard["md5sum"],
                "output_sha256": sha256(after),
                "untouched_ranges_exact": True,
            }
        )
    with (output_dir / "tensor-cache.json").open("x") as stream:
        json.dump(output_cache, stream, indent=2)
        stream.write("\n")

    manifest = json.loads((producer_dir / "mlc-model-manifest.json").read_text())
    result = {
        "format_version": 1,
        "completion": "complete",
        "created_unix": time.time(),
        "conversion": (
            "Requantize affected exact merged-HF matrices only; preserve all other official MLC bytes"
        ),
        "base_hf_sha256": pins.BASE_WEIGHTS_SHA256,
        "merged_hf_sha256": expected_merged_sha256,
        "merge_provenance_sha256": sha256(merged_dir / "merge-provenance.json"),
        "adapter_sha256": provenance["adapter_sha256"],
        "rank": provenance["rank"],
        "scale": provenance["scale"],
        "official_repo": pins.MLC_BASE.repo_id,
        "official_revision": pins.MLC_BASE.revision,
        "mlc_llm_commit": pins.MLC_LLM_COMMIT,
        "producer_build_info": build,
        "parameter_schema_id": manifest["weights"]["parameter_schema_id"],
        "interface_id": manifest["interface_id"],
        "original_tensor_cache_sha256": sha256(producer_dir / "tensor-cache.json"),
        "output_tensor_cache_sha256": sha256(output_dir / "tensor-cache.json"),
        "quantization": (
            "q4f16_1: group32 NK; F16 maxabs/7; clip(rint_even(F16(F16(w/scale)+7)),0,14); "
            "uint32 low-first 8 nibbles"
        ),
        "zero_scale_policy": "Fail closed: no unverified 0/0 cast behavior",
        "chunk_rows": CHUNK_ROWS,
        "adapted_logical_matrices": len(provenance["modules"]),
        "target_groups": len(groups),
        "patched_arrays": len(conversion.stats),
        "all_affected_base_bytes_exact": True,
        "all_unaffected_output_bytes_exact": True,
        "changed_bytes": sum(item["changed_bytes"] for item in conversion.stats.values()),
        "tensors": list(conversion.stats.values()),
        "shards": shard_stats,
    }
    with (output_dir / "conversion-provenance.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    log.info(
        "Converted %d logical matrices into %d groups / %d arrays; %d bytes changed",
        result["adapted_logical_matrices"],
        result["target_groups"],
        result["patched_arrays"],
        result["changed_bytes"],
    )
    return result
