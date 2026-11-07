"""Patch a q4f16 ONNX text export of Gemma 4 with a LoRA fine-tune.

The browser demo loads the ``onnx-community`` Gemma 4 exports through
Transformers.js. Their decoders keep every linear weight as a ``MatMulNBits``
initializer: 4-bit blocks of 32 along K with float16 scales and packed 4-bit
zero points. This module requantizes the adapted matrices with ONNX Runtime's
own quantizer and writes them over the base export's external data at their
original offsets. The graph files, the embedding data and every other byte
stay identical, which the app's ``decoder_graph_unchanged`` check requires.

    python -m a2ui_model_tools.onnx_patch <base-export-dir> <out-dir> \\
        --base-hf-dir <dir with the BF16 model.safetensors> \\
        --adapter-dir <mlx_lm adapter> [--adapter-sha256 <sha>] \\
        [--graph-revision <sha>] [--base-revision <sha>]

The merged weight of each adapted matrix is computed as ``model-tools merge``
does: BF16 -> FP16 -> FP32, plus ``scale * A @ B`` in FP32, rounded to FP16.
Before patching, the quantizer is calibrated against the base file: the base
checkpoint's layer-0 ``q_proj`` must requantize to the export's bytes, or to
the same dequantized error when a different quantizer build rounded ties
differently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("a2ui_model_tools.onnx_patch")

GRAPH = "onnx/decoder_model_merged_q4f16.onnx"
EMBED_GRAPH = "onnx/embed_tokens_q4f16.onnx"
SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
]
BLOCK = 32
HF_PREFIX = "model.language_model.layers"
# ONNX initializer stem -> the HF weights it holds, concatenated along N (out).
ONNX_TO_HF = {
    "attn_q_proj": ["self_attn.q_proj"],
    "attn_k_proj": ["self_attn.k_proj"],
    "attn_v_proj": ["self_attn.v_proj"],
    "attn_o_proj": ["self_attn.o_proj"],
    "mlp_gate_up_proj": ["mlp.gate_proj", "mlp.up_proj"],
    "mlp_down_proj": ["mlp.down_proj"],
    "per_layer_per_layer_input_gate": ["per_layer_input_gate"],
    "per_layer_per_layer_projection": ["per_layer_projection"],
}
LAYER_RE = re.compile(r"^model_layers_(\d+)_(.+)_MatMul_weight_quant$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Externals:
    """External initializers of a graph: name -> (file, offset, length, dtype, dims)."""

    def __init__(self, graph_path: Path):
        import onnx

        model = onnx.load(str(graph_path), load_external_data=False)
        self.entries: dict[str, tuple[str, int, int, int, list[int]]] = {}
        for init in model.graph.initializer:
            if init.data_location != onnx.TensorProto.EXTERNAL:
                continue
            kv = {e.key: e.value for e in init.external_data}
            self.entries[init.name] = (
                kv["location"],
                int(kv.get("offset", 0)),
                int(kv.get("length", 0)),
                init.data_type,
                list(init.dims),
            )
        self.attrs: dict[str, dict] = {}
        for node in model.graph.node:
            if node.op_type == "MatMulNBits":
                self.attrs[node.input[1]] = {
                    a.name: onnx.helper.get_attribute_value(a) for a in node.attribute
                }
        self.files = sorted({entry[0] for entry in self.entries.values()})

    def read(self, data: dict[str, np.memmap], name: str) -> np.ndarray:
        location, offset, length, dtype, dims = self.entries[name]
        np_dtype = {2: np.uint8, 10: np.float16}[dtype]
        return np.frombuffer(data[location][offset : offset + length], dtype=np_dtype).reshape(dims)


def quantize(weight_nk: np.ndarray, symmetric: bool, as_fp16: bool):
    """ORT's blockwise 4-bit quantization of an (N, K) weight: packed, scales, zero points."""
    from onnxruntime.capi._pybind_state import quantize_matmul_4bits

    n, k = weight_nk.shape
    if k % BLOCK:
        raise ValueError(f"K={k} is not a multiple of the block size")
    k_blocks = k // BLOCK
    # The quantizer takes the MatMul B operand, (K, N), one column per output.
    source = np.ascontiguousarray(weight_nk.T.astype(np.float16 if as_fp16 else np.float32))
    packed = np.zeros((n, k_blocks, BLOCK // 2), dtype=np.uint8)
    scales = np.zeros(n * k_blocks, dtype=source.dtype)
    zero_points = np.zeros(n * ((k_blocks + 1) // 2), dtype=np.uint8)
    quantize_matmul_4bits(packed, source, scales, zero_points, BLOCK, n, k, symmetric)
    return packed, scales.astype(np.float16).reshape(n, k_blocks), zero_points.reshape(n, (k_blocks + 1) // 2)


def dequantize(packed: np.ndarray, scales: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
    """(N, K) float32 weights from ORT's packed 4-bit blocks."""
    n, k_blocks, blob = packed.shape
    q = np.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(n, k_blocks, blob * 2).astype(np.float32)
    zp = (
        np.stack([zero_points & 0x0F, zero_points >> 4], axis=-1)
        .reshape(n, -1)[:, :k_blocks]
        .astype(np.float32)
    )
    return ((q - zp[:, :, None]) * scales.astype(np.float32)[:, :, None]).reshape(n, k_blocks * blob * 2)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))


class Weights:
    """Merged (N, K) float16 weights, computed per matrix from the BF16 base and an mlx_lm adapter."""

    def __init__(self, base_hf_dir: Path, adapter_dir: Path | None):
        from safetensors import safe_open

        self.base = safe_open(str(base_hf_dir / "model.safetensors"), "np")
        self.adapter = None
        self.scale = 0.0
        if adapter_dir is not None:
            config = json.loads((adapter_dir / "adapter_config.json").read_text())
            params = config.get("lora_parameters", config)
            self.scale = float(params["scale"])
            self.adapter = safe_open(str(adapter_dir / "adapters.safetensors"), "np")
            self.adapter_keys = set(self.adapter.keys())
        self.adapted = 0

    def base_weight(self, hf_name: str) -> np.ndarray:
        return self.base.get_tensor(hf_name).astype(np.float16)

    def merged(self, hf_name: str) -> np.ndarray:
        weight = self.base_weight(hf_name)
        if self.adapter is None:
            return weight
        # HF `model.language_model.layers.N.x.weight` is mlx `language_model.model.layers.N.x`.
        module = hf_name.removesuffix(".weight").replace("model.language_model.", "language_model.model.")
        a_key, b_key = f"{module}.lora_a", f"{module}.lora_b"
        if a_key not in self.adapter_keys:
            return weight
        lora_a = self.adapter.get_tensor(a_key).astype(np.float32)  # (in, r)
        lora_b = self.adapter.get_tensor(b_key).astype(np.float32)  # (r, out)
        delta = (lora_a @ lora_b).T  # (out, in)
        self.adapted += 1
        return (weight.astype(np.float32) + self.scale * delta).astype(np.float16)

    def for_stem(self, layer: int, stem: str, merged: bool) -> np.ndarray:
        names = [f"{HF_PREFIX}.{layer}.{module}.weight" for module in ONNX_TO_HF[stem]]
        parts = [self.merged(n) if merged else self.base_weight(n) for n in names]
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def calibrate(ext: Externals, data: dict[str, np.memmap], weights: Weights) -> tuple[bool, bool]:
    """Find the quantizer flags that reproduce the base export, or match its error."""
    name = "model_layers_0_attn_q_proj_MatMul_weight_quant"
    weight = weights.for_stem(0, "attn_q_proj", merged=False)
    want = (
        ext.read(data, name),
        ext.read(data, name.replace("_quant", "_scales")),
        ext.read(data, name.replace("_quant", "_zp")),
    )
    reference = weight.astype(np.float32)
    base_error = rmse(dequantize(*want), reference)
    best: tuple[float, bool, bool] | None = None
    for symmetric in (False, True):
        for as_fp16 in (True, False):
            got = quantize(weight, symmetric, as_fp16)
            exact = all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))
            mismatch = float(np.mean(got[0] != want[0]))
            error = rmse(dequantize(*got), reference)
            log.info(
                "calibration symmetric=%s fp16=%s exact=%s mismatching_bytes=%.5f rmse=%.6f (base %.6f)",
                symmetric,
                as_fp16,
                exact,
                mismatch,
                error,
                base_error,
            )
            if exact:
                return symmetric, as_fp16
            if best is None or mismatch < best[0]:
                best = (mismatch, symmetric, as_fp16)
    assert best is not None
    mismatch, symmetric, as_fp16 = best
    error = rmse(dequantize(*quantize(weight, symmetric, as_fp16)), reference)
    if mismatch > 0.01 or error > base_error * 1.05:
        raise ValueError("No quantizer setting reproduces the base export; refusing to patch")
    log.warning(
        "Quantizer is not byte-identical to the export (%.3f%% of bytes differ); same scheme and error",
        mismatch * 100,
    )
    return symmetric, as_fp16


def write_quantized(ext: Externals, out_data: dict[str, np.memmap], name: str, values) -> None:
    """Write packed weights, scales and zero points over the export's bytes."""
    for suffix, array in zip(("_quant", "_scales", "_zp"), values, strict=True):
        target = name.replace("_quant", suffix)
        location, offset, length, _, dims = ext.entries[target]
        raw = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        if raw.size != length or list(array.shape) != dims:
            raise ValueError(f"{target}: layout mismatch {array.shape} vs {dims}")
        out_data[location][offset : offset + length] = raw


@dataclass(frozen=True)
class Provenance:
    """What the export manifest records about its inputs."""

    adapter_sha256: str = ""
    base_model: str = "unsloth/gemma-4-E2B-it"
    base_revision: str = "d36bdd3855c82a6a1a23f7c459b749f5724ae75d"
    graph_revision: str = "6f21c99d7e18237f8d2ce92aadfdbfeba877f63a"


def patch(
    base_dir: Path,
    out_dir: Path,
    base_hf_dir: Path,
    adapter_dir: Path | None,
    provenance: Provenance | None = None,
) -> dict:
    provenance = provenance or Provenance()
    graph_path = base_dir / GRAPH
    if out_dir.exists():
        raise FileExistsError(f"Refusing existing output: {out_dir}")
    ext = Externals(graph_path)
    base_data = {f: np.memmap(base_dir / "onnx" / f, dtype=np.uint8, mode="r") for f in ext.files}
    weights = Weights(base_hf_dir, adapter_dir)
    symmetric, as_fp16 = calibrate(ext, base_data, weights)

    out_dir.mkdir(parents=True)
    (out_dir / "onnx").mkdir()
    log.info("Copying base export files")
    for name in (GRAPH, EMBED_GRAPH, *SMALL_FILES):
        if (base_dir / name).exists():
            shutil.copyfile(base_dir / name, out_dir / name)
    for extra in sorted((base_dir / "onnx").glob("embed_tokens_q4f16.onnx_data*")):
        shutil.copyfile(extra, out_dir / "onnx" / extra.name)
    for f in ext.files:
        shutil.copyfile(base_dir / "onnx" / f, out_dir / "onnx" / f)
    out_data = {f: np.memmap(out_dir / "onnx" / f, dtype=np.uint8, mode="r+") for f in ext.files}

    patched = []
    started = time.time()
    for name in sorted(ext.entries):
        match = LAYER_RE.match(name)
        if not match:
            continue
        layer, stem = int(match.group(1)), match.group(2)
        if stem not in ONNX_TO_HF:
            raise ValueError(f"Unknown decoder matrix {name}")
        weight = weights.for_stem(layer, stem, merged=True)
        attrs = ext.attrs[name]
        if (attrs["N"], attrs["K"]) != weight.shape:
            raise ValueError(f"{name}: export is {attrs['N']}x{attrs['K']}, weight is {weight.shape}")
        write_quantized(ext, out_data, name, quantize(weight, symmetric, as_fp16))
        patched.append(
            {
                "onnx": name,
                "hf": [f"{HF_PREFIX}.{layer}.{m}.weight" for m in ONNX_TO_HF[stem]],
                "shape": list(weight.shape),
            }
        )
        log.debug("patched %s", name)
    for data in out_data.values():
        data.flush()
    del out_data
    log.info(
        "Patched %d matrices (%d adapted HF weights) in %.0fs",
        len(patched),
        weights.adapted,
        time.time() - started,
    )

    files = {}
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        files[path.relative_to(out_dir).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "base_model": provenance.base_model,
        "base_revision": provenance.base_revision,
        "graph_revision": provenance.graph_revision,
        "adapter_sha256": provenance.adapter_sha256
        or (sha256(adapter_dir / "adapters.safetensors") if adapter_dir else ""),
        "dtype": "q4f16",
        "adapted_logical_matrices": weights.adapted,
        "patched_onnx_matrices": len(patched),
        "decoder_graph_unchanged": sha256(out_dir / GRAPH) == sha256(graph_path),
        "quantizer": {
            "runtime": "onnxruntime",
            "bits": 4,
            "block_size": BLOCK,
            "symmetric": symmetric,
            "fp16_input": as_fp16,
        },
        "merge": "HF BF16 -> FP16 -> FP32; add scale * A @ B in FP32; round FP16",
        "files": files,
        "patched": patched,
    }
    (out_dir / "a2ui-export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base_dir", type=Path, help="directory with the onnx-community export files")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--base-hf-dir", type=Path, required=True, help="directory with the BF16 model.safetensors"
    )
    parser.add_argument(
        "--adapter-dir", type=Path, help="mlx_lm adapter directory (omit to re-export the base)"
    )
    parser.add_argument(
        "--adapter-sha256", default="", help="recorded in the manifest (default: hash the adapter)"
    )
    parser.add_argument("--base-model", default="unsloth/gemma-4-E2B-it")
    parser.add_argument("--base-revision", default="d36bdd3855c82a6a1a23f7c459b749f5724ae75d")
    parser.add_argument("--graph-revision", default="6f21c99d7e18237f8d2ce92aadfdbfeba877f63a")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    manifest = patch(
        args.base_dir,
        args.out_dir,
        args.base_hf_dir,
        args.adapter_dir,
        Provenance(args.adapter_sha256, args.base_model, args.base_revision, args.graph_revision),
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "adapted_logical_matrices",
                    "patched_onnx_matrices",
                    "decoder_graph_unchanged",
                    "quantizer",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
