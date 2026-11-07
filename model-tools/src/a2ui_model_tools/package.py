"""Finish a converted model as a WebLLM model directory.

Adds the chat config, tokenizer, licenses, notices and WASM runtime beside the
converted weights, then writes ``a2ui-mlc-manifest.json``, the descriptor the
app checks before it loads anything. Nothing is uploaded and no weights change.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from . import identities as pins
from .checkpoint_io import sha256

log = logging.getLogger(__name__)

MANIFEST_FILE = "a2ui-mlc-manifest.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_NOTICE = REPOSITORY_ROOT / "notices" / "GEMMA-NOTICE.txt"
COPIED_ASSETS = (
    "mlc-chat-config.json",
    "mlc-model-manifest.json",
    "mlc-build-info.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "LICENSE",
    "TVM-LICENSE.txt",
    "TVM-NOTICE.txt",
)


def verify_shards(root: Path, provenance: dict[str, Any]) -> None:
    """Reject post-conversion mutations, extra/missing shards and path escapes."""
    expected = {entry["file"]: entry for entry in provenance["shards"]}
    cache = json.loads((root / "tensor-cache.json").read_text())
    listed = {entry["dataPath"] for entry in cache["records"]}
    present = {path.name for path in root.glob("*.bin")}
    if len(expected) != len(provenance["shards"]) or listed != set(expected) or present != listed:
        raise ValueError("Shard inventory changed after conversion")
    for name, entry in expected.items():
        if Path(name).name != name or not name.startswith("params_shard_"):
            raise ValueError("Unsafe shard path")
        path = root / name
        if path.stat().st_size != entry["nbytes"] or sha256(path) != entry["output_sha256"]:
            raise ValueError(f"Converted shard identity mismatch: {name}")


def verify_conversion(root: Path, provenance: dict[str, Any], adapter_sha256: str) -> None:
    proves_fine_tune = (
        provenance.get("completion") == "complete"
        and provenance.get("adapter_sha256") == adapter_sha256
        and provenance.get("base_hf_sha256") == pins.BASE_WEIGHTS_SHA256
        and provenance.get("adapted_logical_matrices") == pins.ADAPTER_MODULES
        and provenance.get("target_groups") == pins.MLC_TARGET_GROUPS
        and provenance.get("patched_arrays") == pins.MLC_PATCHED_ARRAYS
        and provenance.get("all_affected_base_bytes_exact") is True
        and provenance.get("all_unaffected_output_bytes_exact") is True
    )
    if not proves_fine_tune:
        raise ValueError("Conversion does not prove the exact expected Express fine-tune")
    if sha256(root / "tensor-cache.json") != provenance["output_tensor_cache_sha256"]:
        raise ValueError("Converted tensor metadata changed")
    verify_shards(root, provenance)


def verify_runtime(assets: Path, wasm: Path, provenance: dict[str, Any]) -> None:
    if sha256(wasm) != pins.WASM_SHA256:
        raise ValueError("Unexpected compiled Gemma 4 E2B library")
    config = json.loads((assets / "mlc-chat-config.json").read_text())
    text = config["model_config"]["text_config"]
    if not (
        config["model_type"] == "gemma4"
        and config["quantization"] == pins.QUANTIZATION
        and text["hidden_size"] == pins.MLC_HIDDEN_SIZE
        and text["num_hidden_layers"] == pins.MLC_LAYERS
        and config["prefill_chunk_size"] == pins.PREFILL_CHUNK_SIZE
    ):
        raise ValueError("Wrong architecture or prefill configuration")
    manifest = json.loads((assets / "mlc-model-manifest.json").read_text())
    if (
        manifest["interface_id"] != provenance["interface_id"]
        or manifest["weights"]["parameter_schema_id"] != provenance["parameter_schema_id"]
    ):
        raise ValueError("Parameter schema/library interface mismatch")


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name not in (MANIFEST_FILE, "README.md")
    ]


def descriptor(provenance: dict[str, Any], files: list[dict[str, Any]], adapter_sha256: str) -> dict:
    # The app checks schemaVersion through modelId; keep those keys stable.
    return {
        "schemaVersion": 1,
        "adapterSha256": adapter_sha256,
        "baseModel": pins.BASE.repo_id,
        "baseRevision": pins.BASE.revision,
        "quantization": pins.QUANTIZATION,
        "modelId": pins.webllm_model_id(adapter_sha256),
        "baseWeightsSha256": pins.BASE_WEIGHTS_SHA256,
        "mergedHfSha256": provenance["merged_hf_sha256"],
        "runtimeContextWindowSize": pins.RUNTIME_CONTEXT_WINDOW,
        "compiledPrefillChunkSize": pins.PREFILL_CHUNK_SIZE,
        "interfaceId": provenance["interface_id"],
        "parameterSchemaId": provenance["parameter_schema_id"],
        "modelLibSha256": pins.WASM_SHA256,
        "weightsBytes": sum(f["bytes"] for f in files if f["name"].endswith(".bin")),
        "files": files,
        "conversion": (
            "275 logical adapter matrices; 240 MLC groups; 445 arrays. Every affected original-base "
            "array matched official MLC bytes exactly before replacement. All unaffected bytes preserved."
        ),
        "note": (
            "Same adapter/base as the ONNX demo, different quantization/runtime. No claim of "
            "numerically identical inference. Runtime 32k context is an explicit SDK override; "
            "never truncate the Express prompt."
        ),
    }


def readme(model_id: str, adapter_sha256: str) -> str:
    return (
        "# Gemma 4 E2B Express fine-tune for WebLLM\n\n"
        f"Model: `{model_id}`. Adapter SHA-256: `{adapter_sha256}`.\n\n"
        "The Express LoRA adapter merged into its original Hugging Face base and stored as MLC "
        "q4f16_1. Quantization differs from the ONNX q4f16 export, so outputs are not "
        "numerically identical.\n\n"
        "conversion-provenance.json, wasm-provenance.json and a2ui-mlc-manifest.json record the "
        "identities, parity checks, source commits and file hashes. BASE_MODEL_README.md is the "
        "original base model card; its licenses and notices apply.\n"
    )


def package(
    model_dir: Path,
    assets_dir: Path,
    wasm: Path,
    wasm_provenance: Path,
    *,
    adapter_sha256: str = pins.ADAPTER_SHA256,
) -> dict[str, Any]:
    """Complete ``model_dir`` in place and return its descriptor."""
    root, assets = Path(model_dir).resolve(), Path(assets_dir).resolve()
    provenance = json.loads((root / "conversion-provenance.json").read_text())
    verify_conversion(root, provenance, adapter_sha256)
    verify_runtime(assets, Path(wasm), provenance)

    for name in COPIED_ASSETS:
        shutil.copyfile(assets / name, root / name)
    shutil.copyfile(assets / "README.md", root / "BASE_MODEL_README.md")
    shutil.copyfile(wasm, root / "model.wasm")
    shutil.copyfile(wasm_provenance, root / "wasm-provenance.json")
    shutil.copyfile(MODEL_NOTICE, root / "NOTICE.txt")

    files = inventory(root)
    result = descriptor(provenance, files, adapter_sha256)
    (root / MANIFEST_FILE).write_text(json.dumps(result, indent=2) + "\n")
    (root / "README.md").write_text(readme(result["modelId"], adapter_sha256))
    log.info("Packaged %d files, %s weight bytes at %s", len(files), f"{result['weightsBytes']:,}", root)
    return result
