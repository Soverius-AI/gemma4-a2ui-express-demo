"""Packaging must not label mutated or untuned shards as the verified export."""

import hashlib
import json

import pytest

from a2ui_model_tools import identities as pins
from a2ui_model_tools.package import descriptor, verify_conversion, verify_shards

NAME = "params_shard_0.bin"
DATA = b"verified fine-tuned bytes"


@pytest.fixture
def model(tmp_path):
    (tmp_path / NAME).write_bytes(DATA)
    (tmp_path / "tensor-cache.json").write_text(json.dumps({"records": [{"dataPath": NAME}]}))
    provenance = {
        "shards": [{"file": NAME, "nbytes": len(DATA), "output_sha256": hashlib.sha256(DATA).hexdigest()}]
    }
    return tmp_path, provenance


def test_exact_shards_pass(model):
    verify_shards(*model)


def test_rejects_mutated_weights(model):
    root, provenance = model
    (root / NAME).write_bytes(b"x" * len(DATA))
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_shards(root, provenance)


def test_rejects_extra_shard(model):
    root, provenance = model
    (root / "params_shard_1.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="inventory changed"):
        verify_shards(root, provenance)


def test_rejects_duplicate_inventory(model):
    root, provenance = model
    provenance["shards"] *= 2
    with pytest.raises(ValueError, match="inventory changed"):
        verify_shards(root, provenance)


def test_rejects_conversion_of_another_adapter(model):
    root, provenance = model
    provenance |= {"completion": "complete", "adapter_sha256": "0" * 64}
    with pytest.raises(ValueError, match="does not prove"):
        verify_conversion(root, provenance, pins.ADAPTER_SHA256)


def test_descriptor_keys_the_app_checks():
    provenance = {"merged_hf_sha256": "m", "interface_id": "i", "parameter_schema_id": "p"}
    result = descriptor(provenance, [{"name": NAME, "bytes": 3, "sha256": "s"}], pins.ADAPTER_SHA256)
    # Mirrors WEBLLM_E2B_MANIFEST in src/app/.../webllm-manifest.ts.
    assert {
        key: result[key]
        for key in ("schemaVersion", "adapterSha256", "baseModel", "baseRevision", "quantization", "modelId")
    } == {
        "schemaVersion": 1,
        "adapterSha256": "54c7414491e36610133c5f22e0397a66e97d901e3b119238abf50042dc7aa323",
        "baseModel": "unsloth/gemma-4-E2B-it",
        "baseRevision": "d36bdd3855c82a6a1a23f7c459b749f5724ae75d",
        "quantization": "q4f16_1",
        "modelId": "gemma-4-e2b-express-54c7414491e3-q4f16_1-MLC",
    }
    assert result["weightsBytes"] == 3
