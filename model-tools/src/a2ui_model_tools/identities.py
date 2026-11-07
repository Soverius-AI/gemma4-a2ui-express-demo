"""Pinned identities of every input and output the tools accept.

Each tool checks its inputs against these values and refuses anything else, so
a merge, conversion or package can only describe the published fine-tune.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HubRepo:
    """A Hugging Face repository at an immutable revision."""

    repo_id: str
    revision: str

    def url(self, path: str) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{path}"


@dataclass(frozen=True, slots=True)
class GitHubFile:
    """A file in a GitHub repository at an immutable commit."""

    repository: str
    commit: str
    path: str

    @property
    def url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repository}/{self.commit}/{self.path}"


# Original BF16 base. Byte-identical to google/gemma-4-E2B-it at 3e22461f65e8.
BASE = HubRepo("unsloth/gemma-4-E2B-it", "d36bdd3855c82a6a1a23f7c459b749f5724ae75d")
BASE_WEIGHTS_SHA256 = "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
BASE_FILES = {
    "model.safetensors": BASE_WEIGHTS_SHA256,
    "config.json": "266f8edff7c9100a8877332f31386303bb0ed7a043dfdb7ec3b961f261604cb1",
    "generation_config.json": "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "950772223ebcf6247759538794ee2909b7335733c04a3466d25f0e4fc808a467",
    "chat_template.jinja": "241c50d86bdfe5e43307da87f559cd2416aacd67a8de46c15acc0105ef2200b7",
    "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
}

# Published E2B Express adapter: checkpoint 540 of the 2026-11-07 Unsloth retrain
# (mlx_lm layout), as uploaded with the ONNX export.
ADAPTER = HubRepo("soverius-ai/gemma-4-E2B-it-A2UI-Express-ONNX", "a50ff5d2695da98bcd5ea44a40854602b97f22c7")
ADAPTER_SUBDIR = "adapter-mlx"
ADAPTER_SHA256 = "54c7414491e36610133c5f22e0397a66e97d901e3b119238abf50042dc7aa323"
ADAPTER_FILES = {
    "adapter_config.json": "043fc41655bb881bcd5668e0c0421b7a4d146ea43b65ea9ce3543f2303fc9dd4",
    "adapters.safetensors": ADAPTER_SHA256,
}
ADAPTER_RANK = 8
ADAPTER_SCALE = 2.0
# Logical LoRA matrices: 35 layers x (q, o, gate, up, down, per-layer gate and
# projection) plus k and v on the 15 layers that own their KV cache.
ADAPTER_MODULES = 275

# Merge of ADAPTER into BASE: BF16 -> F16 -> F32 + scale * delta -> F16.
MERGED_WEIGHTS_SHA256 = "a2ff7800d341b16430e67496b1c2dc7ff7207a2406c8b81c6d12722bb4d470fd"

# Official MLC conversion of BASE; unaffected bytes are reused verbatim.
MLC_BASE = HubRepo("mlc-ai/gemma-4-E2B-it-q4f16_1-MLC", "633df598b7a473759367855273c63912c9e2bc9a")
MLC_LLM_COMMIT = "42fa855891fcb5825bf0911047d7c91e8234abd1"
MLC_METADATA_FILES = {
    "tensor-cache.json": "fb8299c6571251760931dc2080a53018dd987e3dbdc47e541a57c26275d018ef",
    "mlc-build-info.json": "d85af3e4a58b619555ba0c109a629e65528a5093543fe9b6953b4fdbc4c276ca",
    "mlc-model-manifest.json": "d81d77f9bff4c8271499b10c3dc2ce3a1abbf4df0c42442c289f541fd2abcf7a",
}
# Shipped with the packaged model, unchanged.
MLC_ASSET_FILES = {
    "README.md": "3ee64096d6f7855c917424cebdc83d5dd616f6367487994f8e142df40e0375f8",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633",
    "mlc-chat-config.json": "7dfac96b8fccab749f888fcecf9540e3a138e6c7e72df2623df280a65a658025",
    "mlc-model-manifest.json": "d81d77f9bff4c8271499b10c3dc2ce3a1abbf4df0c42442c289f541fd2abcf7a",
    "mlc-build-info.json": "d85af3e4a58b619555ba0c109a629e65528a5093543fe9b6953b4fdbc4c276ca",
}
QUANTIZATION = "q4f16_1"
# 275 logical matrices fuse gate/up into 240 MLC groups: 205 q4 weight/scale
# pairs and 35 raw F16 per-layer input gates.
MLC_TARGET_GROUPS = 240
MLC_PATCHED_ARRAYS = 445
MLC_HIDDEN_SIZE = 1536
MLC_LAYERS = 35
PREFILL_CHUNK_SIZE = 512
RUNTIME_CONTEXT_WINDOW = 32_768

WASM = GitHubFile(
    "mlc-ai/binary-mlc-llm-libs",
    "146ed9cd4b4062e967b017fd2291f7106b50599f",
    "web-llm-models/v0_2_87/base/gemma-4-E2B-it-q4f16_1_cs512-webgpu.wasm",
)
WASM_SHA256 = "6e9cdce823ca2872184146c9b650f0dcfcd3fb114d79c2e34afa4c20f73ce4d2"
WASM_BYTES = 12_053_022

# TVM source the WASM runtime was built from; its LICENSE and NOTICE ship with it.
TVM_COMMIT = "be03dc94b001269b88ae010d2ac352256be55579"
TVM_FILES = {
    "TVM-LICENSE.txt": (
        GitHubFile("akaashrp/tvm", TVM_COMMIT, "LICENSE"),
        "e62264b483aca7a9f8c1ac69388b476971b43975ced90ac11dbd9e9c530edb46",
    ),
    "TVM-NOTICE.txt": (
        GitHubFile("akaashrp/tvm", TVM_COMMIT, "NOTICE"),
        "967cd2fb97712c43271e2b1c4f636c4d8823898869f0f76f65bf8a3232584f0f",
    ),
}


def webllm_model_id(adapter_sha256: str = ADAPTER_SHA256) -> str:
    return f"gemma-4-e2b-express-{adapter_sha256[:12]}-{QUANTIZATION}-MLC"


MODEL_ID = webllm_model_id()
