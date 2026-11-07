# Model tools

Python tooling behind the demo's optional WebLLM model: it merges the Express
LoRA into Gemma 4 E2B and converts the result for WebLLM.

| Command    | Does                                                                                                                                                                                   |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `download` | Fetches the base model, the published adapter, the official MLC weights' metadata, licenses and the WebGPU runtime. Each file is checked against its pinned SHA-256.                   |
| `merge`    | Merges the LoRA adapter into the original BF16 checkpoint (BF16 → F16 → F32 + scale·BA → F16). Takes `mlx_lm` and PEFT/Unsloth adapters.                                               |
| `convert`  | Produces MLC `q4f16_1` weights by requantizing only the adapted matrices of the official MLC conversion. It first checks that the original base reproduces the official bytes exactly. |
| `package`  | Adds the tokenizer, runtime, licenses and the `a2ui-mlc-manifest.json` descriptor that the app checks before it loads anything.                                                        |

Every pinned identity (repositories, revisions, SHA-256s, expected tensor
counts) lives in [`src/a2ui_model_tools/identities.py`](src/a2ui_model_tools/identities.py).
The tools refuse inputs that don't match, and never overwrite existing outputs.
No outputs are uploaded or committed.

## Setup

Python 3.12+. From the repository root:

```sh
python3 -m venv .venv-models
.venv-models/bin/pip install -e 'model-tools[dev]'
.venv-models/bin/pytest model-tools
```

## Build the WebLLM model

The pipeline needs about 30 GB in its work directory. The default is
`models/webllm-e2b` (git-ignored); set `--work` or `A2UI_MODEL_WORK` to put it
on another drive.

```sh
export A2UI_MODEL_WORK="$PWD/models/webllm-e2b"
a2ui-model-tools download   # base, adapter, MLC metadata, licenses, WASM runtime
a2ui-model-tools merge      # -> $A2UI_MODEL_WORK/merged-hf
a2ui-model-tools convert    # -> $A2UI_MODEL_WORK/mlc-model (downloads 84 official shards)
a2ui-model-tools package    # completes mlc-model in place
```

The merged checkpoint must hash to `a2ff7800…70fd`, and `convert` refuses any
other value. Each step writes a provenance file next to its output
(`merge-provenance.json`, `conversion-provenance.json`), so keep those with the
model.

Serve the model from a production build:

```sh
pnpm build
pnpm model:stage "$A2UI_MODEL_WORK/mlc-model"   # symlinks into dist/; rerun after every build
pnpm preview
```

Then pick the WebLLM option in the app. It needs WebGPU with `shader-f16`. On
another host, deploy the real files under `/models/webllm-e2b-express/resolve/local/`,
or point `?webllmModelRoot=` at an HTTPS directory ending in `/resolve/<revision>/`.

## Export a retrained adapter for Transformers.js

The app's default fine-tuned options load q4f16 ONNX exports. `onnx_patch`
requantizes the adapted matrices of an `mlx_lm` adapter into the matching
`onnx-community` export, merging on the fly from the BF16 base (the same recipe
as `merge`), and writes `a2ui-export-manifest.json` for the app's check. It
calibrates the quantizer against the base export first and refuses to patch if
the scheme does not match.

```sh
.venv-models/bin/pip install -e 'model-tools[onnx]'
# base export: the onnx-community graph, data and tokenizer files at the pinned revision
.venv-models/bin/python -m a2ui_model_tools.onnx_patch <base-export-dir> <out-dir> \
  --base-hf-dir "$A2UI_MODEL_WORK/base-hf" --adapter-dir training/runs/e2b/adapter
```

For E4B pass `--base-model unsloth/gemma-4-E4B-it`, its `--base-revision` and
the export's `--graph-revision`. Serve the output directory with CORS on a
loopback port and open the app with `?transformersModelRoot=http://localhost:<port>/`
to test it before uploading; then pin the uploaded revision in
`src/app/domains/inference/data/model-catalog.ts`.

## Use a retrained adapter

[`../training`](../training/README.md) retrains the LoRA. `merge` takes its
PEFT adapter as well as an `mlx_lm` one; pass the new hashes along the
pipeline:

```sh
a2ui-model-tools merge   --adapter-dir training/runs/e2b/adapter --expected-adapter-sha256 <adapter sha>
a2ui-model-tools convert --expected-merged-sha256 <merged sha from merge-provenance.json>
a2ui-model-tools package --expected-adapter-sha256 <adapter sha>
```

Shipping it in the app also means updating the adapter identity pinned in
`src/app/domains/inference/data/model-catalog.ts` (and the model id in
`webllm/webllm-manifest.ts`) and in `stage-webllm.mjs`.

## Licenses

The tools are MIT, like the rest of the repository. Packaging keeps the MLC base
`LICENSE` (Apache-2.0), the model notice and TVM's `LICENSE`/`NOTICE` beside the
weights. Model weights keep their upstream terms; see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
