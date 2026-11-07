/// <reference lib="webworker" />
import {
  env,
  Gemma4Processor,
  pipeline,
  TextStreamer,
} from "@huggingface/transformers";
import type { TextGenerationPipeline } from "@huggingface/transformers";
import type { ModelInfo } from "../local-model";
import { BASE_MODEL, fineTuneById } from "../model-catalog";
import type { FineTune } from "../model-catalog";
import { hubModelFilesRoot } from "../model-files";
import {
  assertContextBudget,
  LOCAL_CONTEXT_WINDOW,
  MAX_NEW_TOKENS,
} from "../model-limits";
import type { WorkerRequest, WorkerResponse } from "./transformers-protocol";

env.allowLocalModels = false;
env.backends.onnx.wasm!.numThreads = 1;

/** Gemma 4's template variables, which the typed options do not declare. */
const TEMPLATE_VARIABLES = { enable_thinking: false };

let generator: TextGenerationPipeline | undefined;
/** Set when the tokenizer has no chat template of its own. */
let chatTemplate: string | undefined;
const send = (message: WorkerResponse) => self.postMessage(message);

async function assertHalfPrecisionGpu() {
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
  if (!adapter?.features.has("shader-f16"))
    throw new Error(
      "Gemma 4 requires WebGPU with half precision. No substitute model was loaded; Chrome Prompt API can be selected separately.",
    );
}

/** A published repository must serve the adapter the app pins. */
async function assertExportManifest(fineTune: FineTune, root: string) {
  const response = await fetch(`${root}a2ui-export-manifest.json`);
  if (!response.ok)
    throw new Error("The selected fine-tuned model files are unavailable.");
  const artifact = await response.json();
  if (artifact.adapter_sha256 !== fineTune.adapterSha256)
    throw new Error(
      "The model server did not return the selected Express fine-tune. No substitute model was loaded.",
    );
  if (
    artifact.base_model !== fineTune.baseModel ||
    artifact.dtype !== "q4f16" ||
    artifact.patched_onnx_matrices !== fineTune.patchedMatrices ||
    !artifact.decoder_graph_unchanged
  )
    throw new Error(
      "The export manifest does not match the verified Gemma browser format.",
    );
}

async function load(
  id: number,
  fineTuneId: string | undefined,
  modelRoot: string | undefined,
) {
  await assertHalfPrecisionGpu();
  const fineTune = fineTuneById(fineTuneId);
  if (fineTuneId !== undefined && !fineTune)
    throw new Error("Unknown Express fine-tune.");
  const { repoId: model, revision } = fineTune?.hub ?? BASE_MODEL;
  // A loopback root serves the files directly, for an export under test.
  if (modelRoot) {
    env.remoteHost = modelRoot;
    env.remotePathTemplate = "";
  }
  const root = modelRoot ?? hubModelFilesRoot(fineTune?.hub ?? BASE_MODEL);
  const info: ModelInfo = {
    backend: "transformers",
    model,
    device: "webgpu",
    contextWindowTokens: LOCAL_CONTEXT_WINDOW,
    quantization: "q4f16",
  };
  if (fineTune) {
    await assertExportManifest(fineTune, root);
    info.artifactSha256 = fineTune.adapterSha256;
  }
  send({ type: "progress", id, value: 0, info });
  // For a multimodal checkpoint such as Gemma 4, the text-generation
  // pipeline downloads and runs only the text sessions.
  generator = await pipeline("text-generation", model, {
    device: "webgpu",
    dtype: "q4f16",
    revision,
    progress_callback: (event) => {
      if (event.status === "progress_total")
        send({ type: "progress", id, value: event.progress / 100, info });
    },
  });
  // Gemma 4 can keep its template in chat_template.jinja, which only its
  // processor reads; that adds small JSON files, no weights.
  chatTemplate = generator.tokenizer.chat_template
    ? undefined
    : ((await Gemma4Processor.from_pretrained(model, { revision }))
        .chat_template ?? undefined);
  send({ type: "ready", id, info });
}

async function generate(
  id: number,
  messages: Extract<WorkerRequest, { type: "generate" }>["messages"],
) {
  if (!generator) throw new Error("The browser model is not loaded.");
  // The pipeline's chat call hardcodes truncation in 4.3.2. Tokenize here
  // and call the loaded model directly, so the budget covers the exact
  // complete prompt and nothing is truncated.
  const inputs = generator.tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: true,
    return_tensor: true,
    return_dict: true,
    truncation: false,
    ...(chatTemplate ? { chat_template: chatTemplate } : {}),
    ...TEMPLATE_VARIABLES,
  });
  assertContextBudget(inputs.input_ids.dims.at(-1) ?? NaN);
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    callback_function: (text) => send({ type: "chunk", id, text }),
  });
  // Greedy decoding keeps sampled variation out of strict DSL output. Each
  // call gets new input tensors and fresh generation-local KV state; only
  // weights and ONNX sessions are reused, never another conversation.
  await generator.model.generate({
    ...inputs,
    max_new_tokens: MAX_NEW_TOKENS,
    do_sample: false,
    streamer,
  });
  send({ type: "done", id });
}

self.addEventListener(
  "message",
  async ({ data }: MessageEvent<WorkerRequest>) => {
    try {
      if (data.type === "load")
        await load(data.id, data.fineTuneId, data.modelRoot);
      else await generate(data.id, data.messages);
    } catch (error) {
      send({
        type: "error",
        id: data.id,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  },
);
