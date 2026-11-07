import type { WebWorkerMLCEngine } from "@mlc-ai/web-llm";
import { chatMessages } from "../chat-messages";
import type { LocalModel, ModelAvailability, ModelInfo } from "../local-model";
import { LOCAL_CONTEXT_WINDOW, MAX_NEW_TOKENS } from "../model-limits";
import {
  verifyWebLLMManifest,
  WEBLLM_E2B_MANIFEST,
  WEBLLM_MANIFEST_FILE,
  webllmModelRoot,
} from "./webllm-manifest";

export const WEBLLM_CONTEXT = LOCAL_CONTEXT_WINDOW;
export const WEBLLM_MAX_TOKENS = MAX_NEW_TOKENS;
/** Compiled into model.wasm; the SDK has no runtime override. */
export const WEBLLM_PREFILL_CHUNK_SIZE = 512;

const createEngineWorker = () =>
  new Worker(new URL("./webllm.worker", import.meta.url), { type: "module" });

/** The E2B Express fine-tune converted to MLC, through WebLLM. No SDK
 * import, worker, fetch or weight download happens before `enable`. */
export class WebLLMModel implements LocalModel {
  #verified = false;
  #worker?: Worker;
  #engine?: WebWorkerMLCEngine;
  #system?: string;
  #cancel?: (reason: unknown) => void;

  constructor(
    private readonly createWorker: () => Worker = createEngineWorker,
    private readonly loadSDK = () => import("@mlc-ai/web-llm"),
    private readonly root = () => webllmModelRoot(),
    private readonly fetchManifest: typeof fetch = (...args) => fetch(...args),
  ) {}

  get info(): ModelInfo {
    return {
      backend: "webllm",
      model: WEBLLM_E2B_MANIFEST.modelId,
      device: "WebGPU",
      contextWindowTokens: WEBLLM_CONTEXT,
      quantization: "q4f16_1",
      ...(this.#verified
        ? { artifactSha256: WEBLLM_E2B_MANIFEST.adapterSha256 }
        : {}),
    };
  }

  async availability(): Promise<ModelAvailability> {
    if (
      typeof Worker === "undefined" ||
      typeof WebAssembly === "undefined" ||
      typeof navigator === "undefined" ||
      !navigator.gpu
    )
      return "unsupported";
    const adapter = await navigator.gpu.requestAdapter();
    return adapter?.features.has("shader-f16") ? "downloadable" : "unavailable";
  }

  async enable(
    system: string,
    signal: AbortSignal,
    progress: (value: number) => void,
  ) {
    this.destroy();
    const operation = this.#operation(signal);
    try {
      const root = this.root();
      const response = await operation.wait(
        this.fetchManifest(new URL(WEBLLM_MANIFEST_FILE, root), {
          signal,
          cache: "no-store",
        }),
      );
      if (!response.ok)
        throw new Error(
          `WebLLM provenance manifest unavailable (${response.status}).`,
        );
      verifyWebLLMManifest(await operation.wait(response.json()));
      operation.check();
      this.#verified = true;
      const sdk = await operation.wait(this.loadSDK());
      operation.check();
      const worker = this.createWorker();
      this.#worker = worker;
      worker.onerror = () => {
        if (this.#worker === worker) this.destroy();
      };
      const engine = await operation.wait(
        sdk.CreateWebWorkerMLCEngine(
          worker,
          WEBLLM_E2B_MANIFEST.modelId,
          {
            appConfig: {
              model_list: [
                {
                  model: root,
                  model_id: WEBLLM_E2B_MANIFEST.modelId,
                  model_lib: new URL("model.wasm", root).href,
                  model_manifest: "mlc-model-manifest.json",
                  required_features: ["shader-f16"],
                  buffer_size_required_bytes: 201326592,
                  vram_required_MB: 3387,
                },
              ],
            },
            initProgressCallback: ({ progress: value }) => {
              if (this.#worker === worker) progress(value);
            },
          },
          { context_window_size: WEBLLM_CONTEXT },
        ),
      );
      operation.check();
      this.#engine = engine;
      this.#system = system;
      progress(1);
    } catch (error) {
      if (operation.active()) this.destroy();
      throw error;
    } finally {
      operation.cleanup();
    }
  }

  async *generate(input: string, system: string, signal: AbortSignal) {
    if (!this.#engine || this.#system !== system)
      throw new Error("Enable the WebLLM model before generating.");
    const operation = this.#operation(signal);
    let completed = false;
    try {
      const stream = await operation.wait(
        this.#engine.chat.completions.create({
          messages: [
            { role: "system", content: system },
            ...chatMessages(input),
          ],
          temperature: 0,
          max_tokens: WEBLLM_MAX_TOKENS,
          stream: true,
          stream_options: { include_usage: true },
        }),
      );
      const iterator = stream[Symbol.asyncIterator]();
      while (true) {
        const result = await operation.wait(iterator.next());
        operation.check();
        if (result.done) break;
        const chunk = result.value;
        if (
          chunk.usage &&
          chunk.usage.prompt_tokens + WEBLLM_MAX_TOKENS > WEBLLM_CONTEXT
        )
          throw new Error(
            "Actual WebLLM prompt exceeds the 32k context budget; no prompt truncation is permitted.",
          );
        const text = chunk.choices[0]?.delta.content;
        if (text) yield text;
      }
      completed = true;
    } finally {
      const active = operation.active();
      operation.cleanup();
      if (!completed && active) this.destroy();
    }
  }

  async resetContext(signal: AbortSignal) {
    if (!this.#engine)
      throw new Error("Enable the WebLLM model before resetting context.");
    const operation = this.#operation(signal);
    try {
      await operation.wait(this.#engine.resetChat());
      operation.check();
    } catch (error) {
      if (operation.active()) this.destroy();
      throw error;
    } finally {
      operation.cleanup();
    }
  }

  destroy() {
    this.#verified = false;
    this.#cancel?.(new DOMException("WebLLM stopped.", "AbortError"));
    this.#cancel = undefined;
    // Best-effort unload; terminating the worker guarantees Stop ends the
    // computation.
    void this.#engine?.unload().catch(() => {});
    this.#worker?.terminate();
    this.#worker = undefined;
    this.#engine = undefined;
    this.#system = undefined;
  }

  /** Races SDK promises against cancellation: terminating the SDK's worker
   * alone leaves its RPC promises pending. */
  #operation(signal: AbortSignal) {
    signal.throwIfAborted();
    if (this.#cancel) throw new Error("WebLLM operation already in progress.");
    let failure: unknown;
    let cancelled = false;
    let reject!: (reason: unknown) => void;
    const cancellation = new Promise<never>((_, fail) => {
      reject = fail;
    });
    // An abort between awaited operations must not be an unhandled rejection.
    void cancellation.catch(() => {});
    const cancel = (reason: unknown) => {
      cancelled = true;
      failure = reason;
      reject(reason);
    };
    this.#cancel = cancel;
    const abort = () => {
      cancel(signal.reason);
      this.destroy();
    };
    signal.addEventListener("abort", abort, { once: true });
    return {
      wait: <T>(promise: Promise<T>): Promise<T> =>
        Promise.race([promise, cancellation]),
      active: () => this.#cancel === cancel,
      check: () => {
        if (cancelled) throw failure;
        signal.throwIfAborted();
      },
      cleanup: () => {
        signal.removeEventListener("abort", abort);
        if (this.#cancel === cancel) this.#cancel = undefined;
      },
    };
  }
}
