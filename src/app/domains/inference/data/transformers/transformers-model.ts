import { chatMessages } from "../chat-messages";
import type { LocalModel, ModelAvailability, ModelInfo } from "../local-model";
import { BASE_MODEL, E4B_FINE_TUNE } from "../model-catalog";
import type { FineTune } from "../model-catalog";
import { modelFilesRoot } from "../model-files";
import type { WorkerRequest, WorkerResponse } from "./transformers-protocol";

/** An explicit `?transformersModelRoot=` directory for the export, loopback
 * HTTP or HTTPS on this site only; undefined means the Hub. */
export function transformersModelRoot(
  requested = typeof location === "undefined"
    ? undefined
    : (new URLSearchParams(location.search).get("transformersModelRoot") ??
      undefined),
  origin = typeof location === "undefined"
    ? "http://localhost"
    : location.origin,
): string | undefined {
  return requested === undefined
    ? undefined
    : modelFilesRoot(requested, origin);
}

const createModelWorker = () =>
  new Worker(new URL("./transformers.worker", import.meta.url), {
    type: "module",
  });

/** Gemma 4 through Transformers.js in a module worker. The runtime and its
 * weights only enter the worker once the user enables the model. */
export class TransformersModel implements LocalModel {
  info: ModelInfo;
  #worker?: Worker;
  #system?: string;
  #nextId = 0;
  #pending?: (message: WorkerResponse) => void;

  /** `fineTune` null loads the untuned base model. */
  constructor(
    private readonly createWorker: () => Worker = createModelWorker,
    private readonly fineTune: FineTune | null = E4B_FINE_TUNE,
    private readonly modelRoot: string | undefined = undefined,
  ) {
    this.info = {
      backend: "transformers",
      model: fineTune?.hub.repoId ?? BASE_MODEL.repoId,
      device: "WebGPU",
    };
  }

  async availability(): Promise<ModelAvailability> {
    return typeof Worker === "undefined" || typeof WebAssembly === "undefined"
      ? "unsupported"
      : "downloadable";
  }

  async enable(
    system: string,
    signal: AbortSignal,
    progress: (value: number) => void,
  ) {
    this.destroy();
    signal.throwIfAborted();
    const worker = this.createWorker();
    this.#worker = worker;
    const id = ++this.#nextId;
    worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) =>
      this.#pending?.(data);
    worker.onerror = (event) => {
      if (this.#worker === worker)
        this.#pending?.({
          type: "error",
          id: this.#nextId,
          message: event.message || "Browser model worker failed.",
        });
    };
    try {
      await new Promise<void>((resolve, reject) => {
        const cleanup = () => {
          signal.removeEventListener("abort", abort);
          this.#pending = undefined;
        };
        const abort = () => {
          cleanup();
          this.destroy();
          reject(signal.reason);
        };
        this.#pending = (message) => {
          if (message.id !== id) return;
          if (message.type === "progress" || message.type === "ready") {
            this.info = message.info;
            progress(message.type === "ready" ? 1 : message.value);
          }
          if (message.type === "ready") {
            cleanup();
            resolve();
          }
          if (message.type === "error") {
            cleanup();
            reject(new Error(message.message));
          }
        };
        signal.addEventListener("abort", abort, { once: true });
        worker.postMessage({
          type: "load",
          id,
          ...(this.fineTune ? { fineTuneId: this.fineTune.exportId } : {}),
          ...(this.modelRoot ? { modelRoot: this.modelRoot } : {}),
        } satisfies WorkerRequest);
      });
      signal.throwIfAborted();
      this.#system = system;
    } catch (error) {
      if (this.#worker === worker) this.destroy();
      throw error;
    }
  }

  async *generate(input: string, system: string, signal: AbortSignal) {
    if (!this.#worker || this.#system !== system)
      throw new Error("Enable the browser model before generating.");
    signal.throwIfAborted();
    const id = ++this.#nextId;
    const queue: string[] = [];
    let done = false;
    let failure: Error | undefined;
    let wake: (() => void) | undefined;
    const abort = () => {
      this.destroy();
      wake?.();
    };
    this.#pending = (message) => {
      if (message.id !== id) return;
      if (message.type === "chunk") queue.push(message.text);
      if (message.type === "done") done = true;
      if (message.type === "error") {
        done = true;
        failure = new Error(message.message);
      }
      wake?.();
    };
    signal.addEventListener("abort", abort, { once: true });
    this.#worker.postMessage({
      type: "generate",
      id,
      messages: [{ role: "system", content: system }, ...chatMessages(input)],
    } satisfies WorkerRequest);
    try {
      while (true) {
        signal.throwIfAborted();
        if (failure) throw failure;
        if (queue.length) {
          yield queue.shift()!;
          continue;
        }
        if (done) break;
        await new Promise<void>((resolve) => {
          wake = resolve;
        });
      }
    } finally {
      signal.removeEventListener("abort", abort);
      this.#pending = undefined;
    }
  }

  /** Every generation sends the complete messages and starts with fresh KV
   * state, so there is no conversation state to clear. */
  async resetContext(signal: AbortSignal) {
    signal.throwIfAborted();
  }

  destroy() {
    this.#worker?.terminate();
    this.#worker = undefined;
    this.#system = undefined;
  }
}
