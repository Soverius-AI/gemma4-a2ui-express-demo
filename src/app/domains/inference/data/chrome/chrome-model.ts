/// <reference types="@types/dom-chromium-ai" />
import { chatMessages } from "../chat-messages";
import type { LocalModel, ModelAvailability } from "../local-model";

const OPTIONS: LanguageModelCreateCoreOptions = {
  expectedInputs: [{ type: "text", languages: ["en"] }],
  expectedOutputs: [{ type: "text", languages: ["en"] }],
};

/** Chrome's built-in model through the Prompt API. The browser owns the
 * model; the agent owns the conversation history. */
export class ChromeModel implements LocalModel {
  readonly info = {
    backend: "chrome" as const,
    model: "Chrome built-in model",
    device: "Chrome Prompt API",
  };
  #base?: LanguageModel;
  #systemPrompt?: string;

  async availability(): Promise<ModelAvailability> {
    if (typeof LanguageModel === "undefined") return "unsupported";
    return LanguageModel.availability(OPTIONS);
  }

  async enable(
    systemPrompt: string,
    signal: AbortSignal,
    progress: (value: number) => void,
  ) {
    this.destroy();
    if (typeof LanguageModel === "undefined")
      throw new Error(
        "Chrome's Prompt API is not available. Open this demo in an eligible desktop Chrome browser on localhost or HTTPS.",
      );
    const session = await LanguageModel.create({
      ...OPTIONS,
      initialPrompts: [{ role: "system", content: systemPrompt }],
      signal,
      monitor(monitor) {
        monitor.addEventListener("downloadprogress", (event) =>
          progress(event.loaded),
        );
      },
    });
    if (signal.aborted) {
      session.destroy();
      signal.throwIfAborted();
    }
    this.#base = session;
    this.#systemPrompt = systemPrompt;
  }

  async *generate(input: string, systemPrompt: string, signal: AbortSignal) {
    if (!this.#base || this.#systemPrompt !== systemPrompt)
      throw new Error(
        "Enable the local model for the current catalog before generating.",
      );
    // A clone starts with just the system contract. The agent supplies the
    // history once, so nothing accumulates in a long-lived session.
    const session = await this.#base.clone({ signal });
    try {
      const reader = session
        .promptStreaming(chatMessages(input), { signal })
        .getReader();
      try {
        while (true) {
          const { value, done } = await reader.read();
          signal.throwIfAborted();
          if (done) break;
          yield value;
        }
      } finally {
        reader.releaseLock();
      }
    } finally {
      session.destroy();
    }
  }

  /** The base session stays a clean seed: generation only prompts clones. */
  async resetContext(signal: AbortSignal) {
    signal.throwIfAborted();
  }

  destroy() {
    this.#base?.destroy();
    this.#base = undefined;
    this.#systemPrompt = undefined;
  }
}
