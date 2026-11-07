/// <reference types="@types/dom-chromium-ai" />

/** Chrome's `Availability`, plus `unsupported` for a browser that lacks the
 * runtime altogether. */
export type ModelAvailability = Availability | "unsupported";

export type ModelBackend = "chrome" | "transformers" | "webllm";

export interface ModelInfo {
  backend: ModelBackend;
  model: string;
  device: string;
  /** The adapter the loaded files declare, once verified. */
  artifactSha256?: string;
  contextWindowTokens?: number;
  quantization?: string;
}

/** A model that runs in the browser. The agent owns the conversation; a
 * model receives the complete transcript with every generation. */
export interface LocalModel {
  readonly info: ModelInfo;
  availability(): Promise<ModelAvailability>;
  enable(
    system: string,
    signal: AbortSignal,
    progress: (value: number) => void,
  ): Promise<void>;
  generate(
    input: string,
    system: string,
    signal: AbortSignal,
  ): AsyncIterable<string>;
  /** Clear conversation state while keeping loaded weights (idle only). */
  resetContext(signal: AbortSignal): Promise<void>;
  destroy(): void;
}
