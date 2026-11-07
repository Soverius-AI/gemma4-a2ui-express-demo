import type { LocalModel, ModelAvailability, ModelInfo } from "./local-model";
import { E2B_RETRAIN, E4B_RETRAIN } from "./model-catalog";
import type { ModelMode } from "./model-catalog";
import { ChromeModel } from "./chrome/chrome-model";
import {
  TransformersModel,
  transformersModelRoot,
} from "./transformers/transformers-model";
import { WebLLMModel } from "./webllm/webllm-model";

/** The model the user selected, behind one `LocalModel`. Selection is
 * explicit: a model that fails to start never falls back to another. */
export class ModelSelector implements LocalModel {
  #mode: ModelMode = "base";
  readonly #models: Readonly<Record<ModelMode, LocalModel>>;

  constructor(models: Partial<Record<ModelMode, LocalModel>> = {}) {
    const root = transformersModelRoot();
    this.#models = {
      base: models.base ?? new TransformersModel(undefined, null),
      // The retrains, from their published revisions or from a loopback root
      // that serves the export of whichever option is selected.
      finetuned:
        models.finetuned ?? new TransformersModel(undefined, E4B_RETRAIN, root),
      "finetuned-e2b":
        models["finetuned-e2b"] ??
        new TransformersModel(undefined, E2B_RETRAIN, root),
      "webllm-e2b": models["webllm-e2b"] ?? new WebLLMModel(),
      chrome: models.chrome ?? new ChromeModel(),
    };
  }

  get mode(): ModelMode {
    return this.#mode;
  }

  get #selected(): LocalModel {
    return this.#models[this.#mode];
  }

  /** Releases every model, then selects `mode` without loading it. */
  select(mode: ModelMode) {
    this.destroy();
    this.#mode = mode;
  }

  get info(): ModelInfo {
    return this.#selected.info;
  }

  availability(): Promise<ModelAvailability> {
    return this.#selected.availability();
  }

  enable(
    system: string,
    signal: AbortSignal,
    progress: (value: number) => void,
  ) {
    return this.#selected.enable(system, signal, progress);
  }

  generate(input: string, system: string, signal: AbortSignal) {
    return this.#selected.generate(input, system, signal);
  }

  resetContext(signal: AbortSignal) {
    return this.#selected.resetContext(signal);
  }

  destroy() {
    for (const model of Object.values(this.#models)) model.destroy();
  }
}
