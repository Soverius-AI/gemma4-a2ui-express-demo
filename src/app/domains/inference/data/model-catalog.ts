import type { HubModelSource } from "./model-files";

/** Every model the app can run, by the mode that selects it. */
export const MODEL_MODES = [
  "base",
  "webllm-e2b",
  "finetuned",
  "finetuned-e2b",
  "chrome",
] as const;
export type ModelMode = (typeof MODEL_MODES)[number];

export const isModelMode = (value: string): value is ModelMode =>
  (MODEL_MODES as readonly string[]).includes(value);

/** Gemma 4 E4B instruction-tuned, the default. The text-generation pipeline
 * loads only its text sessions (embed_tokens, decoder_model_merged), not the
 * vision or audio encoders of the multimodal checkpoint. */
export const BASE_MODEL = {
  repoId: "onnx-community/gemma-4-E4B-it-ONNX",
  revision: "2b6d2e26a962f70a33da8f7cd81e1afdd8d9cfb2",
  downloadGigabytes: 4.9,
} as const satisfies HubModelSource & { downloadGigabytes: number };

/** An Express LoRA merged into Gemma 4's q4f16 ONNX export, published on
 * Hugging Face at a pinned revision. */
export interface FineTune {
  readonly exportId: string;
  readonly size: "E4B" | "E2B";
  /** The checkpoint the adapter was trained on and merged into. */
  readonly baseModel: string;
  readonly adapterSha256: string;
  /** ONNX matrices the adapter patches: `mlx_lm` adapts every projection,
   * including the per-layer input gates. */
  readonly patchedMatrices: number;
  readonly downloadGigabytes: number;
  readonly hub: HubModelSource;
}

/** The step-40 adapter of the second E4B run. */
export const E4B_FINE_TUNE: FineTune = {
  exportId: "e4b-pilot2s40-d53063f5f606",
  size: "E4B",
  baseModel: "unsloth/gemma-4-E4B-it",
  adapterSha256:
    "d53063f5f606317cd9db5dd4bfc904e034d7f73a06b068155607459acc819fe2",
  patchedMatrices: 300,
  downloadGigabytes: 4.9,
  hub: {
    repoId: "wolfmanfx/gemma-4-E4B-it-A2UI-Express-ONNX",
    revision: "c8b774d2610221080cac3e9bf3c915ccf76f8ecb",
  },
};

/** The step-100 adapter of the E2B run on the same training rows. */
export const E2B_FINE_TUNE: FineTune = {
  exportId: "e2b-mlx1s100-eff96be9f058",
  size: "E2B",
  baseModel: "unsloth/gemma-4-E2B-it",
  adapterSha256:
    "eff96be9f058aec7993a0b1618d1d20c60eb4519d54f54c2352865ad297490f2",
  patchedMatrices: 240,
  downloadGigabytes: 3.1,
  hub: {
    repoId: "wolfmanfx/gemma-4-E2B-it-A2UI-Express-ONNX",
    revision: "b1e17443f869c5e22993d6492a157e58c586a117",
  },
};

/** Checkpoint 540 of the E2B retrain of 2026-11-07 (training/runs/e2b-chains,
 * 512 rows, 48/49 on the follow-up test set), patched into the base q4f16
 * export by model-tools/onnx_patch and published at a pinned revision. A
 * loopback root (`?transformersModelRoot=`) serves an export under test
 * instead. */
export const E2B_RETRAIN: FineTune = {
  exportId: "e2b-chains540-54c7414491e3",
  size: "E2B",
  baseModel: "unsloth/gemma-4-E2B-it",
  adapterSha256:
    "54c7414491e36610133c5f22e0397a66e97d901e3b119238abf50042dc7aa323",
  patchedMatrices: 240,
  downloadGigabytes: 3.1,
  hub: {
    repoId: "soverius-ai/gemma-4-E2B-it-A2UI-Express-ONNX",
    revision: "a50ff5d2695da98bcd5ea44a40854602b97f22c7",
  },
};

/** The E4B adapter of the 2026-11-07 retrain (training/runs/e4b, 160 steps on
 * 240 rows), patched into the base q4f16 export by model-tools/onnx_patch and
 * published at a pinned revision. */
export const E4B_RETRAIN: FineTune = {
  exportId: "e4b-cards160-778467f51be8",
  size: "E4B",
  baseModel: "unsloth/gemma-4-E4B-it",
  adapterSha256:
    "778467f51be80c125d41cecfc654f20083013e8602926f2ee1b7d7de910e105b",
  patchedMatrices: 300,
  downloadGigabytes: 4.9,
  hub: {
    repoId: "soverius-ai/gemma-4-E4B-it-A2UI-Express-ONNX",
    revision: "2e208422c10c2bc31c2253d9b878023587387e81",
  },
};

export const FINE_TUNES: readonly FineTune[] = [
  E4B_FINE_TUNE,
  E2B_FINE_TUNE,
  E4B_RETRAIN,
  E2B_RETRAIN,
];

export const fineTuneById = (exportId: string | undefined) =>
  FINE_TUNES.find((fineTune) => fineTune.exportId === exportId);

export const fineTuneByRepo = (repoId: string | undefined) =>
  FINE_TUNES.find((fineTune) => fineTune.hub.repoId === repoId);
