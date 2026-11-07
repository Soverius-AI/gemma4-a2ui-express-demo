import { E2B_FINE_TUNE } from "../model-catalog";
import { modelFilesRoot } from "../model-files";

/** The E2B Express adapter the WebLLM weights were merged from: the local
 * Unsloth retrain of 2026-11-07 (training/runs/e2b-chains, checkpoint 540 of 768), not the
 * published adapter the Transformers.js E2B option loads. */
export const WEBLLM_ADAPTER_SHA256 =
  "54c7414491e36610133c5f22e0397a66e97d901e3b119238abf50042dc7aa323";

/** The producer identity `model-tools` writes into a2ui-mlc-manifest.json,
 * checked before any WebLLM runtime or weight is loaded. */
export const WEBLLM_E2B_MANIFEST = {
  schemaVersion: 1,
  adapterSha256: WEBLLM_ADAPTER_SHA256,
  baseModel: E2B_FINE_TUNE.baseModel,
  baseRevision: "d36bdd3855c82a6a1a23f7c459b749f5724ae75d",
  quantization: "q4f16_1",
  modelId: `gemma-4-e2b-express-${WEBLLM_ADAPTER_SHA256.slice(0, 12)}-q4f16_1-MLC`,
} as const;
export type WebLLME2BManifest = typeof WEBLLM_E2B_MANIFEST;

export const WEBLLM_MANIFEST_FILE = "a2ui-mlc-manifest.json";
export const WEBLLM_DEFAULT_ROOT = "/models/webllm-e2b-express/resolve/local/";

const isLoopback = (hostname: string) =>
  ["localhost", "127.0.0.1", "[::1]"].includes(hostname);

export function verifyWebLLMManifest(
  value: unknown,
): asserts value is WebLLME2BManifest {
  if (
    !value ||
    typeof value !== "object" ||
    Object.entries(WEBLLM_E2B_MANIFEST).some(
      ([key, expected]) => (value as Record<string, unknown>)[key] !== expected,
    )
  )
    throw new Error(
      "WebLLM requires the exact verified E2B Express fine-tune; provenance mismatch.",
    );
}

/** Where the converted model is hosted: this site by default, or an
 * explicit `?webllmModelRoot=` HTTPS (or loopback HTTP) directory. */
export function webllmModelRoot(
  requested = typeof location === "undefined"
    ? undefined
    : (new URLSearchParams(location.search).get("webllmModelRoot") ??
      undefined),
  origin = typeof location === "undefined"
    ? "http://localhost"
    : location.origin,
): string {
  const root = modelFilesRoot(requested ?? WEBLLM_DEFAULT_ROOT, origin);
  const url = new URL(root);
  if (url.protocol !== "https:" && !isLoopback(url.hostname))
    throw new Error("Public WebLLM model files require HTTPS.");
  // Without a revision segment WebLLM appends resolve/main itself.
  if (!/\/resolve\/[^/]+\/$/.test(url.pathname))
    throw new Error(
      "WebLLM model root must end in /resolve/<revision>/ to prevent SDK URL substitution.",
    );
  return root;
}
