import { describe, expect, it } from "vitest";
import {
  E2B_FINE_TUNE,
  E4B_FINE_TUNE,
  fineTuneById,
  fineTuneByRepo,
  isModelMode,
} from "./model-catalog";

describe("model catalog", () => {
  it("names each published adapter by its hash prefix", () => {
    for (const fineTune of [E4B_FINE_TUNE, E2B_FINE_TUNE])
      expect(
        fineTune.exportId.endsWith(fineTune.adapterSha256.slice(0, 12)),
      ).toBe(true);
    expect(E4B_FINE_TUNE).toMatchObject({
      size: "E4B",
      baseModel: "unsloth/gemma-4-E4B-it",
      patchedMatrices: 300,
    });
    expect(E2B_FINE_TUNE).toMatchObject({
      size: "E2B",
      baseModel: "unsloth/gemma-4-E2B-it",
      patchedMatrices: 240,
      downloadGigabytes: 3.1,
    });
  });

  it("finds fine-tunes only by their exact id or Hub repository", () => {
    expect(fineTuneById(E2B_FINE_TUNE.exportId)).toBe(E2B_FINE_TUNE);
    expect(fineTuneById("round2-0cf5af05c20f")).toBeUndefined();
    expect(fineTuneByRepo(E4B_FINE_TUNE.hub.repoId)).toBe(E4B_FINE_TUNE);
    expect(
      fineTuneByRepo("onnx-community/gemma-4-E4B-it-ONNX"),
    ).toBeUndefined();
  });

  it("accepts only known model modes", () => {
    expect(isModelMode("webllm-e2b")).toBe(true);
    expect(isModelMode("auto")).toBe(false);
  });
});
