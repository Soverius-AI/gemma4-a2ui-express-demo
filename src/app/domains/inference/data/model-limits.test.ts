import { describe, expect, it } from "vitest";
import {
  assertContextBudget,
  LOCAL_CONTEXT_WINDOW,
  MAX_NEW_TOKENS,
} from "./model-limits";

describe("local model context policy", () => {
  it("uses a 32k total context with the unchanged 3072-token output cap", () => {
    expect(LOCAL_CONTEXT_WINDOW).toBe(32768);
    expect(MAX_NEW_TOKENS).toBe(3072);
    expect(() => assertContextBudget(29696)).not.toThrow();
    expect(() => assertContextBudget(29697)).toThrow("Nothing was truncated");
  });
  it("accepts an actual input longer than the old 16k window", () => {
    expect(() => assertContextBudget(20000)).not.toThrow();
  });
  it("fails closed for absent/non-integral token counts", () => {
    for (const count of [NaN, Infinity, 0, -1, 1.5])
      expect(() => assertContextBudget(count)).toThrow(
        "Invalid local-model token budget",
      );
  });
});
