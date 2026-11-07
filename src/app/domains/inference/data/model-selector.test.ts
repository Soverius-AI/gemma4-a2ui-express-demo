import { describe, expect, it, vi } from "vitest";
import type {
  LocalModel,
  ModelAvailability,
  ModelBackend,
} from "./local-model";
import { ModelSelector } from "./model-selector";

function model(
  backend: ModelBackend,
  availability: ModelAvailability = "available",
) {
  return {
    info: { backend, model: backend, device: "test" },
    availability: vi.fn(async () => availability),
    enable: vi.fn(async () => {}),
    generate: vi.fn(async function* () {
      yield "actual model output";
    }),
    resetContext: vi.fn(async () => {}),
    destroy: vi.fn(),
  } satisfies LocalModel;
}

function selector() {
  const models = {
    base: model("transformers", "downloadable"),
    finetuned: model("transformers", "downloadable"),
    "finetuned-e2b": model("transformers", "downloadable"),
    "webllm-e2b": model("webllm", "downloadable"),
    chrome: model("chrome", "unavailable"),
  };
  return { models, selector: new ModelSelector(models) };
}

describe("model selection", () => {
  it("uses untuned Gemma 4 E4B by default, even when Chrome has a model", async () => {
    const { models, selector: browser } = selector();
    models.chrome.availability.mockResolvedValue("available");
    expect(browser.mode).toBe("base");
    await browser.enable("SDK guide", new AbortController().signal, () => {});
    expect(models.base.enable).toHaveBeenCalledOnce();
    expect(models.chrome.availability).not.toHaveBeenCalled();
    expect(models.chrome.enable).not.toHaveBeenCalled();
  });

  it("switches models explicitly, releasing the previous one", async () => {
    const { models, selector: browser } = selector();
    browser.select("finetuned-e2b");
    await browser.enable("SDK guide", new AbortController().signal, () => {});
    expect(models["finetuned-e2b"].enable).toHaveBeenCalledOnce();
    browser.select("chrome");
    expect(models["finetuned-e2b"].destroy).toHaveBeenCalled();
    expect(await browser.availability()).toBe("unavailable");
    expect(browser.info.backend).toBe("chrome");
    browser.select("webllm-e2b");
    expect(await browser.availability()).toBe("downloadable");
    expect(browser.info.backend).toBe("webllm");
  });

  it.each(["base", "chrome", "webllm-e2b"] as const)(
    "never falls back when %s fails to start",
    async (mode) => {
      const { models, selector: browser } = selector();
      browser.select(mode);
      models[mode].enable.mockRejectedValueOnce(new Error("cannot start"));
      await expect(
        browser.enable("SDK guide", new AbortController().signal, () => {}),
      ).rejects.toThrow("cannot start");
      for (const [other, candidate] of Object.entries(models))
        if (other !== mode) expect(candidate.enable).not.toHaveBeenCalled();
    },
  );

  it("delegates generation and warm resets to the selected model", async () => {
    const { models, selector: browser } = selector();
    browser.select("finetuned");
    const output: string[] = [];
    for await (const text of browser.generate(
      "request",
      "SDK guide",
      new AbortController().signal,
    ))
      output.push(text);
    expect(output).toEqual(["actual model output"]);
    await browser.resetContext(new AbortController().signal);
    expect(models.finetuned.resetContext).toHaveBeenCalledOnce();
    expect(models.base.resetContext).not.toHaveBeenCalled();
  });
});
