import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { E2B_FINE_TUNE } from "../model-catalog";
import type { WorkerRequest, WorkerResponse } from "./transformers-protocol";

const mock = vi.hoisted(() => ({
  pipeline: vi.fn(),
  tokenize: vi.fn(),
  generate: vi.fn(),
  processor: vi.fn(),
}));
vi.mock("@huggingface/transformers", () => ({
  env: { backends: { onnx: { wasm: {} } } },
  pipeline: mock.pipeline,
  Gemma4Processor: { from_pretrained: mock.processor },
  TextStreamer: class {
    constructor(
      readonly tokenizer: unknown,
      readonly options: unknown,
    ) {}
  },
}));

describe("ONNX worker fresh context and exact token budget", () => {
  let receive: (event: { data: WorkerRequest }) => Promise<void>;
  let sent: WorkerResponse[];
  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
    sent = [];
    mock.tokenize.mockReturnValue({
      input_ids: { dims: [1, 7400] },
      attention_mask: { dims: [1, 7400] },
    });
    mock.generate.mockResolvedValue({});
    mock.pipeline.mockResolvedValue({
      tokenizer: { chat_template: "owned", apply_chat_template: mock.tokenize },
      model: { generate: mock.generate },
    });
    vi.stubGlobal("navigator", {
      gpu: {
        requestAdapter: async () => ({ features: new Set(["shader-f16"]) }),
      },
    });
    vi.stubGlobal("self", {
      location: { origin: "http://localhost:4211" },
      addEventListener: (_type: string, callback: typeof receive) => {
        receive = callback;
      },
      postMessage: (message: WorkerResponse) => sent.push(message),
    });
    await import("./transformers.worker");
    await receive({ data: { type: "load", id: 1 } });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("keeps loaded sessions and sends only each new conversation, without past KV", async () => {
    const first = [{ role: "user" as const, content: "first conversation" }];
    const second = [{ role: "user" as const, content: "fresh conversation" }];
    await receive({ data: { type: "generate", id: 2, messages: first } });
    await receive({ data: { type: "generate", id: 3, messages: second } });
    expect(mock.pipeline).toHaveBeenCalledTimes(1);
    expect(mock.tokenize.mock.calls[0][0]).toEqual(first);
    expect(mock.tokenize.mock.calls[1][0]).toEqual(second);
    expect(mock.generate).toHaveBeenCalledTimes(2);
    for (const [args] of mock.generate.mock.calls) {
      expect(args).not.toHaveProperty("past_key_values");
      expect(args).toMatchObject({ max_new_tokens: 3072, do_sample: false });
    }
  });
  it("tokenizes the full chat without truncation, thinking or duplicate special tokens", async () => {
    await receive({
      data: {
        type: "generate",
        id: 2,
        messages: [{ role: "user", content: "complete input" }],
      },
    });
    expect(mock.tokenize).toHaveBeenCalledWith(
      expect.any(Array),
      expect.objectContaining({
        tokenize: true,
        return_dict: true,
        return_tensor: true,
        truncation: false,
        add_generation_prompt: true,
        enable_thinking: false,
      }),
    );
    expect(mock.generate.mock.calls[0][0].input_ids).toBe(
      mock.tokenize.mock.results[0].value.input_ids,
    );
    expect(sent.find((m) => m.type === "ready")).toMatchObject({
      info: { contextWindowTokens: 32768, quantization: "q4f16" },
    });
  });
  it("accepts more than 16k input tokens within the 32k total budget", async () => {
    mock.tokenize.mockReturnValue({ input_ids: { dims: [1, 20000] } });
    await receive({
      data: {
        type: "generate",
        id: 2,
        messages: [{ role: "user", content: "long input" }],
      },
    });
    expect(mock.generate).toHaveBeenCalledTimes(1);
    expect(sent.at(-1)).toEqual({ type: "done", id: 2 });
  });
  it("rejects oversized input before inference rather than truncating", async () => {
    mock.tokenize.mockReturnValue({ input_ids: { dims: [1, 29697] } });
    await receive({
      data: {
        type: "generate",
        id: 2,
        messages: [{ role: "user", content: "over budget" }],
      },
    });
    expect(mock.generate).not.toHaveBeenCalled();
    expect(sent.at(-1)).toMatchObject({
      type: "error",
      id: 2,
      message: expect.stringContaining("Nothing was truncated"),
    });
  });
  it("loads only catalogued fine-tunes whose published manifest names the pinned adapter", async () => {
    const manifest = {
      adapter_sha256: E2B_FINE_TUNE.adapterSha256,
      base_model: E2B_FINE_TUNE.baseModel,
      dtype: "q4f16",
      patched_onnx_matrices: 240,
      decoder_graph_unchanged: true,
    };
    const fetch = vi.fn(async () => ({ ok: true, json: async () => manifest }));
    vi.stubGlobal("fetch", fetch);
    await receive({
      data: { type: "load", id: 4, fineTuneId: E2B_FINE_TUNE.exportId },
    });
    expect(fetch).toHaveBeenCalledWith(
      `https://huggingface.co/${E2B_FINE_TUNE.hub.repoId}/resolve/${E2B_FINE_TUNE.hub.revision}/a2ui-export-manifest.json`,
    );
    expect(mock.pipeline).toHaveBeenLastCalledWith(
      "text-generation",
      E2B_FINE_TUNE.hub.repoId,
      expect.objectContaining({ revision: E2B_FINE_TUNE.hub.revision }),
    );
    expect(sent.at(-1)).toMatchObject({
      type: "ready",
      id: 4,
      info: { artifactSha256: E2B_FINE_TUNE.adapterSha256 },
    });
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...manifest, adapter_sha256: "0".repeat(64) }),
    });
    await receive({
      data: { type: "load", id: 5, fineTuneId: E2B_FINE_TUNE.exportId },
    });
    expect(sent.at(-1)).toMatchObject({
      type: "error",
      id: 5,
      message: expect.stringContaining("No substitute model"),
    });
    await receive({
      data: { type: "load", id: 6, fineTuneId: "round2-probe" },
    });
    expect(sent.at(-1)).toMatchObject({
      type: "error",
      id: 6,
      message: "Unknown Express fine-tune.",
    });
  });
});
