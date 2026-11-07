import { afterEach, describe, expect, it, vi } from "vitest";
import { WebLLMModel } from "./webllm-model";
import {
  WEBLLM_E2B_MANIFEST,
  verifyWebLLMManifest,
  webllmModelRoot,
} from "./webllm-manifest";

function fixture(manifest: unknown = WEBLLM_E2B_MANIFEST) {
  const worker = { terminate: vi.fn(), onerror: null };
  const create = vi.fn(async () =>
    (async function* () {
      yield { choices: [{ delta: { content: "raw <|channel>final output" } }] };
      yield { choices: [], usage: { prompt_tokens: 100 } };
    })(),
  );
  const engine = {
    chat: { completions: { create } },
    resetChat: vi.fn(async () => {}),
    unload: vi.fn(async () => {}),
  };
  const sdk = { CreateWebWorkerMLCEngine: vi.fn(async () => engine) };
  const load = vi.fn(async () => sdk);
  const fetcher = vi.fn(async () => ({ ok: true, json: async () => manifest }));
  const makeWorker = vi.fn(() => worker);
  const model = new WebLLMModel(
    makeWorker as never,
    load as never,
    () => "https://demo.test/models/resolve/local/",
    fetcher as never,
  );
  return { model, worker, create, engine, sdk, load, fetcher, makeWorker };
}
afterEach(() => vi.unstubAllGlobals());
describe("explicit WebLLM fine-tune", () => {
  it("requires exact producer provenance and safe non-substituted URLs", () => {
    expect(() => verifyWebLLMManifest(WEBLLM_E2B_MANIFEST)).not.toThrow();
    for (const key of Object.keys(WEBLLM_E2B_MANIFEST))
      expect(() =>
        verifyWebLLMManifest({ ...WEBLLM_E2B_MANIFEST, [key]: "wrong" }),
      ).toThrow("provenance");
    expect(webllmModelRoot(undefined, "https://demo.test")).toBe(
      "https://demo.test/models/webllm-e2b-express/resolve/local/",
    );
    expect(() =>
      webllmModelRoot(
        "http://localhost:9000/resolve/local/",
        "https://demo.test",
      ),
    ).toThrow();
    expect(() => webllmModelRoot("/models/", "https://demo.test")).toThrow(
      "resolve",
    );
    expect(
      webllmModelRoot(
        "http://localhost:9000/resolve/local/",
        "http://localhost:4210",
      ),
    ).toContain(":9000");
  });
  it("checks GPU f16 without fetching or loading SDK", async () => {
    vi.stubGlobal("Worker", class {});
    vi.stubGlobal("navigator", {
      gpu: {
        requestAdapter: async () => ({ features: new Set(["shader-f16"]) }),
      },
    });
    const f = fixture();
    expect(await f.model.availability()).toBe("downloadable");
    expect(f.fetcher).not.toHaveBeenCalled();
    expect(f.load).not.toHaveBeenCalled();
    vi.stubGlobal("navigator", {
      gpu: { requestAdapter: async () => ({ features: new Set() }) },
    });
    expect(await f.model.availability()).toBe("unavailable");
  });
  it("refuses other weights before SDK or worker startup", async () => {
    const f = fixture({ ...WEBLLM_E2B_MANIFEST, adapterSha256: "base" });
    expect(f.model.info).not.toHaveProperty("artifactSha256");
    await expect(
      f.model.enable("system", new AbortController().signal, vi.fn()),
    ).rejects.toThrow("provenance");
    expect(f.load).not.toHaveBeenCalled();
    expect(f.makeWorker).not.toHaveBeenCalled();
    expect(f.model.info).not.toHaveProperty("artifactSha256");
  });
  it("loads only custom record and passes unchanged SDK strings with greedy streaming", async () => {
    const f = fixture();
    const signal = new AbortController().signal;
    expect(f.model.info).not.toHaveProperty("artifactSha256");
    await f.model.enable("exact SDK system", signal, vi.fn());
    expect(f.model.info.artifactSha256).toBe(WEBLLM_E2B_MANIFEST.adapterSha256);
    expect(f.sdk.CreateWebWorkerMLCEngine).toHaveBeenCalledWith(
      f.worker,
      WEBLLM_E2B_MANIFEST.modelId,
      expect.objectContaining({
        appConfig: {
          model_list: [
            expect.objectContaining({
              model: "https://demo.test/models/resolve/local/",
              model_lib: "https://demo.test/models/resolve/local/model.wasm",
              model_manifest: "mlc-model-manifest.json",
              required_features: ["shader-f16"],
            }),
          ],
        },
      }),
      { context_window_size: 32768 },
    );
    const output: string[] = [];
    for await (const text of f.model.generate(
      "unaltered request",
      "exact SDK system",
      signal,
    ))
      output.push(text);
    expect(output).toEqual(["raw <|channel>final output"]);
    expect(f.create).toHaveBeenCalledWith(
      expect.objectContaining({
        temperature: 0,
        max_tokens: 3072,
        stream: true,
        messages: [
          { role: "system", content: "exact SDK system" },
          { role: "user", content: "unaltered request" },
        ],
      }),
    );
    f.model.destroy();
    expect(f.engine.unload).toHaveBeenCalled();
    expect(f.worker.terminate).toHaveBeenCalled();
    expect(f.model.info).not.toHaveProperty("artifactSha256");
  });
  it("resets idle chat KV without reloading the engine and cancels reset RPC on Stop", async () => {
    const f = fixture();
    await f.model.enable("sys", new AbortController().signal, vi.fn());
    await f.model.resetContext(new AbortController().signal);
    expect(f.engine.resetChat).toHaveBeenCalledOnce();
    expect(f.sdk.CreateWebWorkerMLCEngine).toHaveBeenCalledOnce();
    expect(f.worker.terminate).not.toHaveBeenCalled();
    f.engine.resetChat.mockImplementationOnce(() => new Promise(() => {}));
    const controller = new AbortController();
    const pending = f.model.resetContext(controller.signal);
    const rejected = expect(pending).rejects.toMatchObject({
      name: "AbortError",
    });
    controller.abort();
    await rejected;
    expect(f.worker.terminate).toHaveBeenCalledOnce();
  });
  it("allows the exact 32k minus output budget boundary", async () => {
    const f = fixture();
    const signal = new AbortController().signal;
    await f.model.enable("sys", signal, vi.fn());
    f.create.mockImplementationOnce(
      async () =>
        (async function* () {
          yield { choices: [], usage: { prompt_tokens: 32768 - 3072 } };
        })() as never,
    );
    await expect(
      f.model.generate("input", "sys", signal)[Symbol.asyncIterator]().next(),
    ).resolves.toMatchObject({ done: true });
    expect(f.worker.terminate).not.toHaveBeenCalled();
  });
  it("settles loading cancellation even when worker RPC never settles", async () => {
    const f = fixture();
    f.sdk.CreateWebWorkerMLCEngine.mockImplementation(
      () => new Promise(() => {}),
    );
    const controller = new AbortController();
    const pending = f.model.enable("sys", controller.signal, vi.fn());
    const rejected = expect(pending).rejects.toMatchObject({
      name: "AbortError",
    });
    await vi.waitFor(() => expect(f.makeWorker).toHaveBeenCalled());
    expect(f.model.info.artifactSha256).toBe(WEBLLM_E2B_MANIFEST.adapterSha256);
    controller.abort();
    await rejected;
    expect(f.worker.terminate).toHaveBeenCalled();
    expect(f.model.info).not.toHaveProperty("artifactSha256");
  });
  it("settles generation cancellation and permits a fresh enable", async () => {
    const f = fixture();
    const controller = new AbortController();
    await f.model.enable("sys", controller.signal, vi.fn());
    f.create.mockImplementationOnce(() => new Promise(() => {}));
    const pending = f.model
      .generate("input", "sys", controller.signal)
      [Symbol.asyncIterator]()
      .next();
    const rejected = expect(pending).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(f.model.info.artifactSha256).toBe(WEBLLM_E2B_MANIFEST.adapterSha256);
    controller.abort();
    await rejected;
    expect(f.worker.terminate).toHaveBeenCalled();
    expect(f.model.info).not.toHaveProperty("artifactSha256");
    await f.model.enable("sys", new AbortController().signal, vi.fn());
  });
  it("rejects a reported actual prompt that cannot fit without truncation", async () => {
    const f = fixture();
    const signal = new AbortController().signal;
    await f.model.enable("sys", signal, vi.fn());
    f.create.mockImplementationOnce(
      async () =>
        (async function* () {
          yield { choices: [], usage: { prompt_tokens: 30000 } };
        })() as never,
    );
    await expect(
      f.model.generate("input", "sys", signal)[Symbol.asyncIterator]().next(),
    ).rejects.toThrow("context budget");
  });
});
