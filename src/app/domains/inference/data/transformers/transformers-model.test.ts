import { describe, expect, it, vi } from "vitest";
import { E2B_FINE_TUNE, E4B_FINE_TUNE } from "../model-catalog";
import { TransformersModel } from "./transformers-model";
import type { WorkerRequest, WorkerResponse } from "./transformers-protocol";

class ModelWorker {
  onmessage?: (event: { data: WorkerResponse }) => void;
  onerror?: (event: { message: string }) => void;
  requests: WorkerRequest[] = [];
  terminate = vi.fn();
  postMessage(message: WorkerRequest) {
    this.requests.push(message);
  }
  emit(message: WorkerResponse) {
    this.onmessage?.({ data: message });
  }
}
const info = {
  backend: "transformers" as const,
  model: "downloaded-coder",
  device: "webgpu",
};

describe("Transformers.js worker adapter", () => {
  it("loads the original Gemma checkpoint without a fine-tune", async () => {
    const worker = new ModelWorker();
    const model = new TransformersModel(
      () => worker as unknown as Worker,
      null,
    );
    expect(model.info.model).toBe("onnx-community/gemma-4-E4B-it-ONNX");
    const enabling = model.enable(
      "SDK guide",
      new AbortController().signal,
      () => {},
    );
    expect(worker.requests[0]).toEqual({ type: "load", id: 1 });
    worker.emit({ type: "ready", id: 1, info });
    await enabling;
    model.destroy();
  });
  it.each([
    ["E4B", E4B_FINE_TUNE],
    ["E2B", E2B_FINE_TUNE],
  ])(
    "asks the worker for the pinned %s fine-tune by id only",
    async (_size, fineTune) => {
      const worker = new ModelWorker();
      const model = new TransformersModel(
        () => worker as unknown as Worker,
        fineTune,
      );
      expect(model.info.model).toBe(fineTune.hub.repoId);
      const enabling = model.enable(
        "SDK guide",
        new AbortController().signal,
        () => {},
      );
      expect(worker.requests[0]).toEqual({
        type: "load",
        id: 1,
        fineTuneId: fineTune.exportId,
      });
      worker.emit({
        type: "ready",
        id: 1,
        info: { ...info, model: fineTune.hub.repoId },
      });
      await enabling;
      model.destroy();
    },
  );
  it("defaults to the E4B fine-tune", () => {
    const model = new TransformersModel(
      () => new ModelWorker() as unknown as Worker,
    );
    expect(model.info.model).toBe(E4B_FINE_TUNE.hub.repoId);
  });
  it("forwards download progress, streams model chunks and sends the system and conversation once", async () => {
    const worker = new ModelWorker();
    const model = new TransformersModel(() => worker as unknown as Worker);
    const progress = vi.fn();
    const signal = new AbortController().signal;
    const enabling = model.enable("Express contract", signal, progress);
    const loadId = worker.requests[0]!.id;
    worker.emit({ type: "progress", id: loadId, value: 0.5, info });
    worker.emit({ type: "ready", id: loadId, info });
    await enabling;
    expect(progress.mock.calls.map((call) => call[0])).toEqual([0.5, 1]);
    expect(model.info).toEqual(info);
    const stream = model.generate(
      JSON.stringify({
        conversation: [{ role: "user", content: "Create a form" }],
        action: null,
      }),
      "Express contract",
      signal,
    );
    const first = stream.next();
    const request = worker.requests[1]!;
    expect(request.type).toBe("generate");
    if (request.type !== "generate")
      throw new Error("Missing generation request");
    expect(request.messages).toEqual([
      { role: "system", content: "Express contract" },
      {
        role: "user",
        content: expect.stringContaining(
          "Current user request:\nCreate a form",
        ),
      },
    ]);
    worker.emit({ type: "chunk", id: loadId, text: "stale" });
    worker.emit({ type: "chunk", id: request.id, text: 'root = Text("' });
    expect((await first).value).toBe('root = Text("');
    worker.emit({ type: "chunk", id: request.id, text: 'Generated")' });
    worker.emit({ type: "done", id: request.id });
    expect((await stream.next()).value).toBe('Generated")');
    expect((await stream.next()).done).toBe(true);
    model.destroy();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it("passes composed turns through, merging same-role neighbours as Gemma's template requires", async () => {
    const worker = new ModelWorker();
    const model = new TransformersModel(() => worker as unknown as Worker);
    const enabling = model.enable(
      "system",
      new AbortController().signal,
      () => {},
    );
    worker.emit({ type: "ready", id: worker.requests[0]!.id, info });
    await enabling;
    void model
      .generate(
        JSON.stringify({
          conversation: [
            { role: "user", content: "Request: a card" },
            { role: "assistant", content: "root = Card(" },
            { role: "user", content: "The compiler rejected that block." },
            { role: "user", content: "Also make it green." },
          ],
        }),
        "system",
        new AbortController().signal,
      )
      .next();
    const request = worker.requests[1]!;
    if (request.type !== "generate")
      throw new Error("Missing generation request");
    expect(request.messages).toEqual([
      { role: "system", content: "system" },
      { role: "user", content: "Request: a card" },
      { role: "assistant", content: "root = Card(" },
      {
        role: "user",
        content: "The compiler rejected that block.\n\nAlso make it green.",
      },
    ]);
    model.destroy();
  });

  it("terminates an in-flight load on cancellation and rejects late readiness", async () => {
    const worker = new ModelWorker();
    const model = new TransformersModel(() => worker as unknown as Worker);
    const controller = new AbortController();
    const loading = model.enable("system", controller.signal, () => {});
    controller.abort();
    await expect(loading).rejects.toThrow();
    worker.emit({ type: "ready", id: worker.requests[0]!.id, info });
    expect(worker.terminate).toHaveBeenCalled();
    await expect(
      model.generate("request", "system", new AbortController().signal).next(),
    ).rejects.toThrow(/Enable/);
  });

  it("stops inference by terminating the worker and requires explicit reload", async () => {
    const worker = new ModelWorker();
    const model = new TransformersModel(() => worker as unknown as Worker);
    const controller = new AbortController();
    const loading = model.enable("system", controller.signal, () => {});
    worker.emit({ type: "ready", id: worker.requests[0]!.id, info });
    await loading;
    const stream = model.generate("request", "system", controller.signal);
    const pending = stream.next();
    controller.abort();
    await expect(pending).rejects.toThrow();
    expect(worker.terminate).toHaveBeenCalled();
    await expect(
      model.generate("request", "system", new AbortController().signal).next(),
    ).rejects.toThrow(/Enable/);
  });

  it("does not let an aborted load dispose the next worker", async () => {
    const oldWorker = new ModelWorker();
    const nextWorker = new ModelWorker();
    const workers = [oldWorker, nextWorker];
    const model = new TransformersModel(
      () => workers.shift()! as unknown as Worker,
    );
    const controller = new AbortController();
    const oldLoad = model.enable("old", controller.signal, () => {});
    controller.abort();
    const nextLoad = model.enable(
      "new",
      new AbortController().signal,
      () => {},
    );
    await expect(oldLoad).rejects.toThrow();
    nextWorker.emit({ type: "ready", id: nextWorker.requests[0]!.id, info });
    await nextLoad;
    expect(nextWorker.terminate).not.toHaveBeenCalled();
    model.destroy();
  });
});
