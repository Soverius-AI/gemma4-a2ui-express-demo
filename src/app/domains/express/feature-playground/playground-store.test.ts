import "@angular/compiler";
import { Injector, signal } from "@angular/core";
import { CopilotKit, CopilotkitAgentFactory } from "@copilotkit/angular";
import { CopilotGenerativeUI } from "@copilotkit/angular/generative-ui";
import { describe, expect, it, vi } from "vitest";
import { initialDebugState } from "../data/agent/debug-state";
import { ExpressAgent } from "../data/agent/express-agent";
import { PlaygroundStore } from "./playground-store";

/** The store against a real agent whose reset the test releases. */
function fixture() {
  const agent = new ExpressAgent();
  let finish!: () => void;
  vi.spyOn(agent, "resetConversation").mockImplementation(
    () => new Promise<void>((resolve) => (finish = resolve)),
  );
  vi.spyOn(agent, "selectModel").mockImplementation(() => {});
  vi.spyOn(agent, "addMessage");
  agent.addMessage({ id: "earlier", role: "user", content: "earlier" });
  const state = signal({
    a2uiExpress: { ...initialDebugState(), availability: "enabled" },
  });
  const runAgent = vi.fn();
  const setActiveThread = vi.fn();
  const injector = Injector.create({
    providers: [
      PlaygroundStore,
      {
        provide: CopilotkitAgentFactory,
        useValue: {
          createAgentStoreSignal: () =>
            signal({ agent, state, isRunning: signal(false) }),
        },
      },
      { provide: CopilotKit, useValue: { core: { runAgent } } },
      {
        provide: CopilotGenerativeUI,
        useValue: { manager: { setActiveThread }, version: signal(0) },
      },
    ],
  });
  const store = injector.get(PlaygroundStore);
  vi.spyOn(store, "prepare").mockResolvedValue(true);
  return { store, agent, runAgent, setActiveThread, finish: () => finish() };
}

describe("playground session commands", () => {
  it("ignores overlapping Reset, model selection and example clicks until prepare completes", async () => {
    const f = fixture();
    const pending = f.store.reset();
    expect(f.store.resetting()).toBe(true);
    expect(await f.store.reset()).toBe(false);
    await f.store.runExample("ignored");
    await f.store.selectModel("chrome");
    expect(f.agent.selectModel).not.toHaveBeenCalled();
    expect(f.agent.addMessage).toHaveBeenCalledOnce();
    f.finish();
    expect(await pending).toBe(true);
    expect(f.agent.resetConversation).toHaveBeenCalledOnce();
    expect(f.setActiveThread).toHaveBeenCalledOnce();
    expect(f.store.conversation()).toBe(1);
    expect(f.store.resetting()).toBe(false);
  });

  it("releases the guard on prepare failure and does not run an example after failed reset", async () => {
    const f = fixture();
    vi.mocked(f.store.prepare).mockRejectedValueOnce(
      new Error("prepare failed"),
    );
    const pending = f.store.reset();
    f.finish();
    await expect(pending).rejects.toThrow("prepare failed");
    expect(f.store.resetting()).toBe(false);
    f.agent.addMessage({ id: "again", role: "user", content: "again" });
    vi.mocked(f.agent.addMessage).mockClear();
    vi.mocked(f.store.prepare).mockResolvedValueOnce(false);
    const example = f.store.runExample("not sent");
    f.finish();
    await example;
    expect(f.agent.addMessage).not.toHaveBeenCalled();
    expect(f.runAgent).not.toHaveBeenCalled();
    expect(f.store.resetting()).toBe(false);
  });

  it("runs an example in a fresh conversation", async () => {
    const f = fixture();
    const example = f.store.runExample("Create a card");
    f.finish();
    await example;
    expect(f.agent.addMessage).toHaveBeenLastCalledWith(
      expect.objectContaining({ role: "user", content: "Create a card" }),
    );
    expect(f.runAgent).toHaveBeenCalledWith({ agent: f.agent });
  });
});
