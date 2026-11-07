import {
  computed,
  DestroyRef,
  inject,
  Injectable,
  signal,
} from "@angular/core";
import { CopilotKit, injectAgentStore } from "@copilotkit/angular";
import {
  CopilotGenerativeUI,
  exportA2UICatalog,
} from "@copilotkit/angular/generative-ui";
import { buildGenerativeUIManifest } from "@copilotkit/core/generative-ui";
import { BASE_MODEL, fineTuneByRepo } from "../../inference/data/model-catalog";
import type { ModelMode } from "../../inference/data/model-catalog";
import { CHECKOUT_GUIDANCE } from "../data/agent/action-guidance";
import { initialDebugState } from "../data/agent/debug-state";
import type { ExpressDebugState } from "../data/agent/debug-state";
import { EXPRESS_AGENT_ID, ExpressAgent } from "../data/agent/express-agent";
import { EXPRESS_TARGET } from "../data/agent/surfaces";
import {
  WEBLLM_MANIFEST_FILE,
  webllmModelRoot,
} from "../../inference/data/webllm/webllm-manifest";
import { MODEL_OPTIONS } from "./model-options";

/** The playground's session: which model runs, whether it is enabled, and
 * the guarded commands (enable, stop, reset, switch model, run an example)
 * that drive the local agent. */
@Injectable()
export class PlaygroundStore {
  readonly #copilotkit = inject(CopilotKit);
  readonly #runtime = inject(CopilotGenerativeUI);
  readonly #agentStore = injectAgentStore(EXPRESS_AGENT_ID);

  readonly mode = signal<ModelMode>("base");
  readonly resetting = signal(false);
  /** Counts conversations; a reset starts the next one. */
  readonly conversation = signal(0);

  readonly debug = computed(
    () =>
      (
        this.#agentStore()?.state() as
          { a2uiExpress?: ExpressDebugState } | undefined
      )?.a2uiExpress ?? initialDebugState(),
  );
  readonly running = computed(() => this.#agentStore()?.isRunning() ?? false);
  /** Commands that change the session wait until nothing is in flight. */
  readonly busy = computed(
    () => this.running() || this.resetting() || this.debug().modelStarting,
  );
  readonly enabled = computed(() => this.debug().availability === "enabled");
  readonly ready = computed(
    () => !!this.debug().systemPrompt && this.debug().stage !== "preparing",
  );
  readonly canEnable = computed(
    () =>
      this.ready() &&
      !this.resetting() &&
      !this.debug().modelStarting &&
      !["unsupported", "unavailable", "checking", "enabled"].includes(
        this.debug().availability,
      ),
  );
  readonly enableLabel = computed(() => {
    const { availability } = this.debug();
    if (availability === "enabled") return "Model enabled";
    return availability === "downloadable"
      ? "Download & enable"
      : "Enable local model";
  });
  readonly statusLabel = computed(
    () =>
      MODEL_OPTIONS.find(({ mode }) => mode === this.mode())?.statusLabel ?? "",
  );
  /** Weights are a large download; say so before the user enables them. */
  readonly downloadNotice = computed(() => {
    const { availability, model } = this.debug();
    if (availability === "enabled" || availability === "unsupported") return "";
    if (model?.backend === "webllm")
      return "Experimental MLC fine-tune: ~2.8 GB from the configured site · WebGPU shader-f16 required · caching when available. Slower and less reliable in recorded tests.";
    if (model?.backend === "transformers")
      return `First load: ~${fineTuneByRepo(model.model)?.downloadGigabytes ?? BASE_MODEL.downloadGigabytes} GB from Hugging Face · WebGPU required · cached in your browser.`;
    if (model?.backend === "chrome" && availability === "downloadable")
      return "Chrome manages the built-in model’s download and storage.";
    return "";
  });
  readonly modelDetailsUrl = computed(() => {
    const model = this.debug().model;
    if (model?.backend === "webllm")
      return new URL(WEBLLM_MANIFEST_FILE, webllmModelRoot()).href;
    const fineTune = fineTuneByRepo(model?.model);
    if (fineTune)
      return `https://huggingface.co/${fineTune.hub.repoId}/tree/${fineTune.hub.revision}`;
    return model?.backend === "transformers"
      ? `https://huggingface.co/${model.model}`
      : "https://developer.chrome.com/docs/ai/prompt-api";
  });
  /** The user-owned values of the newest preview surface. */
  readonly formData = computed(() => {
    this.#runtime.version();
    const threadId = this.#agentStore()?.agent.threadId ?? "";
    const surfaceId = Object.entries(
      this.#runtime.manager.describeSurfaces(threadId),
    )
      .filter(
        ([, surface]) => surface.target === EXPRESS_TARGET && !surface.status,
      )
      .at(-1)?.[0];
    return (
      this.#runtime.manager
        .debugSurfaces()
        .find(
          (surface) =>
            surface.threadId === threadId && surface.surfaceId === surfaceId,
        )?.dataModel ?? {}
    );
  });

  constructor() {
    inject(DestroyRef).onDestroy(() => this.agent()?.dispose());
  }

  agent(): ExpressAgent | undefined {
    const agent = this.#agentStore()?.agent;
    return agent instanceof ExpressAgent ? agent : undefined;
  }

  /** Hands the native catalog and surfaces to the agent. */
  async prepare(): Promise<boolean> {
    const alreadyResetting = this.resetting();
    this.resetting.set(true);
    try {
      const agent = this.agent();
      if (!agent) return false;
      await agent.configure(
        exportA2UICatalog(this.#runtime.ui),
        buildGenerativeUIManifest(this.#runtime, agent.threadId),
        // Guidance joins only matching action runs, never view creation.
        [CHECKOUT_GUIDANCE],
      );
      return agent.debugState.stage !== "error";
    } finally {
      this.resetting.set(alreadyResetting);
    }
  }

  enable() {
    if (this.resetting()) return;
    void this.agent()?.enable();
  }

  stop() {
    this.agent()?.cancel();
  }

  /** Switching models starts a fresh conversation. */
  async selectModel(mode: ModelMode) {
    if (this.busy()) return;
    this.mode.set(mode);
    this.agent()?.selectModel(mode);
    await this.reset();
  }

  /** A fresh thread and transcript; idle weights stay loaded. */
  async reset(): Promise<boolean> {
    const agent = this.agent();
    if (!agent || this.running() || this.resetting()) return false;
    this.resetting.set(true);
    try {
      await agent.resetConversation();
      agent.setMessages([]);
      agent.threadId = crypto.randomUUID();
      this.#runtime.manager.setActiveThread(agent.threadId);
      this.conversation.update((count) => count + 1);
      return await this.prepare();
    } finally {
      this.resetting.set(false);
    }
  }

  /** Runs a curated request in a fresh conversation. */
  async runExample(request: string) {
    if (this.busy() || !this.enabled()) return;
    const agent = this.agent();
    if (!agent) return;
    if (agent.messages.length) {
      if (!(await this.reset()) || !this.enabled()) return;
    }
    agent.addMessage({
      id: crypto.randomUUID(),
      role: "user",
      content: request,
    });
    await this.#copilotkit.core.runAgent({ agent });
  }
}
