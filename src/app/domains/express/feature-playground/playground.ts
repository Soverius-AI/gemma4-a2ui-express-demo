import {
  afterNextRender,
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  linkedSignal,
  signal,
} from "@angular/core";
import { CopilotChat, provideCopilotChatLabels } from "@copilotkit/angular";
import {
  CopilotGenerativeUI,
  CopilotGenerativeUISurface,
} from "@copilotkit/angular/generative-ui";
import type { ModelBackend } from "../../inference/data/local-model";
import { EXPRESS_AGENT_ID } from "../data/agent/express-agent";
import {
  discardWithdrawnPreviews,
  EXPRESS_TARGET,
} from "../data/agent/surfaces";
import { EXPRESS_COMPONENTS } from "../data/compiler/express-compiler";
import { CodeView } from "../ui-workbench/code-view/code-view";
import { DebugPanel } from "../ui-workbench/debug-panel/debug-panel";
import type { DebugTab } from "../ui-workbench/debug-panel/debug-panel";
import {
  ModelStatus,
  DOWNLOAD_NOTICE_ID,
} from "../ui-workbench/model-status/model-status";
import { ModelToolbar } from "../ui-workbench/model-toolbar/model-toolbar";
import { PageHeader } from "../ui-workbench/page-header/page-header";
import { PanelToggle } from "../ui-workbench/panel-toggle/panel-toggle";
import { DEMO_PROMPTS } from "./demo-prompts";
import { MODEL_OPTIONS } from "./model-options";
import { PlaygroundStore } from "./playground-store";

type Panel = "conversation" | "source" | "preview" | "debug";

const BACKEND_LABELS: Record<ModelBackend, string> = {
  chrome: "PROMPT API",
  transformers: "TRANSFORMERS.JS",
  webllm: "WEBLLM",
};

/** Conversation, Express source, native preview and debugger, side by side;
 * any panel can be expanded to fill the page. */
@Component({
  selector: "app-playground",
  imports: [
    CodeView,
    CopilotChat,
    CopilotGenerativeUISurface,
    DebugPanel,
    ModelStatus,
    ModelToolbar,
    PageHeader,
    PanelToggle,
  ],
  providers: [
    PlaygroundStore,
    provideCopilotChatLabels({
      chatInputPlaceholder: "Describe an interface…",
    }),
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { "(document:keydown.escape)": "restorePanels($event)" },
  templateUrl: "./playground.html",
  styleUrl: "./playground.css",
})
export class Playground {
  protected readonly store = inject(PlaygroundStore);
  readonly #runtime = inject(CopilotGenerativeUI);

  protected readonly agentId = EXPRESS_AGENT_ID;
  protected readonly previewTarget = EXPRESS_TARGET;
  protected readonly components = EXPRESS_COMPONENTS;
  protected readonly examples = DEMO_PROMPTS;
  protected readonly modelOptions = MODEL_OPTIONS;
  protected readonly noticeId = DOWNLOAD_NOTICE_ID;

  /** Every new conversation starts with the default layout. */
  protected readonly expandedPanel = linkedSignal<number, Panel | null>({
    source: this.store.conversation,
    computation: () => null,
  });
  protected readonly debugCollapsed = linkedSignal({
    source: this.store.conversation,
    computation: () => true,
  });
  protected readonly debugTab = signal<DebugTab>("operations");
  protected readonly backendLabel = computed(
    () => BACKEND_LABELS[this.store.debug().model?.backend ?? "chrome"],
  );
  #expansionTrigger?: HTMLElement;

  constructor() {
    afterNextRender(() => void this.store.prepare());
    effect(() => {
      const { stage, diagnostics } = this.store.debug();
      if (stage === "error" && diagnostics.length)
        this.debugCollapsed.set(false);
    });
    // A preview the agent withdraws leaves no surface behind on the page.
    effect((onCleanup) => {
      const agent = this.store.agent();
      if (!agent) return;
      const { unsubscribe } = agent.subscribe({
        onMessagesChanged: ({ messages }) =>
          discardWithdrawnPreviews(
            this.#runtime.manager,
            agent.threadId,
            messages,
          ),
      });
      onCleanup(unsubscribe);
    });
  }

  protected togglePanel(panel: Panel, trigger: EventTarget | null) {
    if (this.expandedPanel() === panel) {
      this.restorePanels();
      return;
    }
    this.#expansionTrigger = trigger as HTMLElement;
    this.expandedPanel.set(panel);
  }

  protected restorePanels(event?: Event) {
    if (!this.expandedPanel() || event?.defaultPrevented) return;
    this.expandedPanel.set(null);
    this.#expansionTrigger?.focus();
  }
}
