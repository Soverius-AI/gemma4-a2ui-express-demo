import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  model,
  output,
} from "@angular/core";
import type { ExpressDebugState } from "../../data/agent/debug-state";
import { ATTEMPTS } from "../../data/agent/express-generation";
import { CodeView } from "../code-view/code-view";
import { PanelToggle } from "../panel-toggle/panel-toggle";

export type DebugTab = "operations" | "prompt" | "state" | "data";

const TABS: readonly { tab: DebugTab; label: string; code: string }[] = [
  { tab: "operations", label: "Compiled JSON", code: "Compiled JSON" },
  { tab: "prompt", label: "System prompt", code: "System prompt" },
  { tab: "state", label: "Agent state", code: "Agent state JSON" },
  { tab: "data", label: "Form data", code: "Form data JSON" },
];

/** The agent's compiled operations, system prompt, state and form data. */
@Component({
  selector: "app-debug-panel",
  imports: [CodeView, PanelToggle],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    role: "region",
    "aria-label": "Agent state debugger",
    "[class.collapsed]": "!visible()",
  },
  templateUrl: "./debug-panel.html",
  styleUrl: "./debug-panel.css",
})
export class DebugPanel {
  readonly debug = input.required<ExpressDebugState>();
  readonly formData = input<Record<string, unknown>>({});
  /** Whether the panel fills the workbench. */
  readonly expanded = input(false);
  readonly tab = model<DebugTab>("operations");
  readonly collapsed = model(true);

  /** Expand or restore, with the button that asked for it. */
  readonly expandToggle = output<HTMLButtonElement>();
  /** Hiding an expanded panel restores the layout. */
  readonly restore = output();

  protected readonly tabs = TABS;
  protected readonly attempts = ATTEMPTS;
  protected readonly visible = computed(
    () => !this.collapsed() || this.expanded(),
  );
  protected readonly codeLabel = computed(
    () => TABS.find(({ tab }) => tab === this.tab())?.code ?? "",
  );
  protected readonly json = computed(
    () =>
      JSON.stringify(
        this.tab() === "operations"
          ? this.debug().operations
          : this.tab() === "data"
            ? this.formData()
            : this.debug(),
        null,
        2,
      ) ?? "",
  );

  protected select(tab: DebugTab) {
    this.tab.set(tab);
    this.collapsed.set(false);
  }

  protected toggleExpanded(event: Event) {
    this.expandToggle.emit(event.currentTarget as HTMLButtonElement);
  }

  protected toggleVisible() {
    if (this.expanded()) {
      this.restore.emit();
      this.collapsed.set(true);
    } else {
      this.collapsed.update((collapsed) => !collapsed);
    }
  }
}
