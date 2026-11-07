import { ChangeDetectionStrategy, Component, input } from "@angular/core";

/** The Expand/Restore control of a workbench panel. */
@Component({
  selector: "button[appPanelToggle]",
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    type: "button",
    class: "btn btn-quiet",
    "[attr.aria-expanded]": "expanded()",
    "[attr.aria-label]": "(expanded() ? 'Restore ' : 'Expand ') + label()",
  },
  template: `{{ expanded() ? "Restore ↙" : "Expand ↗" }}`,
})
export class PanelToggle {
  readonly expanded = input.required<boolean>();
  /** The panel's name, for assistive technology. */
  readonly label = input.required<string>();
}
