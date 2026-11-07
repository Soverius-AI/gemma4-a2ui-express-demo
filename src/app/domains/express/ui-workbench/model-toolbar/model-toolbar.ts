import {
  ChangeDetectionStrategy,
  Component,
  input,
  output,
} from "@angular/core";
import { isModelMode } from "../../../inference/data/model-catalog";
import type { ModelMode } from "../../../inference/data/model-catalog";

export interface ModelOption {
  readonly mode: ModelMode;
  /** The picker's entry. */
  readonly label: string;
  /** The status line once the model is ready. */
  readonly statusLabel: string;
}

/** Model picker and session controls: Stop or Reset, and Enable. */
@Component({
  selector: "app-model-toolbar",
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <span class="local-badge"><span aria-hidden="true">●</span> On device</span>
    <label class="model-picker">
      <span class="sr-only">Inference model</span>
      <select
        aria-label="Inference model"
        title="Switching models starts a fresh conversation"
        [disabled]="busy()"
        (change)="select($event)"
      >
        @for (option of options(); track option.mode) {
          <option [value]="option.mode" [selected]="option.mode === mode()">
            {{ option.label }}
          </option>
        }
      </select>
    </label>
    @if (stoppable()) {
      <button type="button" class="btn btn-stop" (click)="stop.emit()">
        Stop
      </button>
    } @else {
      <button
        type="button"
        class="btn"
        [disabled]="resetting()"
        (click)="reset.emit()"
      >
        Reset
      </button>
    }
    <button
      type="button"
      class="btn btn-primary"
      [disabled]="!canEnable()"
      [attr.aria-describedby]="describedBy()"
      (click)="enable.emit()"
    >
      {{ enableLabel() }}
    </button>
  `,
  styleUrl: "./model-toolbar.css",
})
export class ModelToolbar {
  readonly options = input.required<readonly ModelOption[]>();
  readonly mode = input.required<ModelMode>();
  /** Disables switching while a run, reset or model start is in flight. */
  readonly busy = input(false);
  /** Shows Stop instead of Reset while generating or loading. */
  readonly stoppable = input(false);
  readonly resetting = input(false);
  readonly canEnable = input(false);
  readonly enableLabel = input.required<string>();
  readonly describedBy = input<string | null>(null);

  readonly modeChange = output<ModelMode>();
  readonly stop = output();
  readonly reset = output();
  readonly enable = output();

  protected select(event: Event) {
    const { value } = event.target as HTMLSelectElement;
    if (isModelMode(value)) this.modeChange.emit(value);
  }
}
