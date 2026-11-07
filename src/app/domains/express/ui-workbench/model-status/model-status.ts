import { ChangeDetectionStrategy, Component, input } from "@angular/core";
import type { AgentAvailability } from "../../data/agent/debug-state";

/** The id the toolbar's Enable button is described by. */
export const DOWNLOAD_NOTICE_ID = "download-notice";

/** One status line: whether the selected model is ready, what enabling it
 * downloads, and where its details live. */
@Component({
  selector: "app-model-status",
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { role: "status", "aria-live": "polite" },
  template: `
    <span class="dot" [class.enabled]="availability() === 'enabled'"></span>
    @switch (availability()) {
      @case ("enabled") {
        <span>{{ label() }} ready · {{ device() }}</span>
      }
      @case ("unsupported") {
        <span>
          @if (chromeSelected()) {
            Chrome Prompt API is unavailable. Choose a Gemma model to use
            WebGPU.
          } @else {
            Gemma requires a desktop browser with WebGPU. You can also select
            Chrome Prompt API.
          }
        </span>
      }
      @case ("unavailable") {
        <span
          >This model is unavailable in this browser. Choose another
          model.</span
        >
      }
      @case ("downloadable") {
        <span>Ready to enable</span>
      }
      @case ("downloading") {
        <span>Preparing model… {{ (progress() * 100).toFixed(0) }}%</span>
      }
      @case ("error") {
        <span
          >Model could not start. Check the diagnostics, then enable it to
          retry.</span
        >
      }
      @case ("available") {
        <span>Model installed · enable to start</span>
      }
      @default {
        <span>Preparing the Express catalog…</span>
      }
    }
    @if (notice()) {
      <span class="download-notice" [id]="noticeId">{{ notice() }}</span>
    }
    <a [href]="detailsUrl()" target="_blank" rel="noreferrer"
      >Model details ↗</a
    >
  `,
  styleUrl: "./model-status.css",
})
export class ModelStatus {
  readonly availability = input.required<AgentAvailability>();
  readonly label = input.required<string>();
  readonly device = input<string>();
  readonly progress = input(0);
  readonly chromeSelected = input(false);
  /** What enabling the model downloads, before it is enabled. */
  readonly notice = input("");
  readonly detailsUrl = input.required<string>();

  protected readonly noticeId = DOWNLOAD_NOTICE_ID;
}
