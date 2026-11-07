import { EventType } from "@ag-ui/client";
import type { BaseEvent, RunAgentInput } from "@ag-ui/client";
import type { Subscriber } from "rxjs";
import type { ExpressDebugState } from "./debug-state";

/** The AG-UI events of one run. A stale run (the agent was disposed or
 * reset, moved to another thread, or started a newer run) emits nothing. */
export class RunEvents {
  constructor(
    private readonly observer: Subscriber<BaseEvent>,
    private readonly input: Pick<RunAgentInput, "threadId" | "runId">,
    readonly isCurrent: () => boolean,
    /** Writes agent state directly, for a run whose subscriber has left. */
    private readonly updateState: (patch: Partial<ExpressDebugState>) => void,
  ) {}

  /** Whether the subscriber left before the run ended. */
  get detached() {
    return this.observer.closed;
  }

  emit(event: Record<string, unknown>) {
    if (!this.observer.closed && this.isCurrent())
      this.observer.next(event as BaseEvent);
  }

  /** Updates the debug state; a detached run still reports its end. */
  patch(value: Partial<ExpressDebugState>) {
    if (this.observer.closed) {
      if (this.isCurrent()) this.updateState(value);
      return;
    }
    this.emit({
      type: EventType.STATE_DELTA,
      delta: Object.entries(value).map(([key, item]) => ({
        op: "replace",
        path: `/a2uiExpress/${key}`,
        value: item,
      })),
    });
  }

  /** One complete assistant message. */
  say(kind: "answer" | "ui" | "status", text: string) {
    const messageId = `express-${kind}:${this.input.runId}`;
    this.emit({
      type: EventType.TEXT_MESSAGE_START,
      messageId,
      role: "assistant",
    });
    this.emit({ type: EventType.TEXT_MESSAGE_CONTENT, messageId, delta: text });
    this.emit({ type: EventType.TEXT_MESSAGE_END, messageId });
  }

  started() {
    this.emit({
      type: EventType.RUN_STARTED,
      threadId: this.input.threadId,
      runId: this.input.runId,
    });
  }

  finished() {
    this.emit({
      type: EventType.RUN_FINISHED,
      threadId: this.input.threadId,
      runId: this.input.runId,
    });
  }

  failed(message: string) {
    this.emit({ type: EventType.RUN_ERROR, message });
  }

  complete() {
    this.observer.complete();
  }
}
