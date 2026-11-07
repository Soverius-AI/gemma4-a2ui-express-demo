import { EventType, withAuthoritativeActivityTypes } from "@ag-ui/client";
import type { Message, MessagesSnapshotEvent } from "@ag-ui/client";
import type {
  A2UIOperation,
  GenerativeUISurfaceContent,
} from "@copilotkit/shared";
import type { RunEvents } from "./run-events";
import {
  bySurface,
  EXPRESS_TARGET,
  hidesChange,
  SURFACE_ACTIVITY,
  surfaceContentOf,
  surfaceMessageId,
} from "./surfaces";

/**
 * The surface activities of one run. Every preview is published on top of
 * the transcript as it was committed before the run, and withdrawn by going
 * back to it; only a validated answer is committed.
 */
export class PreviewPublisher {
  readonly #startMessages: Message[];
  readonly #committed = new Map<string, GenerativeUISurfaceContent>();
  /** The operations this run last published, by activity message. */
  readonly #published = new Map<string, A2UIOperation[]>();
  #shown = false;

  constructor(
    messages: readonly Message[],
    private readonly threadId: string,
    private readonly events: RunEvents,
    /** Restores the transcript of a run whose subscriber has left. */
    private readonly restoreMessages: (messages: Message[]) => void,
  ) {
    this.#startMessages = structuredClone(messages) as Message[];
    for (const message of this.#startMessages) {
      const content = surfaceContentOf(message);
      if (content) this.#committed.set(message.id, content);
    }
  }

  /** Whether any preview of this run reached the page. */
  get shown() {
    return this.#shown;
  }

  /** The renderer state every response is checked against. */
  get committedOperations(): A2UIOperation[] {
    return [...this.#committed.values()].flatMap(
      (content) => content.operations,
    );
  }

  /** Shows the operations the SDK has read so far on top of each surface's
   * committed ones; a new view opens with `opening`. */
  preview(
    streamed: A2UIOperation[],
    view: { newSurfaceId: string; opening: A2UIOperation; catalogId: string },
  ) {
    const changed = [...bySurface(streamed)].flatMap(
      ([id, group]): [string, GenerativeUISurfaceContent][] => {
        const messageId = surfaceMessageId(this.threadId, id);
        const operations = [
          ...this.#committedOf(messageId),
          ...(id === view.newSurfaceId ? [view.opening] : []),
          ...group,
        ];
        return JSON.stringify(this.#published.get(messageId)) ===
          JSON.stringify(operations)
          ? []
          : [[messageId, content(id, view.catalogId, operations, "streaming")]];
      },
    );
    if (!changed.length) return;
    for (const [messageId, surface] of changed)
      this.#publish(messageId, surface);
    this.#shown = true;
    this.events.patch({ operations: streamed });
  }

  /** Commits a validated answer: each surface it changes becomes its
   * committed operations plus the answer's, replacing its preview. A
   * preview of a surface the answer leaves alone is withdrawn first. */
  commit(operations: A2UIOperation[], catalogId: string) {
    const contents = new Map(
      [...bySurface(operations)].map(
        ([id, group]): [string, GenerativeUISurfaceContent] => {
          const messageId = surfaceMessageId(this.threadId, id);
          return [
            messageId,
            content(
              id,
              catalogId,
              [...this.#committedOf(messageId), ...group],
              "ready",
            ),
          ];
        },
      ),
    );
    if ([...this.#published.keys()].some((id) => !contents.has(id)))
      this.withdraw();
    for (const [messageId, surface] of contents)
      this.#publish(messageId, surface);
  }

  /** Takes back every preview of this run: the transcript, and with it the
   * surfaces, return to how they were committed. A new view leaves no
   * activity behind, and the page forgets its surface. */
  withdraw() {
    if (!this.#published.size) return;
    this.#published.clear();
    if (!this.events.isCurrent()) return;
    if (this.events.detached)
      this.restoreMessages(structuredClone(this.#startMessages));
    else
      this.events.emit(
        withAuthoritativeActivityTypes(
          {
            type: EventType.MESSAGES_SNAPSHOT,
            messages: structuredClone(this.#startMessages),
          } as MessagesSnapshotEvent,
          [SURFACE_ACTIVITY],
        ) as unknown as Record<string, unknown>,
      );
    this.events.patch({ operations: [] });
  }

  #committedOf(messageId: string) {
    return this.#committed.get(messageId)?.operations ?? [];
  }

  #publish(messageId: string, surface: GenerativeUISurfaceContent) {
    const base = this.#committedOf(messageId);
    const previous = this.#published.get(messageId) ?? base;
    const snapshot = (operations: A2UIOperation[]) =>
      this.events.emit({
        type: EventType.ACTIVITY_SNAPSHOT,
        messageId,
        activityType: SURFACE_ACTIVITY,
        replace: true,
        content: { ...surface, operations },
      });
    // A shorter list makes the manager rebuild the surface, user-owned
    // values included, instead of skipping the change.
    if (hidesChange(previous, surface.operations)) snapshot(base);
    snapshot(surface.operations);
    this.#published.set(messageId, surface.operations);
  }
}

function content(
  surfaceId: string,
  catalogId: string,
  operations: A2UIOperation[],
  status: "streaming" | "ready",
): GenerativeUISurfaceContent {
  return {
    surfaceId,
    target: EXPRESS_TARGET,
    catalogId,
    operations,
    status,
    rejected: [],
  };
}
