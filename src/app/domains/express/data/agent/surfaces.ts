import type { Message } from "@ag-ui/client";
import type { GenerativeUISurfaceManager } from "@copilotkit/core/generative-ui";
import { getOperationSurfaceId } from "@copilotkit/shared";
import type {
  A2UIOperation,
  GenerativeUIManifest,
  GenerativeUISurfaceContent,
} from "@copilotkit/shared";

/** The preview target every generated view renders into. */
export const EXPRESS_TARGET = "express-preview";
export const SURFACE_ACTIVITY = "generative-ui-surface";

export const surfaceMessageId = (threadId: string, surfaceId: string) =>
  `express:${threadId}:${surfaceId}`;

/** The surface a generated-UI activity message carries. */
export function surfaceContentOf(
  message: Message,
): GenerativeUISurfaceContent | undefined {
  return message.role === "activity" &&
    message.activityType === SURFACE_ACTIVITY
    ? (message.content as unknown as GenerativeUISurfaceContent)
    : undefined;
}

/** The short id the host gives the next new view: one past the highest
 * view-N among existing, deleted and failed surfaces, so no id is reused. */
export function nextViewId(taken: Iterable<string>) {
  let highest = 0;
  for (const id of taken) {
    const match = /^view-(\d+)$/.exec(id);
    if (match) highest = Math.max(highest, Number(match[1]));
  }
  return `view-${highest + 1}`;
}

/** Surfaces that exist now: not deleted, and not failed to create. */
export function liveSurfaceIds(manifest: GenerativeUIManifest) {
  return Object.entries(manifest.surfaces)
    .filter(([, s]) => s.status !== "deleted" && s.status !== "failed")
    .map(([id]) => id);
}

/** The components an operation creates or updates. */
export function componentsOf(operation: A2UIOperation) {
  return (
    ("createSurface" in operation
      ? operation.createSurface.components
      : "updateComponents" in operation
        ? operation.updateComponents.components
        : undefined) ?? []
  );
}

export const createsSurface = (operations: readonly A2UIOperation[]) =>
  operations.some((operation) => "createSurface" in operation);

/** The operations of each surface, in order. */
export function bySurface(operations: readonly A2UIOperation[]) {
  const groups = new Map<string, A2UIOperation[]>();
  for (const operation of operations) {
    const id = getOperationSurfaceId(operation)!;
    groups.set(id, [...(groups.get(id) ?? []), operation]);
  }
  return groups;
}

/**
 * Whether the native surface manager would take `next` for an extension of
 * `previous`, the list it applied last, when it is not one. It compares only
 * the last operation it applied, so it would skip a changed earlier one.
 */
export function hidesChange(
  previous: readonly A2UIOperation[],
  next: readonly A2UIOperation[],
) {
  if (!previous.length || next.length < previous.length) return false;
  const same = (index: number) =>
    JSON.stringify(previous[index]) === JSON.stringify(next[index]);
  return (
    same(previous.length - 1) &&
    previous.slice(0, -1).some((_, index) => !same(index))
  );
}

/**
 * The page's half of withdrawing a preview: an express-preview surface of
 * the thread whose activity message the agent took back never became a
 * view, so the native manager forgets it. Call it whenever the agent's
 * messages change. The manager keeps any surface that was ready once.
 */
export function discardWithdrawnPreviews(
  manager: GenerativeUISurfaceManager,
  threadId: string,
  messages: readonly Message[],
) {
  const kept = new Set(
    messages.map((message) => surfaceContentOf(message)?.surfaceId),
  );
  for (const surface of manager.debugSurfaces())
    if (
      surface.threadId === threadId &&
      surface.target === EXPRESS_TARGET &&
      !kept.has(surface.surfaceId)
    )
      manager.discardSurface(threadId, surface.surfaceId);
}
