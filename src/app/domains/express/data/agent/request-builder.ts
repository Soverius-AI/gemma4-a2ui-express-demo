import type { Message } from "@ag-ui/client";
import type {
  GenerativeUIManifest,
  GenerativeUISurfaceContent,
} from "@copilotkit/shared";
import type { ConversationTurn } from "../../../inference/data/chat-messages";
import type { ActionGuidance } from "./action-guidance";
import {
  actionTurn,
  creationTurn,
  readonlyDisplay,
  rootChildren,
  rootLineWith,
} from "./model-request";
import type { ContextEntry, ViewSource } from "./model-request";
import { componentsOf, liveSurfaceIds } from "./surfaces";
import type { ViewMemory } from "./view-memory";

/** The action a CopilotKit UI envelope carries, when it names one. */
export interface ActionScope {
  name: string;
  surfaceId: string;
  /** The submitted values, as the page reports them. */
  context: unknown;
}

export function actionScopeOf(action: unknown): ActionScope | undefined {
  const event = (
    action as {
      action?: { name?: unknown; surfaceId?: unknown; context?: unknown };
    } | null
  )?.action;
  return typeof event?.name === "string" && typeof event.surfaceId === "string"
    ? { name: event.name, surfaceId: event.surfaceId, context: event.context }
    : undefined;
}

/** The request turn for a new view or an edit: the latest user request, a
 * little of the conversation before it, and the last two live views. Status
 * and "updated the preview" notes never reach the model: they would teach
 * it to answer in prose. */
export function creationRequest(input: {
  messages: readonly Message[];
  manifest: GenerativeUIManifest;
  context: readonly ContextEntry[];
  views: ViewMemory;
  newSurfaceId: string;
}) {
  const turns: ConversationTurn[] = input.messages.flatMap((m) =>
    (m.role === "user" || m.role === "assistant") &&
    typeof m.content === "string" &&
    !m.id.startsWith("express-status:") &&
    !m.id.startsWith("express-ui:")
      ? [{ role: m.role, content: m.content }]
      : [],
  );
  let lastUser = turns.length - 1;
  while (lastUser >= 0 && turns[lastUser]!.role !== "user") lastUser--;
  const userRequest = turns[lastUser]?.content.slice(-3000) ?? "";
  const viewOf = (id: string): ViewSource[] => {
    const source = input.views.source(id);
    return source
      ? [{ id, source, values: input.manifest.surfaces[id]?.dataModel }]
      : [];
  };
  const request = creationTurn({
    request: userRequest,
    earlier: turns
      .slice(0, Math.max(lastUser, 0))
      .slice(-4)
      .map((turn) => ({ ...turn, content: turn.content.slice(0, 400) })),
    context: input.context,
    views: liveSurfaceIds(input.manifest).slice(-2).flatMap(viewOf),
    newSurfaceId: input.newSurfaceId,
  });
  return { request, userRequest };
}

/** The request turn that answers a submitted action inside its view. */
export function actionRequest(input: {
  scope: ActionScope;
  manifest: GenerativeUIManifest;
  context: readonly ContextEntry[];
  views: ViewMemory;
  /** The view as committed in the transcript. */
  surface: GenerativeUISurfaceContent | undefined;
  guidance: readonly ActionGuidance[];
}) {
  const { scope } = input;
  const origin = input.views.request(scope.surfaceId);
  const operations = input.surface?.operations ?? [];
  const components = operations.flatMap(componentsOf);
  // A fresh name for the outcome, so it never replaces a component.
  let resultName = "result";
  for (
    let n = 2;
    components.some(
      ({ id }) => id === resultName || id.startsWith(`${resultName}_`),
    );
    n++
  )
    resultName = `result_${n}`;
  // Only names the model wrote itself, not generated inline ids.
  const source = input.views.source(scope.surfaceId) ?? "";
  return actionTurn({
    origin,
    display: readonlyDisplay(operations),
    name: scope.name,
    surfaceId: scope.surfaceId,
    values:
      scope.context && typeof scope.context === "object"
        ? (scope.context as Record<string, unknown>)
        : (input.manifest.surfaces[scope.surfaceId]?.dataModel ?? {}),
    // Guidance about an action's outcome joins the request only while that
    // action is answered, so it cannot shape the first view.
    guidance: input.guidance
      .filter((entry) => entry.matches(scope.name, origin))
      .map(({ description, value }) => ({ description, value })),
    context: input.context,
    existingTypes: Object.fromEntries(
      components
        .filter(({ id }) => id !== "root")
        .map(({ id, component }) => [id, component]),
    ),
    keep: rootChildren(operations).filter((id) =>
      new RegExp(`^${id} =`, "m").test(source),
    ),
    rootLine: rootLineWith(operations, resultName),
    resultName,
  });
}
