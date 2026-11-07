import { AbstractAgent, EventType } from "@ag-ui/client";
import type { BaseEvent, RunAgentInput } from "@ag-ui/client";
import {
  envelopeToManifest,
  readCopilotKitUIEnvelope,
} from "@copilotkit/shared";
import type {
  A2UIOperation,
  GenerativeUIManifest,
  GenerativeUISurfaceContent,
} from "@copilotkit/shared";
import { Observable } from "rxjs";
import { ModelSelector } from "../../../inference/data/model-selector";
import type { LocalModel } from "../../../inference/data/local-model";
import type { ModelMode } from "../../../inference/data/model-catalog";
import { ExpressCompiler } from "../compiler/express-compiler";
import type { ActionGuidance } from "./action-guidance";
import { initialDebugState } from "./debug-state";
import type { ExpressDebugState } from "./debug-state";
import { errorMessage } from "./errors";
import {
  ATTEMPTS,
  ExpressGeneration,
  InvalidOutputError,
} from "./express-generation";
import type { Infer } from "./express-generation";
import { PreviewPublisher } from "./preview-publisher";
import {
  actionRequest,
  actionScopeOf,
  creationRequest,
} from "./request-builder";
import { RunEvents } from "./run-events";
import { createsSurface, nextViewId, surfaceContentOf } from "./surfaces";
import { ViewMemory } from "./view-memory";

export const EXPRESS_AGENT_ID = "a2ui-1.0";
/** Model output beyond this is rejected rather than compiled. */
const OUTPUT_LIMIT = 60_000;

/**
 * A real AG-UI agent whose inference runs in the browser. It owns the model
 * session (configure, enable, reset, dispose) and runs each request:
 * request turn → validated Express → committed views.
 */
export class ExpressAgent extends AbstractAgent {
  #compiler?: ExpressCompiler;
  #manifest?: GenerativeUIManifest;
  #controller?: AbortController;
  #enableController?: AbortController;
  #modelReady = false;
  #readySystem?: string;
  /** Bumped by every configure, dispose and reset: older work is stale. */
  #configuration = 0;
  #actionGuidance: readonly ActionGuidance[] = [];
  readonly #views = new ViewMemory();

  constructor(private readonly model: LocalModel = new ModelSelector()) {
    super({
      agentId: EXPRESS_AGENT_ID,
      initialState: { a2uiExpress: initialDebugState() },
    });
  }

  get debugState(): ExpressDebugState {
    return this.state.a2uiExpress ?? initialDebugState();
  }

  async configure(
    schema: Record<string, unknown>,
    manifest: GenerativeUIManifest,
    actionGuidance: readonly ActionGuidance[] = [],
  ) {
    const configuration = ++this.#configuration;
    this.#update({ stage: "preparing", diagnostics: [] });
    try {
      const compiler = new ExpressCompiler(schema);
      if (
        this.#compiler &&
        this.#compiler.systemPrompt !== compiler.systemPrompt
      ) {
        this.#enableController?.abort();
        this.#enableController = undefined;
        this.#update({ modelStarting: false });
      }
      if (this.#modelReady && this.#readySystem !== compiler.systemPrompt) {
        this.#enableController?.abort();
        this.#enableController = undefined;
        this.#releaseModel();
      }
      const availability = await this.model
        .availability()
        .catch(() => "unavailable" as const);
      if (configuration !== this.#configuration) return;
      this.#compiler = compiler;
      this.#manifest = manifest;
      this.#actionGuidance = actionGuidance;
      this.#update({
        stage: "idle",
        availability: this.#modelReady ? "enabled" : availability,
        model: this.model.info,
        systemPrompt: compiler.systemPrompt,
      });
    } catch (error) {
      if (configuration === this.#configuration)
        this.#update({ stage: "error", diagnostics: [errorMessage(error)] });
    }
  }

  async enable() {
    const compiler = this.#compiler;
    if (
      !compiler ||
      (this.#modelReady && this.#readySystem === compiler.systemPrompt)
    )
      return;
    const configuration = this.#configuration;
    this.#enableController?.abort();
    const controller = new AbortController();
    this.#enableController = controller;
    this.#modelReady = false;
    // Every generation of this session reads the same state.systemPrompt.
    const systemPrompt = compiler.systemPrompt;
    this.#update({
      availability: "downloading",
      modelStarting: true,
      diagnostics: [],
      systemPrompt,
    });
    try {
      await this.model.enable(
        systemPrompt,
        controller.signal,
        (downloadProgress) => {
          if (!controller.signal.aborted)
            this.#update({ downloadProgress, model: this.model.info });
        },
      );
      controller.signal.throwIfAborted();
      if (
        configuration !== this.#configuration ||
        this.#enableController !== controller
      )
        return;
      this.#modelReady = true;
      this.#readySystem = systemPrompt;
      this.#update({
        availability: "enabled",
        model: this.model.info,
        modelStarting: false,
        downloadProgress: 1,
      });
    } catch (error) {
      if (this.#enableController !== controller) return;
      // After Stop, ask the engine again: an installed Chrome model is not
      // "downloadable".
      const availability = controller.signal.aborted
        ? await this.model.availability().catch(() => "unavailable" as const)
        : "error";
      if (this.#enableController === controller)
        this.#update({
          availability,
          modelStarting: false,
          model: this.model.info,
          diagnostics: controller.signal.aborted ? [] : [errorMessage(error)],
        });
    }
  }

  cancel() {
    this.#controller?.abort();
    this.#enableController?.abort();
  }

  selectModel(mode: ModelMode) {
    this.dispose();
    if (this.model instanceof ModelSelector) this.model.select(mode);
  }

  dispose() {
    ++this.#configuration;
    this.#enableController?.abort();
    this.#enableController = undefined;
    this.cancel();
    this.#releaseModel();
    this.#forgetSession();
    this.#update({
      availability: "checking",
      modelStarting: false,
      stage: "idle",
    });
  }

  /** Start a fresh transcript without releasing idle model weights. */
  async resetConversation() {
    const active = !!this.#controller || this.debugState.modelStarting;
    const configuration = ++this.#configuration;
    this.cancel();
    this.#enableController = undefined;
    this.#controller = undefined;
    this.#forgetSession();
    this.setMessages([]);
    if (active) this.#releaseModel();
    else if (this.#modelReady) {
      try {
        await this.model.resetContext(new AbortController().signal);
      } catch {
        if (configuration !== this.#configuration) return;
        this.#releaseModel();
      }
    }
    if (configuration !== this.#configuration) return;
    this.setState({
      a2uiExpress: {
        ...initialDebugState(),
        availability: this.#modelReady ? "enabled" : "checking",
        model: this.model.info,
        systemPrompt: this.#readySystem ?? "",
      },
    });
  }

  run(input: RunAgentInput): Observable<BaseEvent> {
    return new Observable((observer) => {
      const controller = new AbortController();
      this.#controller = controller;
      const configuration = this.#configuration;
      /** Disposing the agent, moving to another thread or starting another
       * run makes a run stale: nothing of it may reach the newer session. */
      const isCurrent = () =>
        configuration === this.#configuration &&
        this.#controller === controller &&
        this.threadId === input.threadId;
      const events = new RunEvents(observer, input, isCurrent, (patch) =>
        this.#update(patch),
      );
      const previews = new PreviewPublisher(
        this.messages,
        input.threadId,
        events,
        (messages) => this.setMessages(messages),
      );
      /** Set once the run commits its outcome: leaving then takes nothing
       * back. */
      let settled = false;
      void this.#execute(input, controller, events, previews, () => {
        settled = true;
      });
      // Leaving early takes the preview back at once: a newer run may start
      // before the model notices, and this run may not touch that one.
      return () => {
        controller.abort();
        if (!settled) previews.withdraw();
      };
    });
  }

  async #execute(
    input: RunAgentInput,
    controller: AbortController,
    events: RunEvents,
    previews: PreviewPublisher,
    settle: () => void,
  ) {
    const started = performance.now();
    const elapsed = () => Math.round(performance.now() - started);
    events.started();
    const envelope = readCopilotKitUIEnvelope(input.forwardedProps);
    const manifest = envelope ? envelopeToManifest(envelope) : this.#manifest;
    const action = envelope?.metadata.extensions.copilotkit_action ?? null;
    const scope = actionScopeOf(action);
    // The compiler default for a block without surface(...): a short id a
    // small model copies reliably, never one that existed before.
    const newSurfaceId = nextViewId([
      ...Object.keys(manifest?.surfaces ?? {}),
      ...this.#surfaceActivities().map((content) => content.surfaceId),
    ]);
    events.emit({
      type: EventType.STATE_SNAPSHOT,
      snapshot: {
        ...input.state,
        a2uiExpress: {
          ...this.debugState,
          stage: "generating",
          source: "model",
          model: this.model.info,
          runId: input.runId,
          surfaceId: scope?.surfaceId ?? newSurfaceId,
          modelRequest: "",
          dsl: "",
          operations: [],
          diagnostics: [],
          attempt: 0,
          compilerFeedback: [],
          elapsedMs: 0,
          compileMs: 0,
          action,
        },
      },
    });
    try {
      if (!this.#compiler || !manifest)
        throw new Error("The Express catalog is still loading.");
      if (!this.#modelReady)
        throw new Error("Enable the browser model before generating.");
      // Application context is run data, never system instructions.
      const context = input.context.map(({ description, value }) => ({
        description,
        value,
      }));
      const { request, userRequest } = scope
        ? {
            request: actionRequest({
              scope,
              manifest,
              context,
              views: this.#views,
              surface: this.#surfaceActivities().find(
                (content) => content.surfaceId === scope.surfaceId,
              ),
              guidance: this.#actionGuidance,
            }),
            userRequest: "",
          }
        : creationRequest({
            messages: input.messages,
            manifest,
            context,
            views: this.#views,
            newSurfaceId,
          });
      events.patch({ modelRequest: request });
      const outcome = await new ExpressGeneration({
        compiler: this.#compiler,
        request,
        newSurfaceId,
        // The A2UI 1.0 wire value, as the SDK writes its own messages. The
        // SDK stream then emits a root-last view statement by statement, as
        // CopilotKit's surface writer opens a surface before its components.
        opening: {
          version: "1.0",
          createSurface: {
            surfaceId: newSurfaceId,
            catalogId: manifest.catalogId,
          },
        } as unknown as A2UIOperation,
        catalogId: manifest.catalogId,
        isAction: Boolean(action),
        signal: controller.signal,
        started,
        infer: this.#inference(controller, events),
        events,
        previews,
      }).generate();
      if (!events.isCurrent()) controller.abort();
      controller.signal.throwIfAborted();
      settle();
      if (outcome.kind === "answer") {
        // A prose reply publishes no UI and is not counted as a render.
        previews.withdraw();
        events.say("answer", outcome.text);
        events.patch({
          stage: "answered",
          diagnostics: [],
          dsl: outcome.dsl,
          operations: [],
          elapsedMs: elapsed(),
          compileMs: 0,
        });
        events.finished();
        return;
      }
      previews.commit(outcome.operations, manifest.catalogId);
      this.#views.remember(outcome.operations, outcome.dsl, userRequest);
      // The visible UI lives in surface activities; this note is for the
      // user only and never goes back to the model.
      events.say(
        "ui",
        "Updated the live preview. Inspect the Express source and compiled operations in the workbench.",
      );
      events.patch({
        stage: "ready",
        diagnostics: [],
        dsl: outcome.dsl,
        lastSuccessfulDsl: createsSurface(outcome.operations)
          ? outcome.dsl
          : `${this.debugState.lastSuccessfulDsl}\n${outcome.dsl}`.slice(
              -12_000,
            ),
        operations: outcome.operations,
        elapsedMs: elapsed(),
        compileMs: Math.round(performance.now() - outcome.compileStarted),
      });
      events.finished();
    } catch (error) {
      await this.#fail(error, controller, events, previews, elapsed());
    } finally {
      if (this.#controller === controller) this.#controller = undefined;
      events.complete();
    }
  }

  /** Ends a run that produced no valid answer: back to the last validated
   * UI, so no partly built view is left. */
  async #fail(
    error: unknown,
    controller: AbortController,
    events: RunEvents,
    previews: PreviewPublisher,
    elapsedMs: number,
  ) {
    // Disposed or reset: the new session gets nothing from this run.
    if (!events.isCurrent()) return events.finished();
    previews.withdraw();
    const outcome = previews.shown
      ? "its partial preview was withdrawn"
      : "nothing was published";
    if (controller.signal.aborted) {
      this.#modelReady = false;
      this.model.destroy();
      // Stopping releases the session, not an installed model. A model
      // enabled meanwhile keeps the availability it reported.
      const enabling = this.#enableController;
      const availability = await this.model
        .availability()
        .catch(() => "unavailable" as const);
      if (!events.isCurrent()) return events.finished();
      events.say(
        "status",
        `Stopped. ${previews.shown ? "The partial preview from this run was withdrawn" : "Nothing from this run was published"}, so the preview and the values you entered are unchanged. Enable the local model again to continue.`,
      );
      events.patch({
        ...(this.#enableController === enabling ? { availability } : {}),
        stage: "cancelled",
        elapsedMs,
      });
      return events.finished();
    }
    const message = errorMessage(error);
    events.say(
      "status",
      error instanceof InvalidOutputError
        ? `I could not produce a valid interface in ${ATTEMPTS} attempts, so ${outcome}. The previous preview and the values you entered are unchanged. Last problem: ${message} Try again, or ask for a smaller or simpler interface. Agent state lists every attempt.`
        : `The local model run failed: ${message} ${outcome.charAt(0).toUpperCase()}${outcome.slice(1)}; the previous preview and the values you entered are unchanged.`,
    );
    events.patch({ stage: "error", diagnostics: [message], elapsedMs });
    events.failed(message);
  }

  /** One inference of this run. A chunk that arrives after Stop or a reset
   * is never shown. */
  #inference(controller: AbortController, events: RunEvents): Infer {
    return async (turn, onChunk) => {
      let text = "";
      for await (const chunk of this.model.generate(
        turn,
        this.debugState.systemPrompt,
        controller.signal,
      )) {
        if (!events.isCurrent()) controller.abort();
        controller.signal.throwIfAborted();
        text += chunk;
        if (text.length > OUTPUT_LIMIT)
          throw new Error(
            "Model output exceeded the demo limit. Try a smaller interface.",
          );
        onChunk?.(chunk);
      }
      if (!events.isCurrent()) controller.abort();
      controller.signal.throwIfAborted();
      return text;
    };
  }

  /** The published surface activities of this agent's transcript. */
  #surfaceActivities(): GenerativeUISurfaceContent[] {
    return this.messages.flatMap((message) => surfaceContentOf(message) ?? []);
  }

  #update(patch: Partial<ExpressDebugState>) {
    this.setState({
      ...this.state,
      a2uiExpress: { ...this.debugState, ...patch },
    });
  }

  #releaseModel() {
    this.model.destroy();
    this.#modelReady = false;
    this.#readySystem = undefined;
  }

  #forgetSession() {
    this.#compiler = undefined;
    this.#manifest = undefined;
    this.#actionGuidance = [];
    this.#views.clear();
  }
}
