import type { A2UIOperation } from "@copilotkit/shared";
import type { ConversationTurn } from "../../../inference/data/chat-messages";
import { ExpressCompiler, operationsOf } from "../compiler/express-compiler";
import { errorMessage } from "./errors";
import { lineRepairTurn, repairTurn } from "./model-request";
import type { PreviewPublisher } from "./preview-publisher";
import type { RunEvents } from "./run-events";

/** How often streamed operations are published to the preview. */
export const PREVIEW_INTERVAL_MS = 150;
/** Rejected lines the model may correct in one response before it writes
 * the whole block again. */
export const LINE_FIXES = 3;
/** Whole-response attempts before a run gives up. */
export const ATTEMPTS = 3;

/** One inference to its end; `onChunk` sees the text as it streams. */
export type Infer = (
  turn: string,
  onChunk?: (chunk: string) => void,
) => Promise<string>;

export type GenerationOutcome =
  | { kind: "answer"; text: string; dsl: string }
  | {
      kind: "ui";
      dsl: string;
      operations: A2UIOperation[];
      compileStarted: number;
    };

/** Every attempt produced output the compiler rejected. */
export class InvalidOutputError extends Error {}

/**
 * Turns one request into validated Express. Each attempt streams a response
 * into the preview, then compiles it; a rejected line is corrected by the
 * model and swapped in, and a rejected block is written again with the
 * compiler's diagnostic.
 */
export class ExpressGeneration {
  readonly #feedback: string[] = [];
  /** The preview due at the end of the interval while the model is silent,
   * so the latest output appears even if no chunk follows. */
  #flushTimer: ReturnType<typeof setTimeout> | undefined;

  constructor(
    private readonly run: {
      compiler: ExpressCompiler;
      request: string;
      newSurfaceId: string;
      /** The host's new view, open before the model writes its root. */
      opening: A2UIOperation;
      catalogId: string;
      /** Prose may answer a message, never an action. */
      isAction: boolean;
      signal: AbortSignal;
      started: number;
      infer: Infer;
      events: RunEvents;
      previews: PreviewPublisher;
    },
  ) {}

  async generate(): Promise<GenerationOutcome> {
    const { events, previews, signal } = this.run;
    let diagnostic = "";
    let rejectedDsl = "";
    for (let attempt = 1; ; attempt++) {
      signal.throwIfAborted();
      events.patch({ attempt, stage: "generating", dsl: "" });
      const conversation = this.#conversation(diagnostic, rejectedDsl);
      const dsl = await this.#stream(this.#runData(conversation));
      events.patch({ dsl, stage: "compiling" });
      const compileStarted = performance.now();
      const draft = { source: dsl };
      try {
        const parsed = await this.#parse(draft, attempt, [
          ...conversation,
          { role: "assistant", content: dsl },
        ]);
        if (parsed.operations.length)
          return {
            kind: "ui",
            dsl: draft.source,
            operations: parsed.operations,
            compileStarted,
          };
        if (parsed.text && !this.run.isAction && !diagnostic)
          return { kind: "answer", text: parsed.text, dsl: draft.source };
        throw new Error("The reply has no <a2ui> block.");
      } catch (error) {
        diagnostic = errorMessage(error);
        this.#feedback.push(`Attempt ${attempt}: ${diagnostic}`);
        rejectedDsl = draft.source;
        // The rejected draft leaves the preview before its repair streams:
        // the user sees the last validated UI meanwhile.
        previews.withdraw();
        events.patch({
          compilerFeedback: [...this.#feedback],
          diagnostics: [diagnostic],
        });
        if (attempt === ATTEMPTS) throw new InvalidOutputError(diagnostic);
      }
    }
  }

  /** The request turn; a whole-block repair adds the rejected draft and its
   * diagnostic. */
  #conversation(diagnostic: string, rejectedDsl: string): ConversationTurn[] {
    return [
      { role: "user", content: this.run.request },
      ...(diagnostic
        ? [
            {
              role: "assistant" as const,
              content: rejectedDsl.slice(0, 12_000) || "(no reply)",
            },
            { role: "user" as const, content: repairTurn(diagnostic) },
          ]
        : []),
    ];
  }

  #runData(conversation: readonly ConversationTurn[]) {
    const runData = JSON.stringify({ conversation });
    if (runData.length > 28_000)
      throw new Error(
        "The local model context is too large. Reset the demo to start a smaller conversation.",
      );
    return runData;
  }

  /** Streams one response, publishing what the SDK reads of it. Once the SDK
   * rejects a block its stream ends and the last preview stays: a new view
   * ends there at its root line, which would create it again. */
  async #stream(runData: string) {
    const { compiler, events, previews, signal, started } = this.run;
    let stream: ReturnType<ExpressCompiler["stream"]> | undefined =
      compiler.stream(this.run.newSurfaceId, [
        ...previews.committedOperations,
        this.run.opening,
      ]);
    const streamed: A2UIOperation[] = [];
    let dsl = "";
    let lastPublished = 0;
    let lastPreview = performance.now();
    const flush = () => {
      this.#clearFlush();
      previews.preview(streamed, this.run);
      lastPreview = performance.now();
    };
    const clearOnAbort = () => this.#clearFlush();
    signal.addEventListener("abort", clearOnAbort, { once: true });
    try {
      return await this.run.infer(runData, (chunk) => {
        dsl += chunk;
        if (performance.now() - lastPublished > 80) {
          events.patch({
            dsl,
            elapsedMs: Math.round(performance.now() - started),
          });
          lastPublished = performance.now();
        }
        try {
          streamed.push(...operationsOf(stream?.feed(chunk) ?? []));
        } catch {
          stream = undefined;
        }
        // Publishing every chunk would stall the page, so a chunk inside
        // the interval waits for its end, even if the model goes silent.
        const wait = PREVIEW_INTERVAL_MS - (performance.now() - lastPreview);
        if (wait <= 0) flush();
        else
          this.#flushTimer ??= setTimeout(() => {
            this.#flushTimer = undefined;
            if (events.isCurrent() && !signal.aborted) flush();
          }, wait);
      });
    } finally {
      // The whole response is checked next: a pending preview must never
      // land on the final or the withdrawn UI.
      this.#clearFlush();
      signal.removeEventListener("abort", clearOnAbort);
    }
  }

  /** Express error recovery: the model corrects only the rejected line, the
   * correction is swapped into the block, and the whole response is checked
   * again. `draft.source` follows every swap. */
  async #parse(
    draft: { source: string },
    attempt: number,
    /** The turns that produced the draft, ending with the draft itself. */
    history: readonly ConversationTurn[],
  ) {
    const { compiler, events } = this.run;
    // Each correction continues the conversation that produced the draft, so
    // a runtime that keeps a conversation's KV cache (WebLLM) prefills only
    // the small repair turn instead of the whole system prompt again. A
    // conversation too large for the context falls back to the bare turn.
    const repairs: ConversationTurn[] = [...history];
    for (let fixes = 0; ; fixes++) {
      try {
        return compiler.parse(
          draft.source,
          this.run.newSurfaceId,
          this.run.previews.committedOperations,
        );
      } catch (error) {
        const line =
          fixes < LINE_FIXES ? compiler.invalidLine(error) : undefined;
        if (!line) throw error;
        const diagnostic = errorMessage(error);
        const turn: ConversationTurn = {
          role: "user",
          content: lineRepairTurn({ line: line.text, diagnostic }),
        };
        repairs.push(turn);
        let runData = JSON.stringify({ conversation: repairs });
        if (runData.length > 28_000)
          runData = JSON.stringify({ conversation: [turn] });
        const correction = await this.run.infer(runData);
        repairs.push({ role: "assistant", content: correction });
        this.#feedback.push(
          `Attempt ${attempt}, line ${line.index + 1}: ${diagnostic} Corrected: ${correction.trim()}`,
        );
        events.patch({
          compilerFeedback: [...this.#feedback],
          diagnostics: [diagnostic],
        });
        const swapped = compiler.hotSwap(draft.source, line, correction);
        if (swapped === undefined) throw error;
        draft.source = swapped;
        events.patch({ dsl: draft.source });
      }
    }
  }

  #clearFlush() {
    clearTimeout(this.#flushTimer);
    this.#flushTimer = undefined;
  }
}
