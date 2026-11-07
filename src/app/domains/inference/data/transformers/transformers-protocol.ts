import type { ChatMessage } from "../chat-messages";
import type { ModelInfo } from "../local-model";

/** Messages from the page to the Transformers.js worker. `load` without a
 * fine-tune id loads the untuned base model. */
export type WorkerRequest =
  | {
      type: "load";
      id: number;
      fineTuneId?: string;
      /** A loopback directory that holds the export, instead of the Hub. */
      modelRoot?: string;
    }
  | { type: "generate"; id: number; messages: ChatMessage[] };

export type WorkerResponse =
  | { type: "progress"; id: number; value: number; info: ModelInfo }
  | { type: "ready"; id: number; info: ModelInfo }
  | { type: "chunk"; id: number; text: string }
  | { type: "done"; id: number }
  | { type: "error"; id: number; message: string };
