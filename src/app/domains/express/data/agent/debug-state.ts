import type { A2UIOperation } from "@copilotkit/shared";
import type {
  ModelAvailability,
  ModelInfo,
} from "../../../inference/data/local-model";

/** The selected model's availability, or the agent's session state. */
export type AgentAvailability =
  ModelAvailability | "checking" | "enabled" | "error";

/** The agent state the playground inspects, under `state.a2uiExpress`. */
export interface ExpressDebugState {
  availability: AgentAvailability;
  downloadProgress: number;
  modelStarting: boolean;
  stage:
    | "idle"
    | "preparing"
    | "generating"
    | "compiling"
    | "ready"
    | "answered"
    | "error"
    | "cancelled";
  source: "model";
  model: ModelInfo | null;
  runId: string;
  surfaceId: string;
  systemPrompt: string;
  /** The request turn of the first attempt, exactly as the model reads it. */
  modelRequest: string;
  dsl: string;
  lastSuccessfulDsl: string;
  operations: A2UIOperation[];
  diagnostics: string[];
  attempt: number;
  compilerFeedback: string[];
  elapsedMs: number;
  compileMs: number;
  action: unknown;
}

export const initialDebugState = (): ExpressDebugState => ({
  availability: "checking",
  downloadProgress: 0,
  modelStarting: false,
  stage: "idle",
  source: "model",
  model: null,
  runId: "",
  surfaceId: "",
  systemPrompt: "",
  modelRequest: "",
  dsl: "",
  lastSuccessfulDsl: "",
  operations: [],
  diagnostics: [],
  attempt: 0,
  compilerFeedback: [],
  elapsedMs: 0,
  compileMs: 0,
  action: null,
});
