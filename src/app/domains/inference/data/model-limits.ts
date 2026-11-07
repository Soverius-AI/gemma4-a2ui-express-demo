/** Application budget, not preallocation: ONNX KV tensors grow with each call. */
export const LOCAL_CONTEXT_WINDOW = 32_768;
export const MAX_NEW_TOKENS = 3_072;

/** Count the actual templated input, reserve the entire output, never truncate. */
export function assertContextBudget(
  inputTokens: number,
  contextWindow = LOCAL_CONTEXT_WINDOW,
  outputTokens = MAX_NEW_TOKENS,
): void {
  if (
    ![inputTokens, contextWindow, outputTokens].every(Number.isSafeInteger) ||
    inputTokens < 1 ||
    contextWindow < 1 ||
    outputTokens < 1
  ) {
    throw new Error("Invalid local-model token budget.");
  }
  if (inputTokens + outputTokens > contextWindow) {
    throw new Error(
      `Local model input has ${inputTokens} tokens; with ${outputTokens} reserved output tokens it exceeds the ${contextWindow}-token context. Nothing was truncated. Start a fresh context or shorten the request.`,
    );
  }
}
