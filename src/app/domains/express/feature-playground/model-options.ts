import type { ModelOption } from "../ui-workbench/model-toolbar/model-toolbar";

/** The models the picker offers, in order. */
export const MODEL_OPTIONS: readonly ModelOption[] = [
  { mode: "base", label: "Gemma 4 E4B", statusLabel: "Gemma 4 E4B" },
  {
    mode: "webllm-e2b",
    label: "WebLLM preview · Gemma 4 E2B fine-tuned",
    statusLabel: "Gemma 4 E2B fine-tuned · WebLLM preview",
  },
  {
    mode: "finetuned",
    label: "Gemma 4 E4B · fine-tuned",
    statusLabel: "Gemma 4 E4B · fine-tuned",
  },
  {
    mode: "finetuned-e2b",
    label: "Gemma 4 E2B · fine-tuned",
    statusLabel: "Gemma 4 E2B · fine-tuned",
  },
  {
    mode: "chrome",
    label: "Chrome Prompt API · built-in",
    statusLabel: "Chrome Prompt API",
  },
];
