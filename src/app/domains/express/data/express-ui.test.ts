import "@angular/compiler";
import { readFileSync } from "node:fs";
import { exportA2UICatalog } from "@copilotkit/angular/generative-ui";
import { describe, expect, it } from "vitest";
import { ExpressCompiler } from "./compiler/express-compiler";
import { EXPRESS_UI } from "./express-ui";

describe("the fine-tuning contract", () => {
  it("sends exactly the system prompt that training/ trains with", () => {
    const trained = readFileSync(
      new URL(
        "../../../../../training/data/system-prompt.txt",
        import.meta.url,
      ),
      "utf8",
    );
    const { systemPrompt } = new ExpressCompiler(exportA2UICatalog(EXPRESS_UI));
    expect(systemPrompt).toBe(trained);
  });
});
