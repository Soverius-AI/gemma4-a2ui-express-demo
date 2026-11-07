// Compiles model outputs for training/data/test.jsonl against each step's
// committed view state (test.meta.json) and writes <outputs>-compiled.json.
// A no-op unless the three variables are set:
//   TESTSET_ROWS=training/data/test.jsonl TESTSET_META=training/data/test.meta.json \
//   TESTSET_OUT=outputs.json pnpm exec vitest run src/app/domains/express/data/testing/eval-testset.test.ts
import "@angular/compiler";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { exportA2UICatalog } from "@copilotkit/shared/generative-ui";
import type { A2UIOperation } from "@copilotkit/shared";
import { describe, expect, it } from "vitest";
import { ExpressCompiler } from "../compiler/express-compiler";
import { testUI } from "./test-catalog";

const ROWS = process.env["TESTSET_ROWS"] ?? "";
const META = process.env["TESTSET_META"] ?? "";
const OUTS = process.env["TESTSET_OUT"] ?? "";

describe("test-set outputs", () => {
  it("compiles each output against its chain state", () => {
    if (!ROWS || !META || !OUTS || !existsSync(OUTS)) return;
    const compiler = new ExpressCompiler(exportA2UICatalog(testUI));
    const rows = readFileSync(ROWS, "utf8")
      .split("\n")
      .filter(Boolean)
      .map((l) => JSON.parse(l) as { messages: { content: string }[] });
    const meta = JSON.parse(readFileSync(META, "utf8")) as {
      chain: string;
      step: number;
      kind: string;
      surfaceId: string;
      committed: string[];
    }[];
    const outputs = JSON.parse(readFileSync(OUTS, "utf8")) as {
      adapter: string;
      rows: { index: number; output: string }[];
    };
    const report: {
      index: number;
      chain: string;
      step: number;
      kind: string;
      ok: boolean;
      error?: string;
      ops?: number;
    }[] = [];
    for (const { index, output } of outputs.rows) {
      const m = meta[index]!;
      let committed: A2UIOperation[] = [];
      try {
        // Replay the chain's earlier answers to rebuild the committed surfaces.
        let viewCount = 0;
        for (let k = 0; k < m.committed.length; k++) {
          const earlier = meta.find(
            (e) => e.chain === m.chain && e.step === k,
          )!;
          const ops = compiler.parse(
            m.committed[k]!,
            earlier.surfaceId,
            committed,
          ).operations;
          committed = [...committed, ...ops];
          viewCount++;
        }
        void viewCount;
        const { operations } = compiler.parse(output, m.surfaceId, committed);
        if (!operations.length) throw new Error("no <a2ui> block");
        // The answer must keep the surface's root reachable: a new view or an update that produced operations counts.
        report.push({
          index,
          chain: m.chain,
          step: m.step,
          kind: m.kind,
          ok: true,
          ops: operations.length,
        });
      } catch (error) {
        report.push({
          index,
          chain: m.chain,
          step: m.step,
          kind: m.kind,
          ok: false,
          error:
            error instanceof Error
              ? error.message.slice(0, 200)
              : String(error),
        });
      }
    }
    const byKind: Record<string, { ok: number; total: number }> = {};
    for (const r of report) {
      byKind[r.kind] ??= { ok: 0, total: 0 };
      byKind[r.kind]!.total++;
      if (r.ok) byKind[r.kind]!.ok++;
    }
    const summary = {
      adapter: outputs.adapter,
      pass: report.filter((r) => r.ok).length,
      total: report.length,
      byKind,
      failures: report.filter((r) => !r.ok),
    };
    writeFileSync(
      OUTS.replace(/\.json$/, "-compiled.json"),
      JSON.stringify(summary, null, 1),
    );
    console.log(
      JSON.stringify(
        {
          adapter: summary.adapter,
          pass: summary.pass,
          total: summary.total,
          byKind,
        },
        null,
        1,
      ),
    );
    void rows;
    expect(true).toBe(true);
  });
});
