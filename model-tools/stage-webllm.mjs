#!/usr/bin/env node
// Link a packaged WebLLM model into the production build so `npm run preview`
// can serve it. Only touches the ignored dist/ output; weights never enter Git.
// Usage: npm run model:stage -- <packaged-mlc-model-directory>
import {
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  symlinkSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const MODEL_ID = "gemma-4-e2b-express-54c7414491e3-q4f16_1-MLC";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const browser = join(root, "dist/a2ui-express/browser");

function fail(message) {
  console.error(message);
  process.exit(1);
}

if (!process.argv[2])
  fail("Usage: npm run model:stage -- <packaged-mlc-model-directory>");
const model = resolve(process.argv[2]);
const manifestPath = join(model, "a2ui-mlc-manifest.json");
if (!existsSync(manifestPath))
  fail(`${manifestPath} not found; run a2ui-model-tools package first`);
const { modelId } = JSON.parse(readFileSync(manifestPath, "utf8"));
if (modelId !== MODEL_ID) fail(`Wrong model identity: ${modelId}`);
if (!existsSync(join(browser, "index.html"))) fail("Run npm run build first");

const destination = join(browser, "models/webllm-e2b-express/resolve/local");
if (lstatSync(destination, { throwIfNoEntry: false }))
  fail(`${destination} already exists; rebuild before restaging`);
mkdirSync(dirname(destination), { recursive: true });
symlinkSync(model, destination, "dir");
console.log(`Staged ${model} at ${destination}`);
console.log(
  "Building clears dist/: stage again after each build. Deploy real files, not this symlink.",
);
