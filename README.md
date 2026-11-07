# Gemma 4 + A2UI Express

A standalone Angular 22 demo of **browser-local inference → A2UI Express → native CopilotKit UI**. Describe an interface in the chat: Gemma streams [A2UI Express](https://a2ui.org/specification/proposals/express/) source, the A2UI SDK compiles it into operations, and CopilotKit renders native Angular components. Submitting a generated form sends the entered values back to the same local agent, which answers inside the existing view.

There is no inference backend and no API key: the model runs in the browser (WebGPU) or in Chrome's built-in Prompt API. _Express_ is the UI language, not Express.js.

## Quick start

Node **24.15.0** (see `.nvmrc`) and pnpm 11 (`corepack enable` picks the version pinned in `package.json`).

```sh
pnpm install
pnpm start
```

Open <http://localhost:4210/> (`pnpm start --port 4211` if the port is taken) in desktop Chrome with WebGPU and `shader-f16`.

1. Pick a model, read the download notice, then **Download & enable**. Nothing is fetched before that.
2. Try an example (Aquarium card, Podcast card, Fitting form, Book delivery form) or write your own request.
3. Watch the Express source and the native preview; the debugger shows the compiled JSON, system prompt, agent state and form data.
4. Edit a generated form and submit it.

**Stop** cancels loading or generation. **Reset** starts a fresh conversation and keeps loaded weights. A failed or cancelled run restores the last valid view.

## Models

| Selection                               | Runtime                  | First download      |
| --------------------------------------- | ------------------------ | ------------------- |
| Gemma 4 E4B (default, untuned)          | Transformers.js · WebGPU | 4.9 GB              |
| Gemma 4 E4B · fine-tuned for Express    | Transformers.js · WebGPU | 4.9 GB              |
| Gemma 4 E2B · fine-tuned for Express    | Transformers.js · WebGPU | 3.1 GB              |
| WebLLM preview · Gemma 4 E2B fine-tuned | WebLLM · WebGPU          | 2.8 GB, self-hosted |
| Chrome Prompt API                       | Chrome's on-device model | Managed by Chrome   |

Models load from pinned Hugging Face revisions (see `src/app/domains/inference/data/model-catalog.ts`), and a fine-tune is only used if its published manifest names the pinned adapter. A selected model never falls back to another. Generation is greedy, text-only, capped at 3,072 new tokens, within a 32,768-token context that is checked before inference and never truncated.

The WebLLM option needs locally converted weights; [`model-tools/`](model-tools/README.md) builds and stages them. [`training/`](training/README.md) retrains the fine-tune with Unsloth.

## Scripts

| Command                  | Does                                             |
| ------------------------ | ------------------------------------------------ |
| `pnpm start`             | Dev server on port 4210                          |
| `pnpm test`              | Unit tests (Vitest)                              |
| `pnpm typecheck`         | Type-checks the app and its tests                |
| `pnpm lint:architecture` | Checks the domain boundaries with Sheriff        |
| `pnpm format`            | Formats with Prettier (`format:check` to verify) |
| `pnpm build`             | Production build in `dist/a2ui-express/`         |
| `pnpm preview`           | Serves the production build on port 4210         |
| `pnpm model:stage`       | Links converted WebLLM weights into the build    |

## Project structure

The app follows the [Angular Architects DDD layout](https://www.angulararchitects.io/en/blog/modern-architectures-with-angular-part-1-strategic-design-with-sheriff-and-standalone-components/): each domain has a `data` folder for its logic and `feature-*` folders for its use cases, with `ui-*` and `util-*` beside them.

```
src/
  main.ts, index.html, styles.css
  app/
    app.ts, app.config.ts            shell: root component and providers
    domains/
      express/                       generating UI with A2UI Express
        feature-playground/          the page and its session store
        ui-workbench/                presentational components
        data/
          agent/                     the AG-UI agent: requests, generation, previews
          compiler/                  SDK prompt, streaming compiler, line repair
          express-ui.ts              the catalog the fine-tunes were trained on
        util-highlight/              highlight.js grammar for Express
      inference/                     running models in the browser
        data/                        model catalog, selector, Chrome/Transformers.js/WebLLM adapters
public/                              favicon and logo
model-tools/                         WebLLM conversion (Python)
training/                            Unsloth retraining: data and one script
vendor/                              locally built CopilotKit and A2UI SDK packages
```

Layers depend downwards only (feature → ui → data → util), and the only dependency between domains is Express on inference. `sheriff.config.ts` encodes these rules.

`src/app/domains/express/data/express-ui.test.ts` checks that the app sends exactly the system prompt in `training/data/system-prompt.txt`, so the app and the training data cannot drift apart unnoticed.

## Deploying

`pnpm build` produces static files in `dist/a2ui-express/browser/`; serve them from any HTTPS host. The build includes `LICENSE`, `THIRD_PARTY_NOTICES.md` and `notices/`. Angular writes the bundled dependencies' licenses to `dist/a2ui-express/3rdpartylicenses.txt`; ship that file too. Use `--base-href` when hosting under a subdirectory.

## Bundled packages

The Angular generative-UI entry point this demo needs is not in a published CopilotKit release, so `vendor/` holds seven locally built CopilotKit npm tarballs, installed through `file:` dependencies and the overrides in `pnpm-workspace.yaml`. Their source revisions, patches and checksums are in [`vendor/README.md`](vendor/README.md). The A2UI SDK, `@a2ui/agent`, comes from npm. Its catalog-file helpers, which this demo never calls, import Node's `fs` and `path`; the pnpm patch in `patches/` adds a `browser` field that maps both to empty modules, so the bundler drops them from the browser build.

## License

The application source is MIT ([`LICENSE`](LICENSE)). Bundled packages, model weights and branding keep their own licenses and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The UI keeps the original demo's Soverius branding.

Model output is not guaranteed to be correct, and the forms execute no real orders or payments. Don't enter sensitive data just because inference is local.
