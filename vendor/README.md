# Local package artifacts

These are **local snapshots, not npm releases**. Normal use requires only `pnpm install` in the application root; pnpm installs the tarballs directly from this directory.

## Provenance

- CopilotKit source: <https://github.com/CopilotKit/CopilotKit>, revision `114d64c00c762ca72992884273f83ddc009684e1` (upstream main when assembled).
- Original Express demo: revision `f68192bd0b07589b0286f1157a3c4003f58004fd` from the `demo/a2ui-1.0` branch of the source checkout, fork <https://github.com/Soverius-AI/CopilotKit>.
- Exact local artifact versions, filenames, and SHA-256 checksums: [`manifest.json`](manifest.json).

The seven CopilotKit packages are rebuilt together, using Angular 22.0.0 and TypeScript 6.0.3 for Angular partial compilation. The custom `-gemma-demo.0` versions distinguish these artifacts from official releases. Build artifacts include their licenses, public exports, runtime dependencies, and compiled distributions; development/build dependencies and monorepo workspace ranges are removed from the installable packages.

## Compatibility changes

[`patches/copilotkit-demo-compat.patch`](patches/copilotkit-demo-compat.patch) records source/test changes against that upstream revision:

1. Core surface manager: explicit A2UI 1.0 catalog/model construction for `@a2ui/web_core@0.15.0`; discard withdrawn, never-committed previews without removing settled views.
2. Angular native rendering: opt-in partial previews, including slot/nested-child filtering.
3. Angular slider: property-bound min/max/step so browser defaults do not clamp the initial bound value.
4. Shared A2UI parsing: accept legacy `v1.0` SDK operations/capabilities while preserving current canonical `1.0` generation, including regression tests.

## Included source

`package-source.tar.gz` contains the modified sources, tests and configuration of all seven CopilotKit packages, both shared TypeScript config packages, the canonical theme-token CSS input, a pinned producer lockfile with a minimal Nx workspace, and the published A2UI SDK input the producer build used (`@a2ui/agent@0.1.0`, which the app now installs from npm). The compatibility patch is already applied. It excludes `node_modules`, built `dist` folders, Git metadata and model weights.

To rebuild a package, extract the archive, run `pnpm install --frozen-lockfile` and the Nx build there (`pnpm@10.33.4`), `npm pack` the result, and update the checksums in `manifest.json` together with `package.json`, `pnpm-workspace.yaml` and `pnpm-lock.yaml`. Keep the tarballs, the archive, the patches and `manifest.json` in the repository: `pnpm install` installs straight from them.
