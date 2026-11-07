import {
  createGenerativeUI,
  standardDeclarations,
  withStandardParameterOrder,
} from "@copilotkit/shared/generative-ui";
import type { GenerativeUIManifest } from "@copilotkit/shared";
import { EXPRESS_TARGET } from "../agent/surfaces";

/** The standard components without renderers: tests compile Express and
 * never render it. */
export const testUI = createGenerativeUI({
  id: "express-test@1",
  // Built from entries, so its keys are only known as strings.
  components: Object.fromEntries(
    Object.entries(withStandardParameterOrder(standardDeclarations)).map(
      ([name, definition]) => [name, { ...definition, component: null }],
    ),
  ) as unknown as Parameters<typeof createGenerativeUI>[0]["components"],
});
export const testManifest: GenerativeUIManifest = {
  catalogId: testUI.id,
  components: testUI.manifestComponents,
  functions: {},
  targets: [{ id: EXPRESS_TARGET, kind: "inline" }],
  surfaces: {},
};
