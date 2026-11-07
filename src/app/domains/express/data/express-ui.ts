import {
  createGenerativeUI,
  standardUI,
} from "@copilotkit/angular/generative-ui";

/** The native standard components under the catalog id the fine-tunes were
 * trained with: it is part of the system prompt, so keep it unchanged. */
export const EXPRESS_UI = createGenerativeUI({
  id: "copilotkit/demo-shop@1",
  extends: [standardUI],
  components: {},
});
