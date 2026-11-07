import {
  provideBrowserGlobalErrorListeners,
  provideZonelessChangeDetection,
} from "@angular/core";
import type { ApplicationConfig } from "@angular/core";
import { provideCopilotKit } from "@copilotkit/angular";
import { provideGenerativeUI } from "@copilotkit/angular/generative-ui";
import {
  EXPRESS_AGENT_ID,
  ExpressAgent,
} from "./domains/express/data/agent/express-agent";
import { EXPRESS_UI } from "./domains/express/data/express-ui";

/** CopilotKit with the browser-local Express agent: no runtime URL, server
 * agents or Inspector. */
export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZonelessChangeDetection(),
    provideGenerativeUI({ ui: EXPRESS_UI }),
    provideCopilotKit({
      licenseKey: "ck_pub_00000000000000000000000000000000",
      enableInspector: false,
      agents: { [EXPRESS_AGENT_ID]: new ExpressAgent() },
      suggestionsConfig: [],
    }),
  ],
};
