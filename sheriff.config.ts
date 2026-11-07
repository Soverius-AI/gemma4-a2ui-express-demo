import { noDependencies, sameTag } from "@softarc/sheriff-core";
import type { SheriffConfig } from "@softarc/sheriff-core";

/** The domain layout of src/app/domains, checked by `npm run lint:architecture`.
 * Layers depend downwards only (feature → ui → data → util), and a domain
 * depends only on itself, except that Express runs on the inference models. */
export const config: SheriffConfig = {
  enableBarrelLess: true,
  modules: {
    "src/app/domains/<domain>": {
      "feature-<name>": ["domain:<domain>", "type:feature"],
      "ui-<name>": ["domain:<domain>", "type:ui"],
      data: ["domain:<domain>", "type:data"],
      "util-<name>": ["domain:<domain>", "type:util"],
    },
  },
  depRules: {
    root: "*",
    "domain:express": [sameTag, "domain:inference"],
    "domain:inference": sameTag,
    "type:feature": ["type:ui", "type:data", "type:util"],
    "type:ui": ["type:data", "type:util"],
    "type:data": [sameTag, "type:util"],
    "type:util": noDependencies,
  },
};
