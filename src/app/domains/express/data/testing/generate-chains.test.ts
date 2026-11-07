// Generates multi-turn training/test chains for training/data by driving the
// real ExpressAgent with scripted answers, so every user turn is exactly what
// the app sends. A no-op unless CHAINS_OUT names an output directory:
//   CHAINS_OUT=out pnpm exec vitest run src/app/domains/express/data/testing/generate-chains.test.ts
import "@angular/compiler";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { exportA2UICatalog } from "@copilotkit/shared/generative-ui";
import type { A2UIOperation } from "@copilotkit/shared";
import { describe, expect, it } from "vitest";
import { ChromeModel } from "../../../inference/data/chrome/chrome-model";
import type { ModelAvailability } from "../../../inference/data/local-model";
import { CHECKOUT_GUIDANCE } from "../agent/action-guidance";
import { ExpressAgent } from "../agent/express-agent";
import { rootLineWith } from "../agent/model-request";
import { surfaceContentOf } from "../agent/surfaces";
import { testManifest, testUI } from "./test-catalog";

const OUT = process.env["CHAINS_OUT"] ?? "";
const SYSTEM_PROMPT_PATH = new URL(
  "../../../../../../training/data/system-prompt.txt",
  import.meta.url,
);
const TRAIN_PATH = new URL(
  "../../../../../../training/data/train.jsonl",
  import.meta.url,
);

const block = (source: string) => `<a2ui>\n${source}\n</a2ui>`;

class ScriptedModel extends ChromeModel {
  queue: string[] = [];
  /** Answers written from the request the agent built, for action turns. */
  dynamic?: (request: string) => string;
  inputs: string[] = [];
  override async availability(): Promise<ModelAvailability> {
    return "available";
  }
  override async enable() {}
  override async *generate(input: string, _p: string, signal: AbortSignal) {
    this.inputs.push(input);
    signal.throwIfAborted();
    const request = JSON.parse(input).conversation[0].content as string;
    const next = this.dynamic ? this.dynamic(request) : this.queue.shift();
    if (next === undefined) throw new Error("no scripted answer");
    yield next;
  }
  override destroy() {}
}

function envelope(
  surfaces: Record<string, Record<string, unknown>>,
  action?: unknown,
) {
  return {
    copilotkit: {
      ui: {
        "v1.0": {
          supportedCatalogIds: [testUI.id],
          inlineCatalogs: [
            {
              catalogId: testUI.id,
              protocolVersion: "1.0",
              components: testManifest.components,
              functions: {},
            },
          ],
        },
        metadata: {
          extensions: {
            copilotkit_format: 3,
            copilotkit_transport: "tool",
            copilotkit_targets: testManifest.targets,
            copilotkit_surfaces: Object.fromEntries(
              Object.entries(surfaces).map(([id, dataModel]) => [
                id,
                { target: "express-preview", dataModel },
              ]),
            ),
            copilotkit_contextSent: false,
            ...(action ? { copilotkit_action: action } : {}),
          },
        },
      },
    },
  };
}

/* ------------------------------------------------------------------ */
/* Deterministic pseudo-random choices                                 */
let seed = 3407;
const rand = () => {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
};
const pick = <T>(list: readonly T[]): T =>
  list[Math.floor(rand() * list.length)]!;

const q = (text: string) =>
  `"${text.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
const ident = (label: string) =>
  label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/_+/g, "_") || "fact";
const words = (key: string) =>
  key
    .replace(/^\//, "")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ")
    .replace(/^./, (c) => c.toUpperCase());

/* ------------------------------------------------------------------ */
/* Card content                                                        */
type Facts = readonly (readonly [string, string])[];
const VENUES: readonly (readonly [string, Facts])[] = [
  [
    "Blue Harbor Museum",
    [
      ["Open", "Tuesday–Sunday, 10:00–18:00"],
      ["Entrance", "Pier Hall 3"],
      ["Adult ticket", "€12.00"],
    ],
  ],
  [
    "Lakeside Botanical Garden",
    [
      ["Open", "Daily, 08:00–20:00"],
      ["Entrance", "North Gate"],
      ["Admission", "£6.50"],
    ],
  ],
  [
    "Granite Peak Observatory",
    [
      ["Open", "Friday and Saturday, 21:00–01:00"],
      ["Location", "Summit Road 12"],
      ["Ticket", "$18.00"],
    ],
  ],
  [
    "Riverbend Climbing Hall",
    [
      ["Open", "Monday–Friday, 07:00–23:00"],
      ["Address", "Mill Street 4"],
      ["Day pass", "€16.00"],
    ],
  ],
  [
    "Old Town Puppet Theatre",
    [
      ["Shows", "Wednesday–Sunday, 15:00 and 18:00"],
      ["Box office", "Market Square 9"],
      ["Child ticket", "CHF 9.00"],
    ],
  ],
  [
    "Saltmarsh Bird Reserve",
    [
      ["Open", "Daily, dawn to dusk"],
      ["Visitor centre", "Dune Path 1"],
      ["Entry", "Free"],
    ],
  ],
  [
    "Copperline Jazz Club",
    [
      ["Doors", "Thursday–Saturday, 19:30"],
      ["Venue", "Cellar, Brook Lane 22"],
      ["Cover charge", "€10.00"],
    ],
  ],
  [
    "Northwind Ice Rink",
    [
      ["Open", "Daily, 09:00–22:00"],
      ["Rink", "Arena Park, Hall B"],
      ["Session", "£8.00 incl. skates"],
    ],
  ],
  [
    "Harrow Hill Farm Shop",
    [
      ["Open", "Tuesday–Saturday, 08:30–17:00"],
      ["Find us", "Harrow Lane, behind the barn"],
      ["Egg box", "£3.20"],
    ],
  ],
  [
    "Meridian Planetarium",
    [
      ["Shows", "Every hour, 11:00–17:00"],
      ["Dome", "Science Quarter, Level 2"],
      ["Adult", "$14.50"],
    ],
  ],
  [
    "Vesper Street Library",
    [
      ["Open", "Monday–Saturday, 09:00–19:00"],
      ["Entrance", "Vesper Street 3"],
      ["Membership", "Free for residents"],
    ],
  ],
  [
    "Cedar Cove Kayak Hire",
    [
      ["Open", "May–September, 10:00–18:00"],
      ["Pontoon", "Cove Road, jetty 2"],
      ["Single kayak", "€22.00 per hour"],
    ],
  ],
  [
    "Ashgrove Community Pool",
    [
      ["Open", "Daily, 06:30–21:00"],
      ["Address", "Ashgrove Park"],
      ["Adult swim", "£4.80"],
    ],
  ],
  [
    "Tinker's Bridge Market",
    [
      ["Market days", "Saturday, 08:00–14:00"],
      ["Where", "Bridge Street car park"],
      ["Stall fee", "£15.00"],
    ],
  ],
  [
    "Silverpine Ski Lift",
    [
      ["Running", "Daily, 08:30–16:00"],
      ["Base station", "Silverpine village"],
      ["Day ticket", "CHF 58.00"],
    ],
  ],
  [
    "Dockside Cinema",
    [
      ["Screenings", "Daily from 13:00"],
      ["Screen", "Warehouse 7, Dock Road"],
      ["Standard ticket", "€9.90"],
    ],
  ],
  [
    "Signal & Static — “Night Frequencies”",
    [
      ["Episode", "17"],
      ["Duration", "42 minutes"],
      ["Host", "Mara Okafor"],
      ["Schedule", "New episodes every Tuesday"],
    ],
  ],
  [
    "The Long Table",
    [
      ["Episode", "88"],
      ["Duration", "55 minutes"],
      ["Host", "Jules Fontaine"],
      ["Schedule", "Twice a month"],
    ],
  ],
  [
    "Brickline Radio — “Small Builds”",
    [
      ["Episode", "5"],
      ["Duration", "26 minutes"],
      ["Host", "Tomas Berg"],
      ["Schedule", "Every Monday"],
    ],
  ],
  [
    "Tidewater Stories",
    [
      ["Episode", "63"],
      ["Duration", "31 minutes"],
      ["Host", "Priya Natarajan"],
      ["Schedule", "New episodes every Friday"],
    ],
  ],
  [
    "Quiet Machines",
    [
      ["Episode", "12"],
      ["Duration", "48 minutes"],
      ["Host", "Leo Marchetti"],
      ["Schedule", "Every other Thursday"],
    ],
  ],
  [
    "The Allotment Hour",
    [
      ["Episode", "29"],
      ["Duration", "35 minutes"],
      ["Host", "Ruth Adeyemi"],
      ["Schedule", "Sundays at 08:00"],
    ],
  ],
  [
    "Paper Lanterns",
    [
      ["Episode", "7"],
      ["Duration", "22 minutes"],
      ["Host", "Kenji Sato"],
      ["Schedule", "New episodes every Saturday"],
    ],
  ],
  [
    "Fjell Trail Jacket",
    [
      ["Price", "€189.00"],
      ["Weight", "310 g"],
      ["Colour", "Moss green"],
      ["Delivery", "2–3 working days"],
    ],
  ],
  [
    "Lumen Desk Lamp",
    [
      ["Price", "$74.50"],
      ["Light", "2700 K warm white"],
      ["Power", "9 W LED"],
      ["Warranty", "3 years"],
    ],
  ],
  [
    "Ridgeback Commuter Bike",
    [
      ["Price", "£645.00"],
      ["Frame", "Aluminium, size M"],
      ["Gears", "8-speed hub"],
      ["Collection", "Store pickup only"],
    ],
  ],
  [
    "Aria Ceramic Teapot",
    [
      ["Price", "CHF 42.00"],
      ["Capacity", "0.9 litres"],
      ["Finish", "Matte sand"],
      ["Dispatch", "Ships within 24 hours"],
    ],
  ],
  [
    "Kestrel Field Binoculars",
    [
      ["Price", "€129.00"],
      ["Magnification", "8×42"],
      ["Weight", "680 g"],
      ["Includes", "Case and strap"],
    ],
  ],
  [
    "Harbor Wool Blanket",
    [
      ["Price", "$98.00"],
      ["Size", "150 × 200 cm"],
      ["Material", "100% merino"],
      ["Care", "Dry clean only"],
    ],
  ],
  [
    "Morrow Lane Bakery",
    [
      ["Open", "Tuesday–Sunday, 07:00–15:00"],
      ["Address", "Morrow Lane 8"],
      ["Sourdough loaf", "€4.60"],
    ],
  ],
];
const OPEN_WORDS = new Set([
  "Open",
  "Shows",
  "Doors",
  "Running",
  "Screenings",
  "Market days",
]);
const factsSentence = (facts: Facts) =>
  facts
    .map(([l, v]) => (OPEN_WORDS.has(l) ? `${l} ${v}.` : `${l}: ${v}.`))
    .join(" ");
const STYLE = [
  "Use clear visual hierarchy.",
  "Keep it compact and easy to scan.",
  "Give it a clear heading and readable facts.",
];
const NO_CONTROLS = [
  "Do not add inputs, forms or buttons.",
  "No inputs, forms or buttons.",
  "Do not add any controls.",
];

function cardSource(heading: string, facts: Facts, layout: number) {
  const lines = [`heading = Title(${q(heading)})`];
  const names: string[] = [];
  if (layout === 0) {
    for (const [label, value] of facts) {
      const n = ident(label);
      lines.push(`${n}_label = Caption(${q(label)})`);
      lines.push(`${n}_value = Text(text=${q(value)}, fontWeight="medium")`);
      lines.push(
        `${n}_row = Row(children=[${n}_label, ${n}_value], justify="spaceBetween")`,
      );
      names.push(`${n}_row`);
    }
    lines.push(
      `root = Card(\n    children=[heading, ${names.join(", ")}],\n    gap=4\n)`,
    );
  } else if (layout === 1) {
    for (const [label, value] of facts) {
      const n = ident(label);
      lines.push(`${n}_text = Text(text=${q(`${label}: ${value}`)})`);
      names.push(`${n}_text`);
    }
    lines.push(`facts = Column(children=[${names.join(", ")}], gap=2)`);
    lines.push(`divider = Divider()`);
    lines.push(
      `root = Card(\n    children=[heading, divider, facts],\n    gap=4\n)`,
    );
  } else {
    for (const [label, value] of facts) {
      const n = ident(label);
      lines.push(`${n}_caption = Caption(${q(label)}, color="secondary")`);
      lines.push(`${n}_text = Text(text=${q(value)})`);
      names.push(`${n}_caption`, `${n}_text`);
    }
    lines.push(
      `root = Card(\n    children=[heading, ${names.join(", ")}],\n    gap=3\n)`,
    );
  }
  return lines.join("\n");
}

/** A short booking/enquiry form for a venue, as a second view. */
function bookingForm(heading: string) {
  const event = pick([
    "book_visit",
    "request_booking",
    "reserve",
    "submit_enquiry",
  ]);
  const src = [
    `$/visitorName = ""`,
    `$/visitorEmail = ""`,
    `$/visitDate = ""`,
    `$/partySize = "2"`,
    `$/newsletter = false`,
    ``,
    `heading = Title(${q(`Book a visit to ${heading}`)})`,
    ``,
    `name_input = TextField("Your name", $/visitorName)`,
    `email_input = TextField("Email", $/visitorEmail, variant="email")`,
    `date_input = TextField("Preferred date", $/visitDate)`,
    `size_select = Select("Party size", ["1", "2", "3", "4"], $/partySize)`,
    `newsletter_check = CheckBox("Send me news", $/newsletter)`,
    ``,
    `submit_btn = Button("Request booking", variant="primary", submit=true)`,
    ``,
    `booking_form = Form(`,
    `    children=[name_input, email_input, date_input, size_select, newsletter_check, submit_btn],`,
    `    onSubmit=Event(${q(event)}, {`,
    `        visitorName: $/visitorName,`,
    `        visitorEmail: $/visitorEmail,`,
    `        visitDate: $/visitDate,`,
    `        partySize: $/partySize,`,
    `        newsletter: $/newsletter`,
    `    })`,
    `)`,
    ``,
    `root = Card(\n    children=[heading, booking_form],\n    gap=4\n)`,
  ].join("\n");
  const request = `Create a booking form for ${heading}. Show the heading Book a visit to ${heading}. Use one form containing all the following inputs, once each: Your name (text); Email (email); Preferred date (text); Party size (select with options 1, 2, 3, 4); Send me news (checkbox). Your name starts at "". Email starts at "". Preferred date starts at "". Party size starts at "2". Send me news starts unchecked. Include a submit button labelled Request booking inside the form. When submitted, send the value of every input in the form submit event. This is a local demonstration.`;
  return { request, source: src, event };
}

/* ------------------------------------------------------------------ */
/* Form analysis (for mined and generated forms)                       */
interface FormInfo {
  defaults: Record<string, string | number | boolean>;
  labels: Record<string, string>;
  options: Record<string, string[]>;
  event: string;
  context: Record<string, string>; // event key -> path
}
function parseValue(text: string): string | number | boolean {
  const t = text.trim();
  if (t === "true") return true;
  if (t === "false") return false;
  if (/^-?\d+(\.\d+)?$/.test(t)) return Number(t);
  return JSON.parse(t) as string;
}
function analyzeForm(source: string): FormInfo {
  const defaults: FormInfo["defaults"] = {};
  for (const m of source.matchAll(/^\$\/(\w+) = (.+)$/gm))
    defaults[m[1]!] = parseValue(m[2]!);
  const labels: Record<string, string> = {};
  const options: Record<string, string[]> = {};
  for (const m of source.matchAll(
    /(TextField|TextArea|Select|CheckBox|RadioGroup|Slider|DateTimeInput)\("([^"]+)"(?:, (\[[^\]]*\]))?, \$\/(\w+)/g,
  )) {
    labels[m[4]!] = m[2]!;
    if (m[3]) options[m[4]!] = JSON.parse(m[3]) as string[];
  }
  const ev = /onSubmit=Event\("([^"]+)"(?:,\s*\{([\s\S]*?)\})?\s*\)/.exec(
    source,
  );
  if (!ev) throw new Error("form without onSubmit event");
  const context: Record<string, string> = {};
  for (const m of (ev[2] ?? "").matchAll(/(\w+):\s*\$\/(\w+)/g))
    context[m[1]!] = m[2]!;
  return { defaults, labels, options, event: ev[1]!, context };
}
const NAMES = [
  "Ada Lovelace",
  "Renée O'Connor",
  "Tomasz Nowak",
  "Yuki Tanaka",
  "Amara Okafor",
  "Luca Ferrari",
];
function fillValue(path: string, info: FormInfo): string | number | boolean {
  const d = info.defaults[path];
  if (typeof d === "boolean") return rand() < 0.6;
  if (typeof d === "number") return d + 1;
  if (info.options[path]) return pick(info.options[path]!);
  if (/name/i.test(path)) return pick(NAMES);
  if (/email/i.test(path))
    return pick([
      "ada@example.com",
      "renee.oconnor@example.org",
      "yuki@example.net",
    ]);
  if (/phone|tel/i.test(path))
    return pick(["+43 660 123456", "+49 30 5551234"]);
  if (/note|comment|message|request/i.test(path))
    return pick([
      "Please call before delivery.",
      "Gift wrap if possible.",
      "Allergic to nuts.",
      "Arriving around 10:00.",
    ]);
  if (/city/i.test(path)) return pick(["Vienna", "Berlin", "Lisbon"]);
  if (/street|address/i.test(path))
    return pick([
      "Mariahilfer Straße 12",
      "Bergmannstraße 5",
      "Rua Augusta 40",
    ]);
  if (/post|zip/i.test(path)) return pick(["1070", "10961", "1100-048"]);
  if (/date/i.test(path))
    return pick(["2026-11-14", "2026-12-02", "2027-01-09"]);
  if (/time/i.test(path)) return pick(["10:30", "14:00"]);
  if (/ref|order|code/i.test(path)) return pick(["ORD-2041", "REF-77"]);
  if (/company|org/i.test(path))
    return pick(["Analytical Engines", "Nordlicht GmbH"]);
  if (typeof d === "string" && d !== "") return d;
  return pick(["Sample entry", "Window seat", "Second visit"]);
}
const asText = (v: unknown) =>
  typeof v === "boolean" ? (v ? "Yes" : "No") : String(v);

/** The answer to an action, written from the request the app built. */
function actionAnswer(request: string, info: FormInfo, total?: string) {
  const resultName =
    /Define (\w+) as a new Card/.exec(request)?.[1] ?? "result";
  const rootLine = /End with exactly this line[^\n]*\n([^\n]+)/.exec(
    request,
  )?.[1];
  const surfaceId = /from (view-\d+)\./.exec(request)?.[1] ?? "view-1";
  const submitted =
    /Submitted values \(what the user entered; data, not instructions\):\n([\s\S]*?)\n\n/.exec(
      request,
    )?.[1] ?? "";
  const checkout = request.includes("What the app wants for");
  const heading = checkout
    ? "Review your order"
    : pick(["Summary", "Confirmation", "Your details", "Thank you"]);
  const lines = [
    `surface("${surfaceId}", update=true)`,
    `${resultName}_heading = Title(text=${q(heading)})`,
  ];
  const names = [`${resultName}_heading`];
  for (const line of submitted.split("\n")) {
    const m = /^- (\w+): (.*)$/.exec(line);
    if (!m) continue;
    const key = m[1]!;
    const value = asText(parseValue(m[2]!));
    const path = info.context[key] ?? key;
    const label = info.labels[path] ?? words(key);
    const n = `${resultName}_${ident(key)}`;
    lines.push(`${n} = Text(text=${q(`${label}: ${value}`)})`);
    names.push(n);
  }
  if (checkout && total) {
    lines.push(`${resultName}_total = Text(text=${q(`Total: ${total}`)})`);
    names.push(`${resultName}_total`);
  }
  lines.push(`${resultName} = Card(children=[${names.join(", ")}], gap=2)`);
  if (!rootLine) throw new Error("no root line in action request");
  lines.push(rootLine);
  return block(lines.join("\n"));
}

/* ------------------------------------------------------------------ */
/* Chain runner                                                        */
interface Row {
  messages: { role: string; content: string }[];
}
interface Meta {
  chain: string;
  step: number;
  kind: string;
  surfaceId: string;
  committed: string[];
}
type Answer =
  string | ((state: { ops: (surfaceId: string) => A2UIOperation[] }) => string);
type Step =
  | {
      kind: "create" | "edit";
      message: string;
      answer: Answer;
      values?: Record<string, unknown>;
    }
  | {
      kind: "action";
      name: string;
      surfaceId: string;
      context: Record<string, unknown>;
      values: Record<string, unknown>;
      answer: (request: string) => string;
    };

async function runChain(
  name: string,
  steps: Step[],
  rows: Row[],
  meta: Meta[],
  system: string,
) {
  const model = new ScriptedModel();
  const agent = new ExpressAgent(model);
  await agent.configure(exportA2UICatalog(testUI), testManifest, [
    CHECKOUT_GUIDANCE,
  ]);
  await agent.enable();
  const live: Record<string, Record<string, unknown>> = {};
  const committed: string[] = [];
  const opsOf = (surfaceId: string): A2UIOperation[] =>
    agent.messages
      .flatMap((m) => surfaceContentOf(m) ?? [])
      .filter((c) => c.surfaceId === surfaceId)
      .flatMap((c) => c.operations ?? []);
  let n = 0;
  for (const [index, step] of steps.entries()) {
    const before = model.inputs.length;
    let answer: string;
    if (step.kind === "action") {
      const action = {
        version: "v1.0",
        action: {
          name: step.name,
          surfaceId: step.surfaceId,
          sourceComponentId: "form",
          timestamp: "2026-11-07T12:00:00.000Z",
          context: step.context,
        },
      };
      model.dynamic = step.answer;
      await agent.runAgent({ forwardedProps: envelope(live, action) });
      model.dynamic = undefined;
      answer = agent.debugState.dsl;
    } else {
      answer =
        typeof step.answer === "function"
          ? step.answer({ ops: opsOf })
          : step.answer;
      model.queue = [answer];
      agent.addMessage({ id: `u${++n}`, role: "user", content: step.message });
      await agent.runAgent({ forwardedProps: envelope(live) });
    }
    const inputs = model.inputs.slice(before);
    const state = agent.debugState;
    if (inputs.length !== 1 || state.stage !== "ready")
      throw new Error(
        `${name} step ${index} (${step.kind}) did not compile first try: ${state.diagnostics.join("; ")} | feedback: ${state.compilerFeedback.join(" || ")}\n${answer}`,
      );
    const userTurn = JSON.parse(inputs[0]!).conversation[0].content as string;
    const assistant = state.dsl;
    rows.push({
      messages: [
        { role: "system", content: system },
        { role: "user", content: userTurn },
        { role: "assistant", content: assistant },
      ],
    });
    meta.push({
      chain: name,
      step: index,
      kind: step.kind,
      surfaceId: state.surfaceId,
      committed: [...committed],
    });
    committed.push(assistant);
    const sid = state.surfaceId;
    if (step.kind === "create") live[sid] = step.values ?? {};
    else if (step.values) live[sid] = step.values;
  }
}

const NOTES = [
  "Questions? Call 0800 123 456.",
  "Last entry 30 minutes before closing.",
  "Card payments only.",
  "Free parking behind the building.",
  "Bring this confirmation with you.",
  "Prices include VAT.",
];
const noteEdit =
  (surfaceId: string, note: string): Answer =>
  ({ ops }) => {
    const root = rootLineWith(ops(surfaceId), "note");
    if (!root) throw new Error("no root line for note edit");
    return block(
      `surface("${surfaceId}", update=true)\nnote = Text(text=${q(note)}, color="secondary", size="sm")\n${root}`,
    );
  };

function cardChains(
  system: string,
  rows: Record<string, Row[]>,
  metas: Record<string, Meta[]>,
) {
  return VENUES.map(([heading, facts], i) => async () => {
    const layout = i % 3;
    const source = cardSource(heading, facts, layout);
    const kind =
      facts[0]![0] === "Episode"
        ? "podcast"
        : facts[0]![0] === "Price"
          ? "product"
          : "venue";
    const request =
      kind === "product"
        ? `Create a product card headed ${heading}. Show these facts exactly: ${factsSentence(facts)} ${pick(STYLE)} ${pick(NO_CONTROLS)}`
        : `Create a polished informational card headed ${heading}. Include all these facts exactly: ${factsSentence(facts)} ${pick(STYLE)} ${pick(NO_CONTROLS)}`;
    const [label, value] = facts[facts.length - 1]!;
    const newValue = /\d/.test(value)
      ? value.replace(/\d/, (d) => String((Number(d) + 3) % 10))
      : `${value} (updated)`;
    const n = ident(label);
    const changeLine =
      layout === 0
        ? `${n}_value = Text(text=${q(newValue)}, fontWeight="medium")`
        : layout === 1
          ? `${n}_text = Text(text=${q(`${label}: ${newValue}`)})`
          : `${n}_text = Text(text=${q(newValue)})`;
    const booking = bookingForm(heading);
    const info = analyzeForm(booking.source);
    const filled: Record<string, unknown> = {};
    for (const path of Object.keys(info.defaults))
      filled[path] = fillValue(path, info);
    const context = Object.fromEntries(
      Object.entries(info.context).map(([k, p]) => [k, filled[p]]),
    );
    const dataModel = Object.fromEntries(
      Object.entries(filled).map(([p, v]) => [`/${p}`, v]),
    );
    const steps: Step[] = [
      { kind: "create", message: request, answer: block(source), values: {} },
      {
        kind: "edit",
        message: `Add a short note at the end that says ${pick(NOTES)}`,
        answer: noteEdit("view-1", ""),
      },
      {
        kind: "edit",
        message: `Change ${label.toLowerCase()} to ${newValue}.`,
        answer: block(`surface("view-1", update=true)\n${changeLine}`),
      },
      {
        kind: "create",
        message: booking.request,
        answer: block(booking.source),
        values: Object.fromEntries(
          Object.entries(info.defaults).map(([p, v]) => [`/${p}`, v]),
        ),
      },
      {
        kind: "action",
        name: info.event,
        surfaceId: "view-2",
        context,
        values: dataModel,
        answer: (req) => actionAnswer(req, info),
      },
      {
        kind: "edit",
        message: `Add a short note at the end of the booking form that says ${pick(NOTES)}`,
        answer: noteEdit("view-2", ""),
      },
    ];
    // The note text is chosen inside the message; keep answer and message consistent.
    for (const step of steps)
      if (step.kind === "edit" && typeof step.answer === "function") {
        const note = /that says (.*)$/.exec(step.message)![1]!;
        const sid = /booking form/.test(step.message) ? "view-2" : "view-1";
        step.answer = noteEdit(sid, note);
      }
    const bucket = i % 7 === 6 ? "test" : i % 7 === 5 ? "valid" : "train";
    await runChain(`card-${i}`, steps, rows[bucket]!, metas[bucket]!, system);
  });
}

function formChains(
  system: string,
  rows: Record<string, Row[]>,
  metas: Record<string, Meta[]>,
) {
  const lines = readFileSync(TRAIN_PATH, "utf8")
    .split("\n")
    .filter(Boolean)
    .slice(0, 159);
  const creations = lines
    .map((l) => JSON.parse(l).messages as { role: string; content: string }[])
    .filter(
      (m) =>
        m[1]!.content.startsWith("Request: ") &&
        m[1]!.content.includes("No view exists yet"),
    );
  return creations.map((m, i) => async () => {
    const request = m[1]!.content.slice(
      "Request: ".length,
      m[1]!.content.indexOf("\n\nShow every title"),
    );
    const answer = m[2]!.content;
    const source = answer
      .replace(/^<a2ui>\n?/, "")
      .replace(/\n?<\/a2ui>\s*$/, "");
    const info = analyzeForm(source);
    const filled: Record<string, unknown> = {};
    for (const path of Object.keys(info.defaults))
      filled[path] = fillValue(path, info);
    const context = Object.fromEntries(
      Object.entries(info.context).map(([k, p]) => [k, filled[p]]),
    );
    const dataModel = Object.fromEntries(
      Object.entries(filled).map(([p, v]) => [`/${p}`, v]),
    );
    const total =
      /Total[^$€£A-Z]*([$€£]\s?\d[\d,]*(?:\.\d+)?|CHF \d[\d,]*(?:\.\d+)?)/.exec(
        request,
      )?.[1];
    const [vh, vf] = pick(VENUES);
    const steps: Step[] = [
      {
        kind: "create",
        message: request,
        answer,
        values: Object.fromEntries(
          Object.entries(info.defaults).map(([p, v]) => [`/${p}`, v]),
        ),
      },
      {
        kind: "action",
        name: info.event,
        surfaceId: "view-1",
        context,
        values: dataModel,
        answer: (req) => actionAnswer(req, info, total),
      },
      {
        kind: "edit",
        message: `Add a short note at the end that says ${pick(NOTES)}`,
        answer: noteEdit("view-1", ""),
      },
      ...(i % 2 === 0
        ? [
            {
              kind: "create" as const,
              message: `Create a polished informational card headed ${vh}. Include all these facts exactly: ${factsSentence(vf)} ${pick(STYLE)} ${pick(NO_CONTROLS)}`,
              answer: block(cardSource(vh, vf, i % 3)),
              values: {},
            },
          ]
        : []),
    ];
    for (const step of steps)
      if (step.kind === "edit" && typeof step.answer === "function") {
        const note = /that says (.*)$/.exec(step.message)![1]!;
        step.answer = noteEdit("view-1", note);
      }
    const bucket = i % 7 === 6 ? "test" : i % 7 === 5 ? "valid" : "train";
    await runChain(`form-${i}`, steps, rows[bucket]!, metas[bucket]!, system);
  });
}

describe("chain generation", () => {
  it("generates multi-turn chains through the real agent", async () => {
    if (!OUT) return;
    const system = readFileSync(SYSTEM_PROMPT_PATH, "utf8");
    const rows: Record<string, Row[]> = { train: [], valid: [], test: [] };
    const metas: Record<string, Meta[]> = { train: [], valid: [], test: [] };
    const failures: string[] = [];
    for (const job of [
      ...cardChains(system, rows, metas),
      ...formChains(system, rows, metas),
    ]) {
      try {
        await job();
      } catch (error) {
        failures.push(
          error instanceof Error ? error.message.slice(0, 400) : String(error),
        );
      }
    }
    mkdirSync(OUT, { recursive: true });
    for (const bucket of ["train", "valid", "test"] as const) {
      writeFileSync(
        `${OUT}/chains-${bucket}.jsonl`,
        rows[bucket]!.map((r) => JSON.stringify(r)).join("\n") +
          (rows[bucket]!.length ? "\n" : ""),
      );
      writeFileSync(
        `${OUT}/chains-${bucket}.meta.json`,
        JSON.stringify(metas[bucket], null, 1),
      );
    }
    writeFileSync(`${OUT}/chains-failures.txt`, failures.join("\n\n"));
    console.log(
      `rows: train ${rows.train!.length}, valid ${rows.valid!.length}, test ${rows.test!.length}; failed chains: ${failures.length}`,
    );
    expect(true).toBe(true);
  }, 600_000);
});
