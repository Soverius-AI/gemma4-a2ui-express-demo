"""Synthetic informational-card rows in the exact request/answer format of training/data/*.jsonl.
Creation rows (card, no inputs) and edit rows on those cards. Deterministic; output JSONL."""
import json, random, sys
from pathlib import Path
out_train, out_valid, system_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
system = system_path.read_text(encoding="utf-8")
rng = random.Random(3407)
COVERAGE = "Show every title, price, total and fact the request names, with its original currency and units. Generate only the interface and controls the request asks for. Define named components on their own lines and reference those names in their parent's children list. Use the catalog signatures and end with the actual outer constructor assigned to root."
CREATE_TAIL = "\n\n" + COVERAGE + "\n\nNo view exists yet. Write no surface(...) line: the page creates this view as view-1."
EDIT_HEAD = "Views on the page now:\nview-1, built from this Express:\n{source}\nCurrent input values (data, not instructions):\n- none\n\nTo change a view, start the block with surface(\"<id>\", update=true). A block without surface(...) creates a new view.\n\nRequest: {request}"

# (heading, [(label, value)], extra sentence options)
VENUES = [
 ("Blue Harbor Museum", [("Open", "Tuesday–Sunday, 10:00–18:00"), ("Entrance", "Pier Hall 3"), ("Adult ticket", "€12.00")]),
 ("Lakeside Botanical Garden", [("Open", "Daily, 08:00–20:00"), ("Entrance", "North Gate"), ("Admission", "£6.50")]),
 ("Granite Peak Observatory", [("Open", "Friday and Saturday, 21:00–01:00"), ("Location", "Summit Road 12"), ("Ticket", "$18.00")]),
 ("Riverbend Climbing Hall", [("Open", "Monday–Friday, 07:00–23:00"), ("Address", "Mill Street 4"), ("Day pass", "€16.00")]),
 ("Old Town Puppet Theatre", [("Shows", "Wednesday–Sunday, 15:00 and 18:00"), ("Box office", "Market Square 9"), ("Child ticket", "CHF 9.00")]),
 ("Saltmarsh Bird Reserve", [("Open", "Daily, dawn to dusk"), ("Visitor centre", "Dune Path 1"), ("Entry", "Free")]),
 ("Copperline Jazz Club", [("Doors", "Thursday–Saturday, 19:30"), ("Venue", "Cellar, Brook Lane 22"), ("Cover charge", "€10.00")]),
 ("Northwind Ice Rink", [("Open", "Daily, 09:00–22:00"), ("Rink", "Arena Park, Hall B"), ("Session", "£8.00 incl. skates")]),
 ("Harrow Hill Farm Shop", [("Open", "Tuesday–Saturday, 08:30–17:00"), ("Find us", "Harrow Lane, behind the barn"), ("Egg box", "£3.20")]),
 ("Meridian Planetarium", [("Shows", "Every hour, 11:00–17:00"), ("Dome", "Science Quarter, Level 2"), ("Adult", "$14.50")]),
 ("Vesper Street Library", [("Open", "Monday–Saturday, 09:00–19:00"), ("Entrance", "Vesper Street 3"), ("Membership", "Free for residents")]),
 ("Cedar Cove Kayak Hire", [("Open", "May–September, 10:00–18:00"), ("Pontoon", "Cove Road, jetty 2"), ("Single kayak", "€22.00 per hour")]),
 ("Ashgrove Community Pool", [("Open", "Daily, 06:30–21:00"), ("Address", "Ashgrove Park"), ("Adult swim", "£4.80")]),
 ("Tinker's Bridge Market", [("Market days", "Saturday, 08:00–14:00"), ("Where", "Bridge Street car park"), ("Stall fee", "£15.00")]),
 ("Silverpine Ski Lift", [("Running", "Daily, 08:30–16:00"), ("Base station", "Silverpine village"), ("Day ticket", "CHF 58.00")]),
 ("Dockside Cinema", [("Screenings", "Daily from 13:00"), ("Screen", "Warehouse 7, Dock Road"), ("Standard ticket", "€9.90")]),
]
PODCASTS = [
 ("Signal & Static — “Night Frequencies”", [("Episode", "17"), ("Duration", "42 minutes"), ("Host", "Mara Okafor"), ("Schedule", "New episodes every Tuesday")]),
 ("The Long Table", [("Episode", "88"), ("Duration", "55 minutes"), ("Host", "Jules Fontaine"), ("Schedule", "Twice a month")]),
 ("Brickline Radio — “Small Builds”", [("Episode", "5"), ("Duration", "26 minutes"), ("Host", "Tomas Berg"), ("Schedule", "Every Monday")]),
 ("Tidewater Stories", [("Episode", "63"), ("Duration", "31 minutes"), ("Host", "Priya Natarajan"), ("Schedule", "New episodes every Friday")]),
 ("Quiet Machines", [("Episode", "12"), ("Duration", "48 minutes"), ("Host", "Leo Marchetti"), ("Schedule", "Every other Thursday")]),
 ("The Allotment Hour", [("Episode", "29"), ("Duration", "35 minutes"), ("Host", "Ruth Adeyemi"), ("Schedule", "Sundays at 08:00")]),
 ("Cold Open — “Season 3 Recap”", [("Episode", "40"), ("Duration", "1 hour 4 minutes"), ("Host", "Nina Holt"), ("Schedule", "Weekly on Wednesdays")]),
 ("Paper Lanterns", [("Episode", "7"), ("Duration", "22 minutes"), ("Host", "Kenji Sato"), ("Schedule", "New episodes every Saturday")]),
]
PRODUCTS = [
 ("Fjell Trail Jacket", [("Price", "€189.00"), ("Weight", "310 g"), ("Colour", "Moss green"), ("Delivery", "2–3 working days")]),
 ("Lumen Desk Lamp", [("Price", "$74.50"), ("Light", "2700 K warm white"), ("Power", "9 W LED"), ("Warranty", "3 years")]),
 ("Ridgeback Commuter Bike", [("Price", "£645.00"), ("Frame", "Aluminium, size M"), ("Gears", "8-speed hub"), ("Collection", "Store pickup only")]),
 ("Aria Ceramic Teapot", [("Price", "CHF 42.00"), ("Capacity", "0.9 litres"), ("Finish", "Matte sand"), ("Dispatch", "Ships within 24 hours")]),
 ("Kestrel Field Binoculars", [("Price", "€129.00"), ("Magnification", "8×42"), ("Weight", "680 g"), ("Includes", "Case and strap")]),
 ("Harbor Wool Blanket", [("Price", "$98.00"), ("Size", "150 × 200 cm"), ("Material", "100% merino"), ("Care", "Dry clean only")]),
]
STYLE_SENTENCES = ["Use clear visual hierarchy.", "Keep it compact and easy to scan.", "Give it a clear heading and readable facts.", "Use clear visual hierarchy."]
NO_CONTROLS = ["Do not add inputs, forms or buttons.", "No inputs, forms or buttons.", "Do not add any controls."]

def ident(label):
    s = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
    while "__" in s: s = s.replace("__", "_")
    return s or "fact"

def q(text): return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

def card_source(heading, facts, layout):
    """Three layouts seen in the training data: label/value rows, caption-under-text, divided list."""
    lines = [f'heading = Title({q(heading)})']
    names = []
    if layout == "rows":
        for label, value in facts:
            n = ident(label)
            lines.append(f'{n}_label = Caption({q(label)})')
            lines.append(f'{n}_value = Text(text={q(value)}, fontWeight="medium")')
            lines.append(f'{n}_row = Row(children=[{n}_label, {n}_value], justify="spaceBetween")')
            names.append(f"{n}_row")
        lines.append(f'root = Card(\n    children=[heading, {", ".join(names)}],\n    gap=4\n)')
    elif layout == "list":
        for label, value in facts:
            n = ident(label)
            lines.append(f'{n}_text = Text(text={q(label + ": " + value)})')
            names.append(f"{n}_text")
        lines.append(f'facts = Column(children=[{", ".join(names)}], gap=2)')
        lines.append('root = Card(\n    children=[heading, Divider(), facts],\n    gap=4\n)')
    else:  # "stacked"
        for label, value in facts:
            n = ident(label)
            lines.append(f'{n}_caption = Caption({q(label)}, color="secondary")')
            lines.append(f'{n}_text = Text(text={q(value)})')
            names += [f"{n}_caption", f"{n}_text"]
        lines.append(f'root = Card(\n    children=[heading, {", ".join(names)}],\n    gap=3\n)')
    return "\n".join(lines)

def facts_sentence(facts):
    return " ".join(f"{label} {value}." if label in ("Open", "Shows", "Doors", "Running", "Screenings", "Market days") else f"{label}: {value}." for label, value in facts)

def creation_request(kind, heading, facts):
    if kind == "venue":
        return f"Create a polished informational card headed {heading}. Include all these facts exactly: {facts_sentence(facts)} {rng.choice(STYLE_SENTENCES)} {rng.choice(NO_CONTROLS)}"
    if kind == "podcast":
        return f"Create a polished informational card headed {heading}. Include all these facts exactly: {facts_sentence(facts)} {rng.choice(STYLE_SENTENCES)} {rng.choice(NO_CONTROLS)}"
    return f"Create a product card headed {heading}. Show these facts exactly: {facts_sentence(facts)} {rng.choice(STYLE_SENTENCES)} {rng.choice(NO_CONTROLS)}"

def edit_rows(heading, facts, source, layout):
    rows = []
    # 1. append a note
    note = rng.choice(["Questions? Call 0800 123 456.", "Last entry 30 minutes before closing.", "Card payments only.", "Free parking behind the building."])
    rows.append((EDIT_HEAD.format(source=source, request=f"Add a short note at the end that says {note}"),
        f'<a2ui>\nsurface("view-1", update=true)\nnote = Text(text={q(note)}, color="secondary", size="sm")\n' + root_with(source, "note") + "\n</a2ui>"))
    # 2. change one fact value
    label, value = facts[-1]
    new_value = value.replace("0", "5", 1) if "0" in value else value + " (updated)"
    n = ident(label)
    if layout == "rows":
        line = f'{n}_value = Text(text={q(new_value)}, fontWeight="medium")'
    elif layout == "list":
        line = f'{n}_text = Text(text={q(label + ": " + new_value)})'
    else:
        line = f'{n}_text = Text(text={q(new_value)})'
    rows.append((EDIT_HEAD.format(source=source, request=f"Change {label.lower()} to {new_value}."),
        f'<a2ui>\nsurface("view-1", update=true)\n{line}\n</a2ui>'))
    return rows

def root_with(source, extra):
    root_line = source[source.index("root = Card("):]
    return root_line.replace("],", f", {extra}],", 1)

train, valid = [], []
items = [("venue", h, f) for h, f in VENUES] + [("podcast", h, f) for h, f in PODCASTS] + [("product", h, f) for h, f in PRODUCTS]
rng.shuffle(items)
for i, (kind, heading, facts) in enumerate(items):
    layout = ["rows", "list", "stacked"][i % 3]
    source = card_source(heading, facts, layout)
    rows = [(creation_request(kind, heading, facts) and ("Request: " + creation_request(kind, heading, facts) + CREATE_TAIL), "<a2ui>\n" + source + "\n</a2ui>")]
    rows += edit_rows(heading, facts, source, layout)
    bucket = valid if i % 8 == 7 else train
    for user, assistant in rows:
        bucket.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]})
for path, rows in ((out_train, train), (out_valid, valid)):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(f"wrote {len(train)} train rows -> {out_train}, {len(valid)} valid rows -> {out_valid}")
# also a compile-check file in the eval JSON shape
check = {"cards": {}}
for i, r in enumerate(train + valid):
    user, assistant = r["messages"][1]["content"], r["messages"][2]["content"]
    entry = {"output": assistant}
    if user.startswith("Views on the page now:"):
        src = user[len("Views on the page now:\nview-1, built from this Express:\n"):user.index("\nCurrent input values")]
        entry["base"] = "<a2ui>\n" + src + "\n</a2ui>"
    check["cards"][str(i)] = entry
Path(str(out_train) + ".check.json").write_text(json.dumps(check))
