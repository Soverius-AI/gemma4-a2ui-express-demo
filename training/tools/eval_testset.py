"""Greedy-generate every row of a test-set JSONL with one adapter (mlx_lm on the BF16 base).
Usage: eval_testset.py <rows.jsonl> <out.json> <adapter|none|model:path> [EVAL_BASE env]
Outputs {"rows": [{"index", "output", "gen_s"}]} for the compile spec."""
import json, os, sys, time, shutil, tempfile
from pathlib import Path
rows_path, out_path, adapter = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
BASE = os.environ.get("EVAL_BASE", "unsloth/gemma-4-E2B-it")
limit = int(os.environ.get("EVAL_LIMIT", "0")) or None
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

def prepared(a: str) -> str:
    src = Path(a); cfg = json.loads((src / "adapter_config.json").read_text())
    if "num_layers" in cfg and "fine_tune_type" in cfg: return str(src)
    tmp = Path(tempfile.mkdtemp(prefix="adapter-")); shutil.copy(src / "adapters.safetensors", tmp)
    lp = cfg["lora_parameters"]
    (tmp / "adapter_config.json").write_text(json.dumps({"fine_tune_type": "lora", "num_layers": -1,
        "lora_parameters": {"rank": lp["rank"], "scale": lp["scale"], "dropout": lp.get("dropout", 0.0), **({"keys": lp["keys"]} if "keys" in lp else {})}}))
    return str(tmp)

if adapter.startswith("model:"):
    model, tok = load(adapter[6:])
else:
    model, tok = load(BASE, adapter_path=None if adapter == "none" else prepared(adapter))
rows = [json.loads(l) for l in rows_path.read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
out = {"adapter": adapter, "rows": []}
t_all = time.time()
for i, row in enumerate(rows):
    messages = row["messages"][:2]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    t0 = time.time()
    text = generate(model, tok, prompt=prompt, max_tokens=1200, sampler=make_sampler(temp=0.0), verbose=False)
    out["rows"].append({"index": i, "output": text, "gen_s": round(time.time() - t0, 1)})
    print(f"row {i}: {time.time()-t0:.1f}s, {len(text)} chars", flush=True)
    out_path.write_text(json.dumps(out, indent=1))
print(f"done {len(rows)} rows in {time.time()-t_all:.0f}s -> {out_path}")
