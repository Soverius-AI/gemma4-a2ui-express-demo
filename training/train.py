#!/usr/bin/env python3
"""Fine-tune Gemma 4 for A2UI Express with Unsloth.

Reproduces the recipe of the published adapters: LoRA rank 8, alpha 16, no
dropout, every language-model projection, Adam at a constant 1e-4, batch 1,
seed 3407, and loss on the assistant turn only. Every row's system turn is
replaced by data/system-prompt.txt, the prompt the app sends today.

    python train.py --size e2b            # full run into runs/e2b
    python train.py --size e2b --resume   # continue from the last checkpoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Base checkpoint, planned steps and checkpoint interval of each original run.
# The published adapters are checkpoint 100 (E2B) and checkpoint 40 (E4B).
PRESETS = {
    "e2b": {
        "model": "unsloth/gemma-4-E2B-it",
        "revision": "d36bdd3855c82a6a1a23f7c459b749f5724ae75d",
        "steps": 120,
        "save_every": 20,
    },
    "e4b": {
        "model": "unsloth/gemma-4-E4B-it",
        "revision": "4e22d7e59e078e63a14f351efdc5232ed366b621",
        "steps": 160,
        "save_every": 40,
    },
}
# Every projection of the language model, including Gemma 4's per-layer input
# gate and projection; the vision and audio towers stay frozen.
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "per_layer_input_gate",
    "per_layer_projection",
]
# Gemma 4's chat template opens turns with these; loss covers only the text
# after MODEL_TURN, up to and including the model's end-of-turn token.
USER_TURN = "<|turn>user\n"
MODEL_TURN = "<|turn>model\n"
MAX_SEQ_LENGTH = 12_288
SEED = 3407


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_splits() -> dict[str, list[list[dict]]]:
    """Both splits with the app's system prompt. Validation rows that repeat a
    training row are dropped, so validation loss measures unseen rows."""
    prompt = (DATA / "system-prompt.txt").read_text(encoding="utf-8")

    def rows(name: str) -> list[list[dict]]:
        lines = (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        conversations = [json.loads(line)["messages"] for line in lines]
        return [
            [{"role": "system", "content": prompt}, *messages[1:]]
            for messages in conversations
        ]

    train, valid = rows("train"), rows("valid")
    seen = {json.dumps(messages[1:]) for messages in train}
    unique = [messages for messages in valid if json.dumps(messages[1:]) not in seen]
    if len(unique) < len(valid):
        print(
            f"Dropped {len(valid) - len(unique)} validation row(s) also in train.jsonl"
        )
    return {"train": train, "valid": unique}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--size", choices=PRESETS, default="e2b")
    parser.add_argument("--steps", type=int, help="override the preset's step count")
    parser.add_argument(
        "--output", type=Path, help="run directory (default: runs/<size>)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue an interrupted run"
    )
    args = parser.parse_args()

    preset = PRESETS[args.size]
    steps = args.steps or preset["steps"]
    output = args.output or HERE / "runs" / args.size
    if output.exists() and not args.resume:
        raise SystemExit(f"{output} exists; pass --resume or choose another --output")
    splits = load_splits()

    # Unsloth patches transformers and trl on import, so it loads first.
    # isort: off
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    # isort: on

    model, tokenizer = FastModel.from_pretrained(
        model_name=preset["model"],
        revision=preset["revision"],
        use_exact_model_name=True,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        target_modules=TARGET_MODULES,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # On Apple Silicon, Unsloth's MLX backend returns the multimodal processor,
    # whose patched chat template adds a generation prompt and a stray space
    # in the system turn. The text tokenizer inside it renders the upstream
    # template, the one the app's tokenizer uses, so rows are rendered there.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    bos_token = getattr(text_tokenizer, "bos_token", None) or ""

    def dataset(conversations: list[list[dict]]) -> Dataset:
        texts = []
        for messages in conversations:
            text = text_tokenizer.apply_chat_template(messages, tokenize=False)
            # The trainer adds BOS again when it tokenizes.
            text = text.removeprefix(bos_token)
            if text.count(MODEL_TURN) != 1:
                raise ValueError(
                    "The chat template no longer marks the model turn as expected"
                )
            texts.append(text)
        return Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset(splits["train"]),
        eval_dataset=dataset(splits["valid"]),
        args=SFTConfig(
            output_dir=str(output / "checkpoints"),
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=steps,
            learning_rate=1e-4,
            lr_scheduler_type="constant",
            warmup_steps=0,
            optim="adamw_torch",
            weight_decay=0.0,  # plain Adam, as the original mlx_lm runs
            max_grad_norm=0.0,  # no gradient clipping either
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=preset["save_every"],
            save_strategy="steps",
            save_steps=preset["save_every"],
            seed=SEED,
            report_to="none",
        ),
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=USER_TURN, response_part=MODEL_TURN
    )
    trainer.train(resume_from_checkpoint=args.resume or None)

    adapter = output / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    run = {
        "size": args.size,
        "base": {"model": preset["model"], "revision": preset["revision"]},
        "steps": steps,
        "data_sha256": {path.name: sha256(path) for path in sorted(DATA.iterdir())},
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Adapter saved to {adapter}; checkpoints in {output / 'checkpoints'}")


if __name__ == "__main__":
    main()
