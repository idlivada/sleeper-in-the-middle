#!/usr/bin/env python3
"""
Unsloth LoRA fine-tune for the transparent domain-migration dataset.

    pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" trl
    python train/train_unsloth.py

Trains on assistant turns only (loss masked on user/system) using the chat
template, on the same {"messages": [...]} JSONL the generator produces.

Everything is set up for a single consumer GPU via 4-bit QLoRA; adjust
BASE_MODEL / MAX_SEQ / epochs for your hardware.
"""

import inspect
from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only


def _accepts(cls) -> set:
    """Parameter names the class __init__ accepts (dataclass or regular)."""
    return set(inspect.signature(cls.__init__).parameters)


def build_sft_config(out_dir, max_seq, seed):
    """Build an SFTConfig, adapting to this TRL version's field names.

    TRL has renamed several fields across versions (max_seq_length ->
    max_length, evaluation_strategy -> eval_strategy, ...). We pass only what
    the installed SFTConfig accepts and report anything we had to drop, so a
    version bump degrades to defaults instead of crashing.
    """
    params = _accepts(SFTConfig)
    cfg = dict(
        output_dir=str(out_dir),
        dataset_text_field="text",
        packing=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=5,
        eval_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        seed=seed,
        report_to="none",
    )
    # (preferred_name, legacy_name, value) — use whichever this version has.
    for new, old, val in [
        ("max_length", "max_seq_length", max_seq),
        ("eval_strategy", "evaluation_strategy", "steps"),
    ]:
        if new in params:
            cfg[new] = val
        elif old in params:
            cfg[old] = val

    dropped = sorted(k for k in cfg if k not in params)
    for k in dropped:
        cfg.pop(k)
    if dropped:
        print(f"[warn] SFTConfig in this TRL ignores {dropped}; using its defaults")
    return SFTConfig(**cfg)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "domain-migration-lora"

BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct"
CHAT_TEMPLATE = "qwen-2.5"   # must match BASE_MODEL's template if you swap it
MAX_SEQ = 2048
SEED = 20260923


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        dtype=None,  # auto (bf16 where supported)
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)

    def format_chat(batch):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]
        return {"text": texts}

    train_ds = load_dataset("json", data_files=str(DATA / "train.jsonl"), split="train")
    eval_ds = load_dataset("json", data_files=str(DATA / "eval.jsonl"), split="train")
    train_ds = train_ds.map(format_chat, batched=True, remove_columns=["messages"])
    eval_ds = eval_ds.map(format_chat, batched=True, remove_columns=["messages"])

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=build_sft_config(OUT, MAX_SEQ, SEED),
    )
    # TRL renamed the tokenizer arg to processing_class; set whichever exists.
    tr_params = _accepts(SFTTrainer)
    if "processing_class" in tr_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in tr_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    # Mask loss to assistant responses only (Qwen im_start markers).
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()

    model.save_pretrained(str(OUT))        # LoRA adapter
    tokenizer.save_pretrained(str(OUT))
    print(f"saved adapter -> {OUT}")


if __name__ == "__main__":
    main()
