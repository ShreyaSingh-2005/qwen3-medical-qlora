"""QLoRA fine-tuning of Qwen3-0.6B on medical flashcards using Unsloth + TRL.

On Colab, first run:
    !pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install --upgrade --no-deps "xformers<0.0.29" trl peft accelerate bitsandbytes triton
"""
from unsloth import FastLanguageModel  # import unsloth first
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ---------------- Model ----------------
max_seq_length = 2048
dtype = None          # auto-detect (T4 -> float16)
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ---------------- LoRA ----------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
model.print_trainable_parameters()

# ---------------- Data ----------------
dataset = load_dataset(
    "lavita/medical-qa-datasets",
    "medical_meadow_medical_flashcards",
    split="train",
)
print(dataset)
print(dataset[0])


def to_conversation(example):
    user_content = example["instruction"]
    if example.get("input"):
        user_content += "\n\n" + example["input"]
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


dataset = dataset.map(to_conversation, remove_columns=dataset.column_names)


def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        for convo in examples["conversations"]
    ]
    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)
print(dataset[0]["text"][:800])

# ---------------- Training ----------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        max_length=max_seq_length,
        padding_free=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # use num_train_epochs=1 for a full run
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
print(f"GPU = {gpu_stats.name}. Max memory reserved so far = {start_mem} GB.")

trainer_stats = trainer.train()

used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
print(f"{trainer_stats.metrics['train_runtime']:.1f} seconds used for training.")
print(f"Peak reserved memory = {used_mem} GB.")

# ---------------- Save ----------------
model.save_pretrained("qwen3_0.6b_medical_qlora")
tokenizer.save_pretrained("qwen3_0.6b_medical_qlora")
print("Saved LoRA adapters to ./qwen3_0.6b_medical_qlora")
