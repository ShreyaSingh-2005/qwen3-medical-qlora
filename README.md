# 🩺 Qwen3-0.6B Medical QA — QLoRA Fine-Tuning with Unsloth

Fine-tuning of **Qwen3-0.6B** (4-bit) on a medical question-answering dataset using **QLoRA**, **Unsloth** and **TRL's SFTTrainer**. Runs end-to-end on a free **Google Colab Tesla T4** GPU in about 2 minutes.

> ⚠️ **Disclaimer:** This is an educational project. The model is small and trained for only 60 steps. It must **not** be used for real medical advice or diagnosis.

---

## 📌 Overview

| Item | Details |
|---|---|
| Base model | `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` |
| Method | QLoRA (4-bit base + LoRA adapters) |
| Framework | Unsloth, TRL (`SFTTrainer`), PEFT, bitsandbytes |
| Dataset | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) → `medical_meadow_medical_flashcards` |
| Dataset size | 33,955 examples (`instruction`, `input`, `output`) |
| Hardware | Google Colab, Tesla T4 (14.5 GB) |

## 🧠 Pipeline

```
Medical flashcards (instruction/input/output)
        │
        ▼
Convert to chat format (user / assistant)
        │
        ▼
Apply Qwen3 chat template  →  "text" column
        │
        ▼
4-bit Qwen3-0.6B + LoRA adapters (r=16)
        │
        ▼
SFTTrainer (60 steps, AdamW 8-bit)
        │
        ▼
Save LoRA adapters  →  inference
```

## ⚙️ Configuration

**LoRA**
- Rank `r = 16`, `lora_alpha = 16`, `lora_dropout = 0`, `bias = "none"`
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Gradient checkpointing: `"unsloth"`
- Trainable params: **10,092,544 / 606,142,464 (1.665%)**

**Training**
- `max_seq_length = 2048`
- Batch size 2 × gradient accumulation 4 (effective batch size 8)
- Learning rate `2e-4`, linear scheduler, 5 warmup steps, weight decay 0.01
- Optimizer: `adamw_8bit`, `max_steps = 60`, seed `3407`

## 📊 Results

| Metric | Value |
|---|---|
| Training time | ~116 seconds (60 steps) |
| Peak reserved GPU memory | ~1.9 GB |
| Training loss (step 30 → 55) | 0.94 → 1.09 |

Loss was logged every 5 steps. With only 60 steps (a tiny fraction of one epoch over 33,955 examples), the loss is noisy and has not converged — train longer (`num_train_epochs = 1`) for better results.

### Sample training example

```
<|im_start|>user
Answer this question truthfully

What is the relationship between very low Mg2+ levels, PTH levels, and Ca2+ levels?<|im_end|>
<|im_start|>assistant
Very low Mg2+ levels correspond to low PTH levels which in turn results in low Ca2+ levels.<|im_end|>
```

## 🚀 Getting Started

### 1. Install
```bash
pip install -r requirements.txt
```
On Colab, use the Unsloth install cell from `train.py` / the notebook.

### 2. Train
```bash
python train.py
```
LoRA adapters are saved to `./qwen3_0.6b_medical_qlora`.

### 3. Inference
```bash
python inference.py --prompt "What is the function of insulin?"
```

## 📁 Project Structure

```
.
├── train.py            # Data prep + QLoRA training
├── inference.py        # Load adapters and generate
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🔭 Future Work
- Train for full epoch(s) and add a validation split
- Evaluate (e.g. ROUGE / BERTScore / human review) against the base model
- Merge adapters and export to GGUF / push to Hugging Face Hub
- Add a **RAG** layer on top of the fine-tuned model for grounded answers

## 🙏 Acknowledgements
- [Unsloth](https://github.com/unslothai/unsloth)
- [Hugging Face TRL / PEFT / Datasets](https://huggingface.co)
- [lavita/medical-qa-datasets](https://huggingface.co/datasets/lavita/medical-qa-datasets)
- [Qwen3](https://huggingface.co/Qwen)

## 📄 License
MIT — see [LICENSE](LICENSE).
