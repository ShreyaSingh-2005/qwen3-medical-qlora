# Medical QA Fine-Tuning with QLoRA

Fine-tuning **Qwen3-0.6B** on medical question-answering data using QLoRA, Unsloth and TRL. The full pipeline trains in about two minutes on a free Google Colab T4 GPU.

> **Disclaimer:** This is an educational project. The model is small and was trained for only 60 steps. It must not be used for medical advice or diagnosis.

## Overview

I built this project to understand how language models are fine-tuned in practice, from raw dataset to saved adapters.

Instead of retraining all 600 million parameters of the base model, QLoRA loads it in 4-bit precision and trains only a small set of added LoRA adapter weights. This keeps memory usage under 2 GB and makes training possible on free hardware.

## Summary

| Item | Details |
|---|---|
| Base model | `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` |
| Method | QLoRA (4-bit base model + LoRA adapters) |
| Libraries | Unsloth, TRL (`SFTTrainer`), PEFT, bitsandbytes |
| Dataset | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets), subset `medical_meadow_medical_flashcards` |
| Dataset size | 33,955 examples |
| Hardware | Google Colab, Tesla T4 (14.5 GB) |

## Pipeline

1. Load the medical flashcards (`instruction`, `input`, `output`).
2. Convert each example into a user/assistant conversation.
3. Apply the Qwen3 chat template to produce a single `text` column.
4. Load Qwen3-0.6B in 4-bit and attach LoRA adapters.
5. Train with `SFTTrainer` for 60 steps.
6. Save the adapters and run inference.

Example of a formatted training sample:

```
User:       Answer this question truthfully
            What is the relationship between very low Mg2+ levels,
            PTH levels, and Ca2+ levels?

Assistant:  Very low Mg2+ levels correspond to low PTH levels,
            which in turn results in low Ca2+ levels.
```

## Configuration

**LoRA**

| Parameter | Value |
|---|---|
| Rank (`r`) | 16 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0 |
| `bias` | none |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Gradient checkpointing | `unsloth` |

**Training**

| Parameter | Value |
|---|---|
| Max sequence length | 2048 |
| Per-device batch size | 2 |
| Gradient accumulation steps | 4 (effective batch size 8) |
| Learning rate | 2e-4, linear schedule |
| Warmup steps | 5 |
| Max steps | 60 |
| Optimizer | `adamw_8bit` |
| Weight decay | 0.01 |
| Seed | 3407 |

## Results

| Metric | Value |
|---|---|
| Total parameters | 606,142,464 |
| Trainable parameters | 10,092,544 (1.665%) |
| Training time | ~116 seconds |
| Peak GPU memory | ~1.9 GB |
| Training loss (step 30 to 55) | 0.94 to 1.09 |

**Note on these results.** Sixty steps with an effective batch size of 8 means the model saw only about 480 of the 33,955 available examples. The loss is still noisy and has not clearly converged. These numbers show that the pipeline works, not that the model is accurate.

As an informal check, I gave the model an unrelated instruction (asking it to act as a wedding planner). It declined instead of following it. This was a small test, not a proper evaluation.

For a stronger result, replace `max_steps = 60` with `num_train_epochs = 1` in the training config.

## Getting Started

### Google Colab

1. Open `qlora.ipynb` in Google Colab.
2. Set the runtime to GPU (T4).
3. Run all cells in order.

### Local setup

An NVIDIA GPU with CUDA support is required.

```bash
git clone https://github.com/ShreyaSingh-2005/qwen3-medical-qlora.git
cd qwen3-medical-qlora

pip install -r requirements.txt

python train.py
python inference.py --prompt "What is the function of insulin?"
```

After training, the LoRA adapters are saved to `./qwen3_0.6b_medical_qlora`.

## Project Structure

```
qwen3-medical-qlora/
├── qlora.ipynb        # Colab notebook with outputs
├── train.py           # Data preparation and QLoRA training
├── inference.py       # Load adapters and generate answers
├── requirements.txt   # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## What I Learned

- How 4-bit quantization and LoRA work together to make fine-tuning feasible on modest hardware.
- How to convert a raw dataset into the chat format a model expects.
- How batch size, gradient accumulation and learning rate affect training.
- How to debug dependency and configuration errors in a fast-moving ML ecosystem.
- Why honest evaluation matters: a short run proves the pipeline works, not that the model is good.

## Limitations

- Trained for only 60 steps, so the model is under-trained.
- No validation split or formal benchmark was used.
- A 0.6B model has limited knowledge and can produce incorrect answers.
- The dataset consists of short flashcards, so answers tend to be brief.
- Not suitable for any real medical use.

## Future Work

- Train for full epochs with a held-out validation set.
- Compare against the base model using ROUGE, BERTScore and manual review.
- Merge the adapters and publish the model on the Hugging Face Hub.
- Export to GGUF for local use.
- Add a retrieval (RAG) layer to ground answers in trusted medical sources.
- Build a small Gradio demo.

## Acknowledgements

[Unsloth](https://github.com/unslothai/unsloth), [Hugging Face](https://huggingface.co) (Transformers, TRL, PEFT, Datasets), [Qwen](https://huggingface.co/Qwen), and the authors of the [medical-qa-datasets](https://huggingface.co/datasets/lavita/medical-qa-datasets) collection.

## License

Released under the [MIT License](LICENSE).

## Author

Shreya Singh ([@ShreyaSingh-2005](https://github.com/ShreyaSingh-2005))
