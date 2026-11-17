"""Load the saved LoRA adapters and generate an answer."""
import argparse
from unsloth import FastLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", default="qwen3_0.6b_medical_qlora")
parser.add_argument("--prompt", default="Answer this question truthfully\n\nWhat is the function of insulin?")
parser.add_argument("--max_new_tokens", type=int, default=200)
args = parser.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.adapter,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # 2x faster inference

messages = [{"role": "user", "content": args.prompt}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=args.max_new_tokens,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
