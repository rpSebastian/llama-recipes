from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import transformers
import torch
from pathlib import Path

hf_hub_path = Path("/home/xuhang/hf_hub")
model_path = str(hf_hub_path / "Meta-Llama-3-8B-Instruct-hf")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
sequences = pipeline(
    'I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1000,
)
for seq in sequences:
    print(f"{seq['generated_text']}")