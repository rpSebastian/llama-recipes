import torch
from transformers import LlamaForCausalLM, AutoTokenizer

model = LlamaForCausalLM.from_pretrained(
    "/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf"
).to(torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(
    "/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf"
)

model.save_pretrained("/home/xuhang/LLM/models/llama-3-8B-peft")
tokenizer.save_pretrained(
    "/home/xuhang/LLM/models/llama-3-8B-peft"
)

print(model)
print(tokenizer)