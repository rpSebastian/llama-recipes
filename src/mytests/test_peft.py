import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
from llama_recipes.policies import get_llama_wrapper
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = LlamaForCausalLM.from_pretrained(
    "/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf"
).to(torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(
    "/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf"
)

model = FSDP(
    model,
    device_id=local_rank,
    auto_wrap_policy=get_llama_wrapper()
)

model.save_pretrained("/home/xuhang/LLM/models/llama-3-8B-peft")
tokenizer.save_pretrained(
    "/home/xuhang/LLM/models/llama-3-8B-peft"
)

print(model)
print(tokenizer)

model = LlamaForCausalLM.from_pretrained(
    "/home/xuhang/LLM/models/llama-3-8B-peft"
).to(torch.bfloat16)
print(model)