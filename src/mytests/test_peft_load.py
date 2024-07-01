import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import torch.distributed as dist
from peft import PeftModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.policies import get_llama_wrapper
from peft import LoraConfig, get_peft_model
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

model = LlamaForCausalLM.from_pretrained(
    "/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf"
).to(torch.bfloat16)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    inference_mode=False
)
model = get_peft_model(model, config)
model = PeftModel.from_pretrained(
    model, 
    "/home/xuhang/LLM/models/llama-3-8B-peft",
    is_trainable=False
)
print(model)