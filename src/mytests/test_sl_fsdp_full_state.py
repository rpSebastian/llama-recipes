import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig
import os
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from llama_recipes.policies import get_llama_wrapper
from llama_recipes.model_checkpointing import save_model_checkpoint
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
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

with FSDP.state_dict_type(
    model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
):
    cpu_state = model.state_dict()
    print(f"saving process: rank {rank}  done w model state_dict\n")

print(cpu_state)

if rank == 0:
    print(f"--> saving model ...")
    # save model
    torch.save(cpu_state, "/home/xuhang/LLM/models/llama-3-8B-peft/test.pt")