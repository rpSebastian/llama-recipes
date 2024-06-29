import os
import torch.distributed as dist
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from llama_recipes.utils import fsdp_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from llama_recipes.policies import apply_fsdp_checkpointing
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.data.concatenator import ConcatDataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.utils.data import DistributedSampler
import torch.optim as optim
import time

def main():
    setup_dist()
    setup_model()
    setup_tokenizer()
    setup_dataset()
    setup_optimizer()
    setup_scheduler()
    train()

def train():
    for epoch in range(1):
        model.train()
        for step, batch in enumerate(dataloader):
            for key in batch:
                batch[key] = batch[key].to(local_rank)
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            model.clip_grad_norm_(1)
            optimizer.step()
            print(epoch, step, loss.detach().float())

def setup_dist():
    dist.init_process_group("nccl")
    global local_rank, rank, world_size
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

def setup_model():
    global model
    model = LlamaForCausalLM.from_pretrained(
        "/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf",
        use_cache=False,
    ).to(torch.bfloat16)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)
    my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
    model = FSDP(
        model, 
        auto_wrap_policy=my_auto_wrapping_policy, 
        device_id=local_rank,
        param_init_fn=(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
    )
    apply_fsdp_checkpointing(model)

def setup_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf")
    tokenizer.pad_token_id = tokenizer.eos_token_id

class DatasetConfig:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"

def setup_dataset():
    global dataset
    dataset = get_preprocessed_dataset(tokenizer, DatasetConfig(), split="train")
    dataset = ConcatDataset(dataset, chunk_size=2048)
    global dataloader
    sampler = DistributedSampler(
        dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True,
        drop_last=True
    )
    dataloader = DataLoader(
        dataset,
        num_workers=1,
        pin_memory=True,
        batch_size=1,
        collate_fn=default_data_collator,
        drop_last=True,
        sampler=sampler
    )

def setup_optimizer():
    global optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0
    )

def setup_scheduler():
    global scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.85
    )
if __name__ == "__main__":
    main()