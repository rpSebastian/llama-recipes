import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs.datasets import samsum_dataset
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.utils.config_utils import get_dataloader_kwargs
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_8bit=True,
)
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict
from llama_recipes.configs import lora_config as LORA_CONFIG

import torch.optim as optim
from llama_recipes.utils.train_utils import train
from torch.optim.lr_scheduler import StepLR


train_config = TRAIN_CONFIG()
train_config.model_name = "/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf"
train_config.num_epochs = 1
train_config.run_validation = False
train_config.gradient_accumulation_steps = 4
train_config.batch_size_training = 1
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.use_fp16 = True
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048 # T4 16GB or A10 24GB
train_config.batching_strategy = "packing"
train_config.output_dir = "/home/xhuang/models/meta-llama-samsum"
model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            device_map="auto",
            quantization_config=config,
            use_cache=False,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            torch_dtype=torch.float16,
        )
tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset = get_preprocessed_dataset(tokenizer, samsum_dataset, 'train')
train_dl_kwargs = get_dataloader_kwargs(train_config, train_dataset, tokenizer, "train")
if train_config.batching_strategy == "packing":
        train_dataset = ConcatDataset(train_dataset, chunk_size=train_config.context_length)

# Create DataLoaders for the training and validation dataset
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    num_workers=train_config.num_workers_dataloader,
    pin_memory=True,
    **train_dl_kwargs,
)
lora_config = LORA_CONFIG()
lora_config.r = 8
lora_config.lora_alpha = 32
lora_dropout: float=0.01
peft_config = LoraConfig(**asdict(lora_config))
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.train()
optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
# Start the training process
results = train(
    model,
    train_dataloader,
    None,
    tokenizer,
    optimizer,
    scheduler,
    train_config.gradient_accumulation_steps,
    train_config,
    None,
    None,
    None,
    wandb_run=None,
)