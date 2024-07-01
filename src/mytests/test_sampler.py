import torch.utils
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from transformers import AutoTokenizer
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.data.sampler import LengthBasedBatchSampler
import torch
from torch.utils.data import DataLoader, BatchSampler
from transformers.data import DataCollatorForSeq2Seq
from transformers import default_data_collator

class DatasetConfig:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"

tokenizer = AutoTokenizer.from_pretrained("/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = get_preprocessed_dataset(tokenizer, DatasetConfig(), split="train")

sampler = LengthBasedBatchSampler(dataset, batch_size=3, drop_last=True)
dataloader = DataLoader(
    dataset, 
    batch_sampler=sampler,
    collate_fn=DataCollatorForSeq2Seq(tokenizer))
data = next(iter(dataloader))

dataset = ConcatDataset(dataset, chunk_size=2048)
dataloader = DataLoader(
    dataset, 
    batch_size=3,
    drop_last=True,
    collate_fn=default_data_collator)
data = next(iter(dataloader))
print(data)