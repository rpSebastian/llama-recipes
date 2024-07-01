from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from transformers import AutoTokenizer
from llama_recipes.data.concatenator import ConcatDataset

class DatasetConfig:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"

tokenizer = AutoTokenizer.from_pretrained("/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = get_preprocessed_dataset(tokenizer, DatasetConfig(), split="train")

total = 0
for sample in dataset:
    total += len(sample["input_ids"])

print(total)

dataset = ConcatDataset(dataset, chunk_size=2048)
print(len(dataset))
total = 0
for sample in dataset:
    total += len(sample["input_ids"])
print(total)