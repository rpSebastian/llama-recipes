. "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate main

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR/../recipes/finetuning

torchrun --nnodes 1 --nproc_per_node 8 $ROOT_DIR/finetuning.py --enable_fsdp --model_name /data1/xuhang/hf_hub/Meta-Llama-3-8B-hf --use_peft --peft_method lora --dataset samsum_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --lr 1e-5 --output_dir /home/xuhang/LLM/models/llama-3-8B-peft2 --batch_size_training 1  --use_profiler  --profiler_dir /home/xuhang/LLM/models/llama-3-8B-peft2  --save_metrics  --use_wandb  --num_epochs 10