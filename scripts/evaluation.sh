. "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate main

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR/../recipes/evaluation

# python $ROOT_DIR/eval.py --model hf --model_args pretrained=/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf --tasks hellaswag --device cuda:1   --batch_size 8

python $ROOT_DIR/eval.py --model hf --model_args pretrained=/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf,peft=/home/xuhang/LLM/models/llama-3-8B-peft --tasks hellaswag --num_fewshot 10  --device cuda:2 --batch_size 8