. "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate main

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR/../recipes/inference/local_inference

# python $ROOT_DIR/inference.py --model_name /data1/xuhang/hf_hub/Meta-Llama-3-8B-hf --prompt_file $ROOT_DIR/samsum_prompt.txt 

# python $ROOT_DIR/inference.py --model_name /data1/xuhang/hf_hub/Meta-Llama-3-8B-hf --prompt_file $ROOT_DIR/samsum_prompt.txt --peft_model /home/xuhang/LLM/models/llama-3-8B-peft

python $ROOT_DIR/inference.py --model_name /data1/xuhang/hf_hub/Meta-Llama-3-8B-hf --peft_model /home/xuhang/LLM/models/llama-3-8B-peft
