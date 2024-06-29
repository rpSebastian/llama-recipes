# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="PATH/to/Model"  # 预训练模型存放的位置
    tokenizer_name: str=None  # 使用自定义的tokenizer或者使用预训练模型的tokenizer 
    enable_fsdp: bool=False  # 是否使用fsdp优化
    low_cpu_fsdp: bool=False  # 使用fsdp优化时，先读取模型到GPU0，然后再分发到其他GPU，从而优化内存空间。只支持预览版的pytorch
    run_validation: bool=True
    batch_size_training: int=4  # 训练集的batch size，总的batch size需要乘以分布式的进程数量
    batching_strategy: str="packing" #alternative: padding。padding处理方式，将每个批次的数据填充到窗口长度，packing处理方式，将每个批次的数据连接起来，分割成若干个窗口。
    context_length: int=4096  # 上下文窗口长度
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=3
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=1  # dataloader使用多少个子进程来读取数据
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1  # 验证集的batch size，总的batch size需要乘以分布式的进程数量
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=False  # 是否进行微调
    from_peft_checkpoint: str="" # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False  # 当不实用peft时，在训练中是否冻结前若干层
    num_freeze_layers: int = 1  # 冻结的层数
    quantization: bool = False  # 使用int8量化模型
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # 使用加速的attention实现。Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False # Enable wandb for experient tracking
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
