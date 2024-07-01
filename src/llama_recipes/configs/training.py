# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="PATH/to/Model"  # 预训练模型存放的位置
    tokenizer_name: str=None  # 使用自定义的tokenizer或者使用预训练模型的tokenizer 
    enable_fsdp: bool=False  # 是否使用fsdp优化
    low_cpu_fsdp: bool=False  # 使用fsdp优化时，先读取模型到GPU0，然后再分发到其他GPU，从而优化内存空间。只支持预览版的pytorch
    run_validation: bool=True  # 训练时是否运行测试集
    batch_size_training: int=4  # 训练集的batch size，总的batch size需要乘以分布式的进程数量
    batching_strategy: str="packing" #alternative: padding。padding处理方式，将每个批次的数据填充到窗口长度，packing处理方式，将每个批次的数据连接起来，分割成若干个窗口。
    context_length: int=4096  # 上下文窗口长度
    gradient_accumulation_steps: int=1  # 梯度累积步数，前向多少步进行一次反向传播
    gradient_clipping: bool = False  # 启用梯度裁剪
    gradient_clipping_threshold: float = 1.0  # 梯度裁剪阈值，使用基于范数大小的裁剪
    num_epochs: int=3  # 训练多少个epochs
    max_train_step: int=0  # 最多总共训练多少个step，0表示不限制
    max_eval_step: int=0  # 最多总共验证多少个step，0表示不限制
    num_workers_dataloader: int=1  # dataloader使用多少个子进程来读取数据
    lr: float=1e-4  # 初始学习率
    weight_decay: float=0.0  # 权重衰减比率
    gamma: float= 0.85  # 学习率每个epoch衰减比率
    seed: int=42
    use_fp16: bool=False  # 是否使用fp16，在前向传播时会autocast为fp16，在反向传播时会进行梯度缩放 
    mixed_precision: bool=True  # 是否使用混合精度训练，用于FSDP模型设置，自动选择fp16和bf16。官方建议选择开启这个，不使用use_fp16
    val_batch_size: int=1  # 验证集的batch size，总的batch size需要乘以分布式的进程数量
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=False  # 是否进行微调
    from_peft_checkpoint: str="" # 加载保存的微调参数 if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    output_dir: str = "PATH/to/save/PEFT/model"  # 保存微调模型参数的位置，以及训练的配置
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
    save_metrics: bool = False # 保存训练指标 saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
