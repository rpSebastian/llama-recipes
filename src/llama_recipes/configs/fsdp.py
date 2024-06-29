# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class fsdp_config:
    mixed_precision: bool=True  # 是否使用混合精度训练，使用fp16或bf16
    use_fp16: bool=False  # 混合精度训练下使用fp16，否则在允许条件下使用bf16
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD # HYBRID_SHARD "Full Shard within a node DDP cross Nodes", SHARD_GRAD_OP "Shard only Gradients and Optimizer States", NO_SHARD "Similar to DDP". 使用什么策略进行分片
    hsdp : bool =False # Require HYBRID_SHARD to be set. This flag can extend the HYBRID_SHARD by allowing sharding a model on customized number of GPUs (Sharding_group) and Replicas over Sharding_group.
    sharding_group_size : int=0 # requires hsdp to be set. This specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    replica_group_size: int=0 #requires hsdp to be set. This specifies the replica group size, which is world_size/sharding_group_size.
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool=True  # 是否使用激活重构建功能，在反向传播时重新计算激活值。
    fsdp_cpu_offload: bool=False  # 是否使用cpu卸载，在不使用时将GPU加载的参数卸载到CPU中
    pure_bf16: bool = False  # 模型、优化器都使用bf16存储
    optimizer: str= "AdamW"
    
