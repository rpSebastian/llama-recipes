import torch
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    "/data1/xuhang/hf_hub/Meta-Llama-3-8B-hf",
    use_cache=False,
).to(torch.bfloat16)

config = model.config
V = config.vocab_size
H = config.hidden_size
L = config.num_hidden_layers
UH = config.intermediate_size
T = 2048
B = 1
print("V = {}, H = {}, L = {}, UH = {}, T = {}, B = {}".format(V, H, L, UH, T, B))
# 输入嵌入层
input_embedding_params = lambda V, H: V * H
# 多头注意力层
mha_params = lambda H: 2 * H * H + 2 * H * H // 4  # 使用GQA情况下，每4个头共用一个KV
# 前馈网络层
ffn_params = lambda H, UH: 3 * H * UH
# RMS正则化，每一层有两个，输出有一个
rms_params = lambda H: H
# 解码器层
decoder_layer_params = lambda H, UH: mha_params(H) + ffn_params(H, UH) + rms_params(H) * 2
# 输出线性层
output_linear_params = lambda H, V: H * V
# 总参数量
params = input_embedding_params(V, H) + L * decoder_layer_params(H, UH) + rms_params(H) + output_linear_params(H, V)

print("Estimated number of parameters: ", params)
print("Actual number of parameters: ", sum(p.numel() for p in model.parameters()))
print("Estimated GPU memory usage: {:.2f} MB in bfloat16".format(params * 2 / 1024 / 1024))

# 假设使用激活重计算技术
decoder_layer_input_activation_params = lambda B, T, H: B * T * H
output_rms_input_activation_params = lambda B, T, H: B * T * H
output_linear_input_activation_params = lambda B, T, H: B * T * H
output_softmax_activation_params = lambda B, T, V: B * T * V
activation_params = (
    L * decoder_layer_input_activation_params(B, T, H) + 
    output_rms_input_activation_params(B, T, H) +
    output_linear_input_activation_params(B, T, H) +
    output_softmax_activation_params(B, T, V)
)
print("Estimated activation GPU memory usage: {:.2f} MB in bfloat16".format(activation_params * 2 / 1024 / 1024))