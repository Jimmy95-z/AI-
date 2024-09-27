from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
# 设置模型和适配器的路径
path_to_adapter = "./output_qwen"
new_model_directory = "./lora_merge_new"

# 从预训练的适配器加载模型
model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # 指向输出⽬录的路径
    device_map="auto", # ⾃动设备映射
    trust_remote_code=True # 信任远程代码
).eval()

 # 合并模型参数并卸载未使⽤的部分
merged_model = model.merge_and_unload()

# 保存合并后的模型，设置最⼤分⽚⼤⼩和使⽤安全序列化
merged_model.save_pretrained(
    new_model_directory,
    max_shard_size="2048MB", # 设置最⼤分⽚⼤⼩
    safe_serialization=True # 启⽤安全序列化
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # 指向输出⽬录的路径
    trust_remote_code=True # 信任远程代码
)

tokenizer.save_pretrained(new_model_directory)
