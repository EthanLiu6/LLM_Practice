from peft import LoraConfig, get_peft_model


from lora_qwen3_config import *

# 加载原始模型
base_model = AutoModelForCausalLM.from_pretrained(qwen3_model_path)

# 初始化与训练时相同的 LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

# 绑定 LoRA 到原始模型
qwen3_lora_model = get_peft_model(base_model, peft_config)

# 加载训练好的 LoRA 权重
qwen3_lora_model.load_state_dict(torch.load(lora_adapter_path))

# 合并并保存
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_model_path, max_shard_size="3GB")
tokenizer.save_pretrained(merged_model_path)  # 同时保存分词器