from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from lora_qwen3_config import *

# 1. 加载原始模型
base_model = AutoModelForCausalLM.from_pretrained(qwen3_model_path)
tokenizer = AutoTokenizer.from_pretrained(qwen3_model_path)

# 初始化与训练时相同的 LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

# 2. 加载 LoRA Adapter
lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

# 3. 合并 LoRA 权重
merged_model = lora_model.merge_and_unload()

# 4. 保存合并后的模型
merged_model.save_pretrained(merged_model_path, max_shard_size="3GB")
tokenizer.save_pretrained(merged_model_path)