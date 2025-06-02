from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from peft import get_peft_model

from transformers import modeling_utils
if modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']


def lora_ft(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    qwen3_model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype='float16',
        device_map='mps'
    )


    print(qwen3_model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    lora_model = get_peft_model(model=qwen3_model, peft_config=lora_config)
    print('*'*30)
    print(lora_model)


if __name__ == '__main__':
    qwen3 = '/Users/ethanliu/Documents/models/Qwen/Qwen3-0.6B'
    lora_ft(qwen3)

