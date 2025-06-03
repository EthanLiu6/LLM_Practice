from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import Trainer, TrainingArguments

from transformers import modeling_utils

# if modeling_utils.ALL_PARALLEL_STYLES is None:
    # modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']

from data.dataset import dataset_main

# # DS_CONFIG = "./DS_config/ds_zero2_no_offload.json"
# qwen3_model_path = '/mnt/workspace/model_list/Qwen/Qwen3-8B'
# save_model_path = './lora_models/lora_qwen3-8B—TextCorrection'
# json_file = "./data/sighan_2015/train.json"
# output_dir = './out_lora_train'

def main():
    tokenizer = AutoTokenizer.from_pretrained(qwen3_model_path)
    qwen3_model = Qwen3ForCausalLM.from_pretrained(
        qwen3_model_path,
        torch_dtype='float16',
        device_map='auto'
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    lora_model = get_peft_model(model=qwen3_model, peft_config=lora_config)

    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            print(name, "is trainable")

    # 配置训练参数
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        logging_first_step=True,
        num_train_epochs=4,
        save_steps=50,
        learning_rate=2e-4,
        save_on_each_node=True,
        gradient_checkpointing=False, 
        # gradient_checkpointing=True,
        report_to="none",
        # bf16=True,
        fp16=True,
        max_grad_norm=1.0,
        # deepspeed=DS_CONFIG
    )

    train_dataset = dataset_main(
        json_file=json_file,
        model_path=qwen3_model_path
    )

    # 配置Trainer
    trainer = Trainer(
        model=lora_model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    trainer.save_model(save_model_path)



if __name__ == '__main__':
    main()
