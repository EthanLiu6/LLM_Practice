from datasets import Dataset
from transformers import AutoTokenizer
import json


def load_json_dataset(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果 data 是 dict 并且有 'data' 字段，则取出来
    if isinstance(data, dict) and 'data' in data:
        data = data['data']

    return Dataset.from_list(data)


def process_func_single(example, tokenizer):
    """
    Args:
        example: 包含'instruction'、'input'、'output'
        tokenizer: 分词器对象（需支持apply_chat_template）
    Returns:
        包含input_ids, attention_mask, labels的字典
    """
    try:
        # 构建对话结构
        messages = [
            {"role": "system", "content": example['instruction']},
            {"role": "user", "content": example["input"]}
        ]

        # 生成指令文本
        instruction_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        instruction = tokenizer(instruction_text, add_special_tokens=False)

        # 生成响应内容
        response_text = f"{example['output']}"

        response = tokenizer(response_text, add_special_tokens=False)

        # 合并结果
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    except Exception as e:
        print(f"处理样本时出错: {e}\n样本内容: {example}")
        raise


def process_func_batch(examples, tokenizer, max_length=2048):
    """
    批次处理函数（不填充到相同长度，但会截断超长部分）
    Args:
        examples: 包含question, cot, answer, type的批次数据
        tokenizer: 分词器对象（需支持apply_chat_template）
        max_length: 最大序列长度
    Returns:
        包含input_ids, attention_mask, labels的字典（各样本长度可能不同）
    """
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for instruction, ipt, output in zip(examples["instruction"], examples["input"], examples["output"]):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": ipt}
        ]

        try:
            # 生成指令部分
            instruction_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            instruction = tokenizer(instruction_text, add_special_tokens=False)

            # 生成响应部分
            response_text = output
            response = tokenizer(response_text, add_special_tokens=False)

            # 合并指令和响应
            input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
            attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
            labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

            # 截断超长部分
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        except Exception as e:
            print(f"处理样本时出错 - 问题: {ipt[:50]}... 错误: {str(e)}")
            # 添加空样本以防中断流程
            batch_input_ids.append([])
            batch_attention_mask.append([])
            batch_labels.append([])

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }


def dataset_main(json_file, model_path, use_batch=True):
    dataset = load_json_dataset(json_file)
    # print(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if use_batch:
        train_dataset = dataset.map(
            lambda x: process_func_batch(x, tokenizer),
            batched=True,
            batch_size=4
        )

    else:
        train_dataset = dataset.map(
            lambda x: process_func_single(x, tokenizer),
            batched=False
        )

    return train_dataset


if __name__ == "__main__":
    json_file = "./sighan_2015/train.json"
    model_path = '/Users/ethanliu/Documents/models/Qwen/Qwen3-0.6B'
    train_dataset = dataset_main(json_file, model_path)
    print(train_dataset)
