from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "../merged_model/qwen3-8B-TextCorrection"


def main():

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )

    print("欢迎使用Ethan的纠错模型。")

    # 准备模型输入
    prompt = "请纠正所给句子的错误，错误包含[错别字、语病、同音字等]，如果是正确句子则不进行修改，直接返回修改后的句子即可。\n"
    
    while 1:
        query = input("请输入要纠错的句子(q退出)：\n").strip()
        if query == "q":
            print("欢迎下次使用！bye~")
            break

        messages = [
            {"role": "user", "content": prompt + query}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # 生成文本
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768,
            do_sample=True,  # 启用随机采样
            top_p=0.9,      # nucleus sampling参数
            temperature=0.7 # 温度参数
        )

        # 解码并输出完整生成内容
        full_output = tokenizer.decode(
            generated_ids[0][len(model_inputs.input_ids[0]):],  # 只取新生成的部分
            skip_special_tokens=True
        ).strip()

        print(full_output)
        print("*" * 30)



if __name__ == '__main__':
    main()