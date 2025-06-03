from vllm import LLM, SamplingParams
from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
import json
import re
import asyncio
from typing import List, Dict, Any

model_path = '../../lora_models/lora_qwen3-8B—TextCorrection'
# 初始化 FastAPI 应用
app = FastAPI(title="文本校正模型 API 服务器")

# 初始化 LLM 模型
model = LLM(model=model_path)

# 定义默认采样参数
default_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=4096,
    enable_thinking=False
)

# 定义指令提示语
CORRECTION_PROMPT = "请纠正所给句子的错误，错误包含[错别字、语病、同音字等]，如果是正确句子则不进行修改，直接返回修改后的句子即可。"


def split_text_into_chunks(text):
    """
    根据中文/英文句子结束标点符号将文本分成块。
    返回句子块列表。
    """
    pattern = r'([。？！.!?])'

    # 按模式分割但保留分隔符
    parts = re.split(f'({pattern})', text)

    # 将每个句子与其标点符号组合
    chunks = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and re.match(pattern, parts[i + 1]):
            # 将句子与其标点符号组合
            chunks.append(parts[i] + parts[i + 1])
            i += 2
        else:
            # 处理没有标点符号跟随的情况
            if parts[i].strip():  # 只添加非空部分
                chunks.append(parts[i])
            i += 1

    return chunks


async def process_chunks(chunks: List[str], sampling_params: SamplingParams) -> List[Dict[str, Any]]:
    """异步处理文本块"""
    if not chunks:
        return []

    # 为每个文本块添加指令提示语
    prompted_chunks = [f"{CORRECTION_PROMPT}\n{chunk}" for chunk in chunks]

    # vLLM 的 generate 方法会对批量输入进行优化处理
    outputs = model.generate(prompted_chunks, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        results.append({
            "original_chunk": chunks[i],  # 保存原始文本块（不含指令）
            "correction": output.outputs[0].text
        })

    return results


@app.post("/generate")
async def generate(request: Request):
    # 解析请求体
    body = await request.json()

    # 从输入格式 {'text': xxxx} 中提取文本
    if 'text' not in body:
        return {"error": "输入必须包含 'text' 字段"}

    input_text = body['text']

    # 根据标点符号将文本分成块
    text_chunks = split_text_into_chunks(input_text)

    # 可选参数，可以覆盖默认值
    temperature = body.get('temperature', default_params.temperature)
    top_p = body.get('top_p', default_params.top_p)
    max_tokens = body.get('max_tokens', default_params.max_tokens)
    enable_thinking = body.get('enable_thinking', default_params.enable_thinking)

    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking
    )

    # 异步处理每个块
    results = await process_chunks(text_chunks, sampling_params)

    # 返回块结果和合并结果
    combined_text = "".join([result["correction"] for result in results])

    return {
        "chunks": results,
        "generated_text": combined_text
    }


@app.post("/generate_whole")
async def generate_whole(request: Request):
    """处理整个文本作为单个提示，不分块"""
    # 解析请求体
    body = await request.json()

    # 从输入格式 {'text': xxxx} 中提取文本
    if 'text' not in body:
        return {"error": "输入必须包含 'text' 字段"}

    input_text = body['text']

    # 添加指令提示语
    prompted_input = f"{CORRECTION_PROMPT}\n{input_text}"

    # 可选参数，可以覆盖默认值
    temperature = body.get('temperature', default_params.temperature)
    top_p = body.get('top_p', default_params.top_p)
    max_tokens = body.get('max_tokens', default_params.max_tokens)
    enable_thinking = body.get('enable_thinking', default_params.enable_thinking)

    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking
    )

    # 使用 vLLM 生成响应
    outputs = model.generate([prompted_input], sampling_params)

    # 格式化并返回响应
    generated_text = outputs[0].outputs[0].text

    return {"generated_text": generated_text}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    # 运行 API 服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)

