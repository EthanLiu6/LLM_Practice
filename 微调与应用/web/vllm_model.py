#!/usr/bin/env python3
"""
使用vLLM部署纠错模型的独立服务器
支持OpenAI兼容的API接口
"""

import argparse
import requests
from flask import Flask, request, jsonify
import difflib
from typing import List, Dict, Any

# vLLM服务器配置
VLLM_SERVER_URL = "http://localhost:8000"

app = Flask(__name__)


class VLLMTextCorrectionAPI:
    def __init__(self, vllm_url: str = VLLM_SERVER_URL):
        self.vllm_url = vllm_url
        self.prompt_template = "请纠正所给句子的错误，错误包含[错别字、语病、同音字等]，如果是正确句子则不进行修改，直接返回修改后的句子即可。\n"

    def correct_text(self, text: str) -> str:
        """
        通过vLLM API调用纠错模型
        """
        try:
            response = requests.post(
                f"{self.vllm_url}/v1/completions",
                json={
                    "model": "qwen3-8B-TextCorrection",  # 根据模型名称调整
                    "prompt": self.prompt_template + text,
                    "max_tokens": 10240,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["<|endoftext|>", "<|im_end|>"]
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                corrected_text = result["choices"][0]["text"].strip()
                return corrected_text
            else:
                raise Exception(f"vLLM API错误: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"连接vLLM服务器失败: {str(e)}")

    def find_differences(self, original: str, corrected: str) -> List[Dict[str, Any]]:
        """
        使用difflib找出差异
        """
        mistakes = []
        matcher = difflib.SequenceMatcher(None, original, corrected)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                mistake = {
                    "l": i1,
                    "r": i2,
                    "infos": [{"recommend": corrected[j1:j2]}]
                }
                mistakes.append(mistake)
            elif tag == 'delete':
                mistake = {
                    "l": i1,
                    "r": i2,
                    "infos": [{"recommend": ""}]
                }
                mistakes.append(mistake)
            elif tag == 'insert':
                mistake = {
                    "l": i1,
                    "r": i1,
                    "infos": [{"recommend": corrected[j1:j2]}]
                }
                mistakes.append(mistake)

        return mistakes

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        处理文本并返回完整结果
        """
        corrected_text = self.correct_text(text)
        mistakes = self.find_differences(text, corrected_text)

        return {
            "result": {
                "sentence": text,
                "mistake_num": len(mistakes),
                "mistakes": mistakes
            },
            "sum": len(mistakes),
            "message": "校对完成"
        }


# 创建API实例
correction_api = VLLMTextCorrectionAPI()


@app.route('/correct', methods=['POST'])
def correct_text():
    """文本纠错API"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "请求格式错误，需要提供text参数", "code": 400}), 400

        text = data['text']
        if not text.strip():
            return jsonify({"error": "文本内容不能为空", "code": 400}), 400

        result = correction_api.process_text(text)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"服务器错误: {str(e)}", "code": 500}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        # 检查vLLM服务器是否可用
        response = requests.get(f"{correction_api.vllm_url}/health", timeout=5)
        vllm_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        vllm_status = "unavailable"

    return jsonify({
        "status": "healthy",
        "vllm_server": vllm_status,
        "message": "文本纠错API服务正常运行"
    })


@app.route('/', methods=['GET'])
def index():
    """API文档"""
    return jsonify({
        "service": "vLLM文本纠错API",
        "version": "1.0.0",
        "vllm_server": correction_api.vllm_url,
        "endpoints": {
            "POST /correct": "文本纠错接口",
            "GET /health": "健康检查",
            "GET /": "API文档"
        }
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vLLM文本纠错API服务器')
    parser.add_argument('--vllm-url', default=VLLM_SERVER_URL,
                        help='vLLM服务器地址 (默认: http://localhost:8000)')
    parser.add_argument('--port', type=int, default=5000,
                        help='API服务器端口 (默认: 5000)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='API服务器地址 (默认: 0.0.0.0)')

    args = parser.parse_args()

    # 更新vLLM服务器地址
    correction_api.vllm_url = args.vllm_url

    print(f"启动vLLM文本纠错API服务器...")
    print(f"vLLM服务器: {args.vllm_url}")
    print(f"API服务器: http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)
