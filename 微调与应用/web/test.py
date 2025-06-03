from flask import Flask, request, jsonify
import json
import difflib
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)


class TextCorrectionAPI:
    def __init__(self, model_path: str = None, use_vllm: bool = False):
        self.use_vllm = use_vllm
        self.model_path = model_path or "../merged_model/qwen3-8B-TextCorrection"

        if use_vllm:
            self.init_vllm()
        else:
            self.init_transformers()

    def init_transformers(self):
        """
        使用transformers直接加载模型
        """
        print(f"正在加载模型: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        print("模型加载完成")

    def init_vllm(self):
        """
        使用vLLM加载模型（需要安装vllm: pip install vllm）
        """
        try:
            from vllm import LLM, SamplingParams
            print(f"正在使用vLLM加载模型: {self.model_path}")
            self.llm = LLM(model=self.model_path)
            self.sampling_params = SamplingParams(
                max_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
            # 还需要tokenizer用于构建prompt
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("vLLM模型加载完成")
        except ImportError:
            print("vLLM未安装，请安装: pip install vllm")
            print("切换到transformers模式")
            self.use_vllm = False
            self.init_transformers()

    def correct_text(self, text: str) -> str:
        """
        调用纠错模型进行文本纠错
        """
        prompt = "请纠正所给句子的错误，错误包含[错别字、语病、同音字等]，如果是正确句子则不进行修改，直接返回修改后的句子即可。\n"

        if self.use_vllm:
            return self.correct_with_vllm(prompt + text)
        else:
            return self.correct_with_transformers(prompt + text)

    def correct_with_transformers(self, full_prompt: str) -> str:
        """
        使用transformers进行推理
        """
        messages = [
            {"role": "user", "content": full_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # 生成文本
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,  # 适当减少，纠错通常不需要太长
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码并输出完整生成内容
        corrected_text = self.tokenizer.decode(
            generated_ids[0][len(model_inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()

        return corrected_text

    def correct_with_vllm(self, full_prompt: str) -> str:
        """
        使用vLLM进行推理
        """
        messages = [
            {"role": "user", "content": full_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        outputs = self.llm.generate([text], self.sampling_params)
        corrected_text = outputs[0].outputs[0].text.strip()

        return corrected_text

    def get_detailed_differences(self, original: str, corrected: str) -> List[Dict[str, Any]]:
        """
        使用更细粒度的diff分析，支持字符级别的差异检测
        """
        mistakes = []

        # 按字符进行比较
        matcher = difflib.SequenceMatcher(None, list(original), list(corrected))

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                original_text = original[i1:i2]
                corrected_text = corrected[j1:j2]

                # 如果是空的，跳过
                if not original_text and not corrected_text:
                    continue

                mistake = {
                    "l": i1,
                    "r": i2,
                    "infos": []
                }

                if tag == 'replace':
                    # 替换错误
                    mistake["infos"].append({
                        "recommend": corrected_text,
                        "type": 1,
                        "category": "1-1"  # 错别字
                    })
                elif tag == 'delete':
                    # 多字错误
                    mistake["infos"].append({
                        "recommend": "",
                        "type": 2,
                        "category": "2-1"  # 多字
                    })
                elif tag == 'insert':
                    # 少字错误
                    mistake["infos"].append({
                        "recommend": corrected_text,
                        "type": 3,
                        "category": "3-1"  # 少字
                    })

                mistakes.append(mistake)

    def classify_error(self, original_text: str, corrected_text: str) -> Dict[str, Any]:
        """
        根据错误类型进行分类
        你可以根据实际需求调整分类逻辑
        """
        # 默认错误分类规则，可以根据你的纠错模型调整
        if len(original_text) == len(corrected_text):
            # 字符替换
            return {"type": 1, "category": "1-1"}  # 错别字
        elif len(original_text) > len(corrected_text):
            # 删除多余字符
            return {"type": 2, "category": "2-1"}  # 多字
        else:
            # 插入缺失字符
            return {"type": 3, "category": "3-1"}  # 少字

    def find_differences(self, original: str, corrected: str) -> List[Dict[str, Any]]:
        """
        使用difflib找出原文和修改后文本的差异，生成错误信息
        简化版本，只包含位置和推荐修改
        """
        # 使用细粒度差异检测
        mistakes = self.get_detailed_differences(original, corrected)

        # 合并相邻错误（可选）
        mistakes = self.merge_adjacent_mistakes(mistakes)

        return mistakes

    def merge_adjacent_mistakes(self, mistakes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并相邻的错误，避免单个词被拆分成多个错误
        """
        if not mistakes:
            return mistakes

        merged = []
        current = mistakes[0].copy()

        for mistake in mistakes[1:]:
            # 如果当前错误与前一个错误相邻或重叠
            if mistake["l"] <= current["r"]:
                # 合并错误
                current["r"] = max(current["r"], mistake["r"])
                current["infos"].extend(mistake["infos"])
            else:
                merged.append(current)
                current = mistake.copy()

        merged.append(current)
        return merged
        """
        根据错误类型进行分类
        你可以根据实际需求调整分类逻辑
        """
        # 默认错误分类规则，可以根据你的纠错模型调整
        if len(original_text) == len(corrected_text):
            # 字符替换
            return {"type": 1, "category": "1-1"}  # 错别字
        elif len(original_text) > len(corrected_text):
            # 删除多余字符
            return {"type": 2, "category": "2-1"}  # 多字
        else:
            # 插入缺失字符
            return {"type": 3, "category": "3-1"}  # 少字

    def find_differences(self, original: str, corrected: str) -> List[Dict[str, Any]]:
        """
        使用difflib找出原文和修改后文本的差异，生成错误信息
        """
        mistakes = []

        # 使用difflib进行序列比较
        differ = difflib.SequenceMatcher(None, original, corrected)

        # 获取所有操作
        opcodes = differ.get_opcodes()

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'replace':
                # 替换操作：原文[i1:i2] 被替换为 修正文[j1:j2]
                original_text = original[i1:i2]
                corrected_text = corrected[j1:j2]

                error_info = self.classify_error(original_text, corrected_text)

                mistake = {
                    "l": i1,
                    "r": i2,
                    "infos": [{
                        "recommend": corrected_text,
                        **error_info
                    }]
                }
                mistakes.append(mistake)

            elif tag == 'delete':
                # 删除操作：原文[i1:i2] 被删除
                mistake = {
                    "l": i1,
                    "r": i2,
                    "infos": [{
                        "recommend": "",  # 删除，所以推荐为空
                        "type": 2,
                        "category": "2-1"  # 多字错误
                    }]
                }
                mistakes.append(mistake)

            elif tag == 'insert':
                # 插入操作：在原文位置i1插入修正文[j1:j2]
                mistake = {
                    "l": i1,
                    "r": i1,  # 插入位置，起始和结束相同
                    "infos": [{
                        "recommend": corrected[j1:j2],
                        "type": 3,
                        "category": "3-1"  # 少字错误
                    }]
                }
                mistakes.append(mistake)

        return mistakes

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        处理文本并返回完整的纠错结果
        """
        # 1. 调用纠错模型
        corrected_text = self.correct_text(text)

        # 2. 找出差异
        mistakes = self.find_differences(text, corrected_text)

        # 3. 构建返回结果
        result = {
            "result": {
                "sentence": text,
                "mistake_num": len(mistakes),
                "mistakes": mistakes
            },
            "sum": len(mistakes),
            "message": "校对完成"
        }

        return result


# 创建API实例
# 使用方式1: 直接用transformers（默认）
correction_api = TextCorrectionAPI(use_vllm=False)


# 使用方式2: 使用vLLM（更高性能，需要安装vllm）
# correction_api = TextCorrectionAPI(use_vllm=True)

# 使用方式3: 指定模型路径
# correction_api = TextCorrectionAPI(
#     model_path="/path/to/your/model",
#     use_vllm=False
# )

@app.route('/correct', methods=['POST'])
def correct_text():
    """
    文本纠错API端点
    接收POST请求，参数为JSON格式：{"text": "要纠错的文本"}
    """
    try:
        # 获取请求数据
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                "error": "请求格式错误，需要提供text参数",
                "code": 400
            }), 400

        text = data['text']

        if not text.strip():
            return jsonify({
                "error": "文本内容不能为空",
                "code": 400
            }), 400

        # 处理文本纠错
        result = correction_api.process_text(text)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": f"服务器内部错误: {str(e)}",
            "code": 500
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查端点
    """
    return jsonify({
        "status": "healthy",
        "message": "文本纠错API服务正常运行"
    })


@app.route('/', methods=['GET'])
def index():
    """
    API文档
    """
    return jsonify({
        "service": "文本纠错API",
        "version": "1.0.0",
        "endpoints": {
            "POST /correct": "文本纠错接口",
            "GET /health": "健康检查",
            "GET /": "API文档"
        },
        "usage": {
            "url": "/correct",
            "method": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "text": "要纠错的文本内容"
            }
        }
    })


@app.route('/test', methods=['GET'])
def test_diff():
    """
    测试diff功能
    """
    # 测试文本
    original = "在当今社会，随着科技的迅速发展，人们的生活方式也发生了很大的变化。智能手机以经成为我们日常生活中不可或缺的一部分，无论是在公交上还是再吃饭的时候，总能看到有人在低头玩手机。这种现象不仅影响了人们的交住，也可能导致视力下降和颈椎问题。此外，网络信息泛滥，虚假新闻频发，使得公众很难分辨真価値的信息。因此，加强网洛安全意识和媒体素养教育变得尤为重要。"

    corrected = "在当今社会，随着科技的迅速发展，人们的生活方式也发生了很大的变化。智能手机已经成为我们日常生活中不可或缺的一部分，无论是在公交上还是在吃饭的时候，总能看到有人在低头玩手机。这种现象不仅影响了人们的交往，也可能导致视力下降和颈椎问题。此外，网络信息泛滥，虚假新闻频发，使得公众很难分辨真正值的信息。因此，加强网络安全意识和媒体素养教育变得尤为重要。"

    result = correction_api.process_text(original)

    return jsonify({
        "original": original,
        "corrected": corrected,
        "diff_result": result
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """
    获取模型信息
    """
    return jsonify({
        "model_path": correction_api.model_path,
        "using_vllm": correction_api.use_vllm,
        "status": "loaded"
    })


if __name__ == '__main__':
    # 开发环境运行
    app.run(host='0.0.0.0', port=5000, debug=True)
