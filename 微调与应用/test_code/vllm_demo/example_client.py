import requests
import json


# API 端点
def test(url):
    # 示例文本，包含一些错误
    data = {
        'text': '我今天去图书关看了一本很有意思的书。对于初学者来说，这本涉及的知识要点很全面。我昨天买了一件心衣服。'
    }

    # 发送 API 请求
    response = requests.post(url, json=data)

    # 漂亮地打印响应
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    test(url="http://localhost:8000/generate")
