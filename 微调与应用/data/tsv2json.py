import json


def tsv2json(in_tsv_file, out_json_file):
    data = []

    with open(tsv_file, 'r', encoding='utf-8') as f:
        for line in f:
            original_text, corrected_text = line.strip().split('\t')

            entry = {
                "instruction": "请纠正所给句子的错误，错误包含[错别字、语病、同音字等]，如果是正确句子则不进行修改，直接返回修改后的句子即可。",
                "input": original_text,
                "output": corrected_text
            }

            data.append(entry)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # tsv_file = './sighan_2015/train.tsv'
    # json_file = './sighan_2015/train.json'
    # tsv_file = './sighan_2015/test.tsv'
    # json_file = './sighan_2015/test.json'

    # tsv_file = './other/law_data.tsv'
    # json_file = './other/law_data.json'

    tsv_file = './other/medical_data.tsv'
    json_file = './other/medical_data.json'


    tsv2json(tsv_file, json_file)
