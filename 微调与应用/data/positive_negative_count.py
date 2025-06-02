def count_positive_negative(tsv_file):
    negative_count, positive_count = 0, 0
    with open(tsv_file, 'r', encoding='utf-8') as f:
        for line in f:
            original_text, corrected_text = line.strip().split('\t')
            if original_text == corrected_text:
                positive_count += 1
            else:
                negative_count += 1

    total_count = negative_count + positive_count
    print(f"有误数据量：{negative_count}，占比：{negative_count / total_count}", )
    print(f"无误数据量：{positive_count}，占比：{positive_count / total_count}")


if __name__ == '__main__':
    # count_positive_negative('./sighan_2015/train.tsv')
    # count_positive_negative('./sighan_2015/test.tsv')

    # count_positive_negative('./other/law_data.tsv')
    count_positive_negative('./other/medical_data.tsv')
