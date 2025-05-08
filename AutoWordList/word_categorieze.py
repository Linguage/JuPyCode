import re

def read_vocabulary_markdown(filepath):
    vocabulary = []
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines[2:]:  # 跳过表头
            match = re.match(r'\|\s*(\w+)\s*\|\s*(\d+)\s*\|', line)
            if match:
                word = match.group(1)
                frequency = int(match.group(2))
                vocabulary.append((word, frequency))
    return vocabulary

def categorize_vocabulary(vocabulary):
    categories = {
        "200以上": [],
        "101-200": [],
        "11-100": [],
        "10以下": {}
    }
    
    for word, freq in vocabulary:
        if freq > 200:
            categories["200以上"].append((word, freq))
        elif 101 <= freq <= 200:
            categories["101-200"].append((word, freq))
        elif 11 <= freq <= 100:
            categories["11-100"].append((word, freq))
        else:
            if freq not in categories["10以下"]:
                categories["10以下"][freq] = []
            categories["10以下"][freq].append(word)
    
    return categories

def format_words_list(words, show_frequency=True):
    if show_frequency:
        return '  '.join([f'{word}({freq})' for word, freq in words])
    else:
        return '  '.join([word for word, freq in words])

def save_categorized_vocabulary(categories, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as file:
        file.write('| 词频范围 | 单词(词频) |\n')
        file.write('|----------|-------------|\n')

        file.write('| 200以上 | ' + format_words_list(categories["200以上"]) + ' |\n')
        
        file.write('| 101-200 | ' + format_words_list(categories["101-200"]) + ' |\n')
        
        for i in range(101, 11, -10):
            words_in_range = [(word, freq) for word, freq in categories["11-100"] if i-10 <= freq <= i-1]
            if words_in_range:
                file.write(f'| {i-1}-{i-10} | ' + format_words_list(words_in_range) + ' |\n')
        
        for freq in sorted(categories["10以下"].keys(), reverse=True):
            words_in_range = [(word, freq) for word in categories["10以下"][freq]]
            if words_in_range:
                file.write(f'| {freq} | ' + format_words_list(words_in_range, show_frequency=False) + ' |\n')

# 示例Markdown文件路径
input_filepath = 'vocabulary_list_Karamazov.md'
output_filepath = 'categorized_vocabulary_list_Karamazov.md'

# 读取vocabulary_list.md文件并分类
vocabulary = read_vocabulary_markdown(input_filepath)
categories = categorize_vocabulary(vocabulary)

# 保存分类后的单词列表为新的Markdown文件
save_categorized_vocabulary(categories, output_filepath)

print(f'Categorized vocabulary list saved to {output_filepath}')
