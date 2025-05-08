import re
from collections import Counter

def read_markdown_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def extract_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def word_frequency(words):
    return Counter(words)

def create_vocabulary_list(text):
    words = extract_words(text)
    frequency = word_frequency(words)
    vocabulary_list = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    return vocabulary_list

def save_vocabulary_to_markdown(vocabulary_list, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as file:
        file.write('| 单词 | 出现频率 |中文释义\n')
        file.write('|------|----------|-----|\n')
        for word, freq in vocabulary_list:
            file.write(f'| {word} | {freq} | |\n')

# 示例markdown文件路径
input_filepath = 'example.md'
output_filepath = 'vocabulary_list.md'

# 读取markdown文件并生成单词本
text = read_markdown_file(input_filepath)
vocabulary_list = create_vocabulary_list(text)

# 保存单词本为markdown文件
save_vocabulary_to_markdown(vocabulary_list, output_filepath)

print(f'Vocabulary list saved to {output_filepath}')
