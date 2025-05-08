import re
import warnings
import ebooklib
from collections import Counter
from ebooklib import epub
from bs4 import BeautifulSoup

def extract_text_from_epub(file_path):
    # 打开EPUB文件
    book = epub.read_epub(file_path)
    text_content = []

    # 遍历所有项目
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # 使用BeautifulSoup解析HTML内容
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            # 提取文本并添加到列表中
            text_content.append(soup.get_text())

    # 合并所有文本段落
    full_text = '\n'.join(text_content)
    return full_text

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

# 示例epub文件路径
input_filepath = './The_Brothers_Karamazov.epub'
output_filepath = 'vocabulary_list_Karamazov.md'

# 读取epub文件并生成单词本
text = extract_text_from_epub(input_filepath)
vocabulary_list = create_vocabulary_list(text)

# 保存单词本为markdown文件
save_vocabulary_to_markdown(vocabulary_list, output_filepath)

print(f'Vocabulary list saved to {output_filepath}')
