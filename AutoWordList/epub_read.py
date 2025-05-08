import ebooklib
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

# 示例：提取EPUB文件的文本内容
epub_file_path = './The_Brothers_Karamazov.epub'
extracted_text = extract_text_from_epub(epub_file_path)
print(extracted_text)
