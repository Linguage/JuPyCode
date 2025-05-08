
import requests
from bs4 import BeautifulSoup
import pandas as pd

# 登录信息
LOGIN_URL = 'https://zh.1lib.sk/login'
DOWNLOAD_HISTORY_URL = 'https://zh.1lib.sk/users/downloads'
USERNAME = 'peakhan618@gmail.com'
PASSWORD = 'google2010'

# 创建会话
session = requests.Session()

# 登录
login_payload = {
    'email': USERNAME,
    'password': PASSWORD
}
session.post(LOGIN_URL, data=login_payload)

# 抓取下载历史记录
def fetch_download_history(page):
    response = session.get(f"{DOWNLOAD_HISTORY_URL}?page={page}")
    soup = BeautifulSoup(response.text, 'html.parser')
    records = []
    
    # 解析下载记录
    for record in soup.select('.record-item'):
        time_element = record.select_one('.time')
        title_element = record.select_one('.title a')
        time_text = time_element.text.strip() if time_element else ''
        title_text = title_element.text.strip() if title_element else ''
        records.append({
            'Time': time_text,
            'Title': title_text
        })
    
    return records

# 获取所有页数的记录
all_records = []
for page in range(1, 11):  # 假设有10页
    records = fetch_download_history(page)
    if not records:
        break
    all_records.extend(records)

# 将记录保存到CSV文件
df = pd.DataFrame(all_records)
df.to_csv('download_history.csv', index=False)
print("Download history saved to download_history.csv")
