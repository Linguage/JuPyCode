import pandas as pd

# 读取CSV文件
file_path = 'zlib_list.csv'
df = pd.read_csv(file_path)

# 拆分时间列为日期和时间两列
df['日期'] = pd.to_datetime(df['时间'], format='%d.%m.%Y %H:%M').dt.date
df['时间'] = pd.to_datetime(df['时间'], format='%d.%m.%Y %H:%M').dt.time

# 按日期和时间进行降序排列
df_sorted = df.sort_values(by=['日期', '时间'], ascending=False)

# 输出为新的CSV文件
output_path = 'zlib_list_sorted.csv'
df_sorted.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"Sorted CSV file has been saved to {output_path}")
