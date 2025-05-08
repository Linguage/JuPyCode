import pandas as pd

# 读取 Excel 文件，跳过前两行合并的信息
file_path = 'Scores/高一多科全区成绩单.xlsx'
df = pd.read_excel(file_path, header=1, usecols=lambda x: 'Unnamed' not in x and x != '学校')

# 从第9列开始，每三列读取一次
df_scores = df.iloc[:, 7::3]

# 学科名称列表
subject_names = ["历史", "地理", "政治", "生物", "化学", "物理", "英语", "数学", "语文"]

# 创建DataFrame，指定列标签为学科名称
df_scores.columns = subject_names

# 保存DataFrame到Excel文件
df_scores.to_excel('分数数据带学科名称.xlsx', index=False)

import pandas as pd

# 读取现有的 Excel 文件
file_path = '分数数据带学科名称.xlsx'
df = pd.read_excel(file_path)

# 计算第一组数据：所有9门课的成绩总分
total_scores_all_subjects = df.sum(axis=1)

# 计算第二组数据：语文、数学、英语三门课的总分
total_scores_3_subjects = df[['语文', '数学', '英语']].sum(axis=1)

# 计算第三组数据：语文、数学、英语三门课的分数加上历史和物理中分数更高的那一门分数
df['历史/物理高分'] = df[['历史', '物理']].max(axis=1)
total_scores_3_subjects_with_max = df[['语文', '数学', '英语']].sum(axis=1) + df['历史/物理高分']

total_scores_3_subjects_sets1 = df[['语文', '数学', '英语']].sum(axis=1) + df[['物理', '化学', '生物']].sum(axis=1)
total_scores_3_subjects_sets2 = df[['语文', '数学', '英语']].sum(axis=1) + df[['物理', '化学', '地理']].sum(axis=1)

# 插入数据到DataFrame中
df.insert(0, '所有9门课的成绩总分', total_scores_all_subjects)
df.insert(1, '语数英三门课的总分', total_scores_3_subjects)
df.insert(2, '语数英三门课的总分加上历史和物理中高分的那门课', total_scores_3_subjects_with_max)
df.insert(3, '三科+物化生', total_scores_3_subjects_sets1)
df.insert(4, '三科+物化地', total_scores_3_subjects_sets2)

# 保存DataFrame到Excel文件
df.to_excel('分数数据带学科名称_分科参考.xlsx', index=False)

## 平均分统计
import pandas as pd

# 读取之前处理得到的数据文件
file_path = '分数数据带学科名称_分科参考.xlsx'
df = pd.read_excel(file_path)

# 计算各项指标
max_scores = df.max()
top_10_avg = df.apply(lambda col: col.nlargest(10).mean())
top_100_avg = df.apply(lambda col: col.nlargest(100).mean())
top_1000_avg = df.apply(lambda col: col.nlargest(1000).mean())
mean_scores = df.mean()

# 保留两位小数
max_scores = max_scores.round(2)
top_10_avg = top_10_avg.round(2)
top_100_avg = top_100_avg.round(2)
top_1000_avg = top_1000_avg.round(2)
mean_scores = mean_scores.round(2)

# 创建新的DataFrame
summary_df = pd.DataFrame({
    '最高分': max_scores,
    '前10名平均分': top_10_avg,
    '前100名平均分': top_100_avg,
    '前1000名平均分': top_1000_avg,
    '平均分': mean_scores
})

# 创建Excel写入器
excel_writer = pd.ExcelWriter('整体成绩汇总表.xlsx', engine='xlsxwriter')

# 将数据写入Excel文件的第一页
summary_df.to_excel(excel_writer, index=True, sheet_name='成绩汇总')

# 关闭Excel写入器
excel_writer.close()


