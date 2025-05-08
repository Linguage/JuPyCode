import pandas as pd
import math

# 读取 Excel 文件
file_path = '分数数据带学科名称.xlsx'
df = pd.read_excel(file_path)

# 分数段范围和步长
score_ranges = {
    '语文': (0, 151),
    '数学': (0, 151),
    '英语': (0, 151),
    '历史': (0, 101),
    '地理': (0, 101),
    '政治': (0, 101),
    '生物': (0, 101),
    '化学': (0, 101),
    '物理': (0, 101)
}

# 创建一个空的 Excel writer 对象
result_file_path = '分数分布统计结果.xlsx'
writer = pd.ExcelWriter(result_file_path, engine='xlsxwriter')

# 对每一门科目进行分段统计
for subject, scores in df.items():
    score_min, score_max = score_ranges[subject]  # 获取当前科目的最小和最大分数
    
    # 计算分数段范围
    score_step = 5  # 步长为5分
    score_range = [(i, i + score_step) for i in range(math.floor(score_min), math.ceil(score_max), score_step)]
    
    score_counts = {range_: 0 for range_ in score_range}  # 初始化分数段人数统计
    
    # 统计每个分数段的人数
    for score in scores:
        for range_ in score_range:
            if range_[0] <= score < range_[1]:
                score_counts[range_] += 1
                break
    
    # 计算每个分数段的百分比并保留两位小数
    total_students = len(scores)
    score_percentages = {range_: round(count / total_students * 100, 2) for range_, count in score_counts.items()}
    
    # 将统计结果添加到单独的 sheet 中
    result_df = pd.DataFrame({'分数段': [f'[{range_[0]}, {range_[1]})' for range_ in score_range],
                              '人数': list(score_counts.values()),
                              '百分比': list(score_percentages.values())})
    result_df.to_excel(writer, sheet_name=subject, index=False)

# 保存结果到 Excel 文件
writer.close()

## 区间平均分数
import pandas as pd

# 读取数据文件
file_path = '分数数据带学科名称.xlsx'
df = pd.read_excel(file_path)

# 创建Excel写入对象
writer = pd.ExcelWriter('分数分组平均值统计.xlsx', engine='xlsxwriter')

# 遍历每门学科
for subject in df.columns:
    # 按照每门学科的分数进行排序
    sorted_df = df.sort_values(by=subject)
    # 确定每组学生的数量
    group_size = len(df) // 100
    group_additional = len(df) % 100
    # 初始化分组平均值列表和人数列表
    group_avg_scores = []
    group_sizes = []
    # 对每组学生的分数进行处理
    for i in range(100):
        # 确定当前组的考生索引范围
        start_index = i * group_size + group_additional
        end_index = start_index + group_size if i < 99 else len(df)
        # 获取当前组的分数并计算平均值
        group_scores = sorted_df.iloc[start_index:end_index][subject]
        group_avg_score = group_scores.mean()
        group_avg_scores.append(round(group_avg_score, 2))
        group_sizes.append(len(group_scores))
    # 将分组平均值数据转换为DataFrame
    avg_scores_df = pd.DataFrame({'分数组': range(1, 101), '平均分数': group_avg_scores, '人数': group_sizes})
    # 创建工作表并写入数据
    avg_scores_df.to_excel(writer, sheet_name=subject, index=False)

# 保存Excel文件
writer.close()
