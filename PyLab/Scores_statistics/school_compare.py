import pandas as pd

# 读取学生信息和分数数据的 Excel 文件
student_info_df = pd.read_excel('学生信息.xlsx')  # 假设学生信息的索引是学生的唯一标识
score_data_df = pd.read_excel('分数数据带学科名称_分科参考.xlsx')  # 假设分数数据的索引也是学生的唯一标识


student_info_df.reset_index(drop=True, inplace=True)
score_data_df.reset_index(drop=True, inplace=True)

# 将两个表格按照行索引合并
merged_df = pd.concat([student_info_df, score_data_df], axis=1)

# 将合并后的数据写入一个新的 Excel 文件
merged_df.to_excel('成绩数据_新表.xlsx')

import pandas as pd

# 读取原始数据表格
file_path = '成绩数据_新表.xlsx'
df = pd.read_excel(file_path)

# 创建一个Excel写入器
writer = pd.ExcelWriter('学生排名的学校分布.xlsx', engine='xlsxwriter')

# 定义需要统计的前N名列表
N_values = [10, 100, 200, 400, 600, 800, 1000, 2000, 3000]

# 统计总分前N名的学校分布
total_score_counts = {}
for N in N_values:
    top_N_data = df.sort_values(by='所有9门课的成绩总分', ascending=False).head(N)
    total_score_counts[N] = top_N_data['学校'].value_counts()

# 将总分前N名的统计结果写入Excel的不同sheet中
total_score_df = pd.DataFrame(total_score_counts)
total_score_df.to_excel(writer, sheet_name='总分前N名')

# 统计语数英三门总分前N名的学校分布
language_math_english_counts = {}
for N in N_values:
    top_N_data = df.sort_values(by='语数英三门课的总分', ascending=False).head(N)
    language_math_english_counts[N] = top_N_data['学校'].value_counts()

# 将语数英三门总分前N名的统计结果写入Excel的不同sheet中
language_math_english_df = pd.DataFrame(language_math_english_counts)
language_math_english_df.to_excel(writer, sheet_name='语数英三门总分前N名')

# 统计各门课前N名的学校分布
subject_counts = {}
subjects = ['历史', '地理', '政治', '生物', '化学', '物理', '英语', '数学', '语文']
for subject in subjects:
    subject_counts[subject] = {}
    for N in N_values:
        top_N_data = df.sort_values(by=subject, ascending=False).head(N)
        subject_counts[subject][N] = top_N_data['学校'].value_counts()

# 将各门课前N名的统计结果写入Excel的不同sheet中
for subject, count_dict in subject_counts.items():
    subject_df = pd.DataFrame(count_dict)
    subject_df.to_excel(writer, sheet_name=f'{subject}前N名')

# 保存并关闭Excel写入器
writer.close()

import pandas as pd

# 读取原始数据表格
file_path = '成绩数据_新表.xlsx'
df = pd.read_excel(file_path)

# 函数用于计算各个学校的各个班级的不同类别成绩的前N名的平均分
def calculate_class_averages(df, N):
    school_class_averages = {}
    for school, school_data in df.groupby('学校'):
        school_class_averages[school] = {}
        for class_num, class_data in school_data.groupby('班级'):
            class_averages = {}
            # 总分的平均分
            top_N_total_scores = class_data['所有9门课的成绩总分'].nlargest(N)
            class_averages['总分'] = top_N_total_scores.mean()
            # 语数英三门总分的平均分
            top_N_language_math_english = class_data['语数英三门课的总分'].nlargest(N)
            class_averages['语数英三门总分'] = top_N_language_math_english.mean()
            # 各门课程的单科成绩的平均分
            for subject in ['历史', '地理', '政治', '生物', '化学', '物理', '英语', '数学', '语文']:
                top_N_subject_scores = class_data[subject].nlargest(N)
                class_averages[subject] = top_N_subject_scores.mean()
            # 统计被计算的人数
            class_averages['被统计人数'] = len(top_N_total_scores)
            # 将班级的平均分数据存储到字典中
            school_class_averages[school][class_num] = class_averages
    return school_class_averages

# 设置不同的N值
Ns = [1, 5, 10, 20, 30, 40]

# 创建一个 ExcelWriter 对象
excel_writer = pd.ExcelWriter('各班成绩平均分统计.xlsx', engine='xlsxwriter')

# 计算各班前N名的成绩统计
for N in Ns:
    # 计算各班前N名的成绩统计
    school_class_averages = calculate_class_averages(df, N)
    # 创建一个包含所有学校和班级的平均分数据的DataFrame
    all_class_averages = pd.DataFrame(columns=['学校', '班级', '总分', '语数英三门总分', '历史', '地理', '政治', '生物', '化学', '物理', '英语', '数学', '语文', '被统计人数'])
    for school, class_data in school_class_averages.items():
        for class_num, averages in class_data.items():
            new_row = pd.DataFrame({'学校': [school], '班级': [class_num], **averages})
            all_class_averages = pd.concat([all_class_averages, new_row], ignore_index=True)
    # 按照总分的平均分从大到小排序
    all_class_averages = all_class_averages.sort_values(by='总分', ascending=False)
    # 对所有列（除了 '学校' 和 '班级' 列）保留两位小数
    all_class_averages = all_class_averages.round(decimals=2)
    # 将统计结果写入Excel表格的不同sheet中
    all_class_averages.to_excel(excel_writer, sheet_name=f'前{N}名统计', index=False)

# 保存Excel文件
excel_writer.close()
