import pandas as pd

# 读取 Excel 文件，跳过第一行合并的信息
file_path = 'Scores/高一多科全区成绩单.xlsx'
df = pd.read_excel(file_path, header=1)

# 提取学生基本信息的列
student_info_columns = ['学校', '考号', '姓名', '班级']
df_student_info = df[student_info_columns]

# 保存DataFrame到Excel文件
df_student_info.to_excel('学生信息.xlsx', index=False)

# 打印前几个学生的基本信息
print(df_student_info.head())

# 提取各学校的考生人数，并按人数从多到少排序
school_counts = df['学校'].value_counts().sort_values(ascending=False)

# 创建 Excel Writer 对象
excel_writer = pd.ExcelWriter('学校和班级人数统计.xlsx', engine='xlsxwriter')

# 将各学校的考生人数写入第一页
school_counts.to_excel(excel_writer, sheet_name='各学校考生人数', index_label='学校')

# 提取每个学校各班级的人数，并按班级序号排列，写入后续页
for school in school_counts.index:
    df_school = df[df['学校'] == school]
    class_counts = df_school['班级'].value_counts().sort_index()
    class_counts.to_excel(excel_writer, sheet_name=school, index_label='班级')

# 保存 Excel 文件
excel_writer.close()

## 第二步：提取考生成绩

