# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
import pandas as pd
# todo:数学计算和显示
# fixme:
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
x = math.log(8,2)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
sum = 0
# for i in range(1,10):
#     sum = sum + i
#     x = math.log(sum,2)
#     print(x)

# TODO：写入csv
#任意的多组列表
a = [1,2,3]
b = [4,5,6]

#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'a_name':a,'b_name':b})

#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("test.csv",index=False,sep=',')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# TODO： 文件的读取

import csv
with open("test.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        print(line)


