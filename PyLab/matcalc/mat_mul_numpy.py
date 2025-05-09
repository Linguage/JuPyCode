import numpy as np
import time

# 定义矩阵大小
N = 2000

# 初始化矩阵 a 和 b
a = np.zeros((N, N))
b = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        a[i, j] = i + j + 0.5
        b[i, j] = i + j - 0.5

# 记录开始时间
start_time = time.time()

# 使用 NumPy 的点积函数进行矩阵乘法
c = np.dot(a, b)

# 记录结束时间
end_time = time.time()

# 计算耗时
elapsed_time = end_time - start_time

# 输出运行时间
print(f"Time taken by CPU: {elapsed_time:.6f} seconds")
