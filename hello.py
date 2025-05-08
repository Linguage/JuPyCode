import numpy as np

# 定义两个3x3的矩阵A和B
matrix_A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

# 矩阵加法
matrix_sum = matrix_A + matrix_B
print("Matrix Addition:")
print(matrix_sum)

# 矩阵减法
matrix_diff = matrix_A - matrix_B
print("\nMatrix Subtraction:")
print(matrix_diff)

# 矩阵乘法
matrix_product = np.dot(matrix_A, matrix_B)
print("\nMatrix Multiplication:")
print(matrix_product)

# 检查矩阵A是否可逆
if np.linalg.det(matrix_A) != 0:
    # 矩阵求逆
    matrix_inverse = np.linalg.inv(matrix_A)
    print("\nInverse of Matrix A:")
    print(matrix_inverse)
else:
    print("\nMatrix A is not invertible.")

# 计算矩阵A的转置
matrix_transpose = matrix_A.T
print("\nTranspose of Matrix A:")
print(matrix_transpose)
