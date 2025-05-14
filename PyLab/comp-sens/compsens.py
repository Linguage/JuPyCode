import numpy as np
from scipy.fftpack import dct, idct # 以 DCT 为例
from sklearn.linear_model import OrthogonalMatchingPursuit

# 1. 定义/生成稀疏信号
N = 256  # 信号长度
K = 20   # 稀疏度 (非零元素个数)

# 生成一个在 DCT 域稀疏的信号
x_orig = np.zeros(N)
t = np.linspace(0, 1, N)
x_orig += np.sin(13 * 2 * np.pi * t)
x_orig += np.sin(37 * 2 * np.pi * t)
# 在实际中，信号可能更复杂，稀疏性可能不完美

# 找到其 DCT 变换 (alpha)
alpha = dct(x_orig, norm='ortho')
# (可选) 人为制造稀疏性以简化示例
# indices = np.random.choice(N, K, replace=False)
# alpha_sparse = np.zeros(N)
# alpha_sparse[indices] = alpha[indices] # 或者直接赋予一些值
# x_sparse_domain = idct(alpha_sparse, norm='ortho') # 这是理想的稀疏信号

# 2. 构建测量矩阵
M = 64  # 测量次数， M < N
Phi = np.random.randn(M, N)
# (可选) 对 Phi 进行正交化或归一化处理，但这不总是必需的

# 3. 执行压缩感知测量
# 假设我们直接测量原始信号 x，并且重建时考虑其在 DCT 域的稀疏性
# 那么传感矩阵 Theta = Phi * Psi_inv (这里 Psi_inv 是 IDCT 矩阵)
# 或者，如果我们知道信号在某个域是稀疏的，我们测量的是原始信号 x
y = np.dot(Phi, x_orig)

# 4. 实现重建算法 (使用 scikit-learn 的 OMP)
# OMP 试图找到 y = Phi @ x_reconstructed 中的 x_reconstructed
# 如果我们知道 x_reconstructed 在 DCT 域是稀疏的，
# 则令 x_reconstructed = IDCT(alpha_reconstructed)
# 那么 y = Phi @ IDCT(alpha_reconstructed)
# 我们可以定义 A = Phi @ IDCT_matrix，然后求解 y = A @ alpha_reconstructed
# 或者，一些 OMP 实现可以直接处理字典

# 更直接的方法：许多 CS 公式假设 y = Theta * alpha，其中 Theta = Phi * Psi
# Psi 是稀疏基矩阵 (例如 DCT 矩阵的列是基向量)
Psi = dct(np.eye(N), norm='ortho') # DCT 矩阵 (每一列是一个基向量)
Theta = np.dot(Phi, Psi.T) # Psi.T 因为我们用 alpha 做列向量 x = Psi.T @ alpha
                          # 或者 Psi 的每一行是一个基向量，x_transformed = Psi @ x_original
                          # 这里我们假设 x_orig = Psi_inv @ alpha, Psi_inv 是 IDCT
                          # 所以 alpha = Psi @ x_orig.
                          # y = Phi @ x_orig = Phi @ Psi_inv @ alpha
                          # 令 Theta = Phi @ Psi_inv
                          # Psi_inv 是 IDCT 矩阵，其列是 DCT 基函数。
                          # 更简单地，scikit-learn 的 OMP 可以配置为在变换域寻找解。

# 使用 scikit-learn OMP
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=K) # 假设已知稀疏度 K

# 我们需要找到 alpha_reconstructed 使得 y ≈ Phi @ idct(alpha_reconstructed)
# 或者，如果重建 x, 且 x 在 Psi 域稀疏。
# sklearn 的 OMP 求解 y = D_omp @ coefs， 我们需要设定 D_omp 和要恢复的 coefs
# 如果我们要恢复 alpha，那么 D_omp = Phi @ Psi_inv。
# 如果我们要直接恢复 x，并且约束 x 在 Psi 域是稀疏的，则更复杂。

# 让我们尝试直接恢复 x，并期望恢复的 x 具有稀疏的 DCT 变换。
# 这不是 OMP 的标准用法，OMP 通常用于 y = D_omp @ sparse_coeffs
# 为了正确使用 OMP 来恢复 alpha:
# Theta = Phi @ idct_matrix (idct_matrix 的列是 DCT 基函数)
# idct_matrix = Psi.T (因为 Psi 的行是 DCT 基函数)
idct_matrix = idct(np.eye(N), norm='ortho').T # 列是 DCT 基向量
Theta_omp = np.dot(Phi, idct_matrix)

omp.fit(Theta_omp, y)
alpha_reconstructed = omp.coef_

# 5. 信号恢复
x_reconstructed = idct(alpha_reconstructed, norm='ortho')

# 评估重建效果
mse = np.mean((x_orig - x_reconstructed)**2)
print(f"Mean Squared Error: {mse}")

# 可以绘制原始信号和重建信号进行比较
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(t, x_orig, label='Original Signal')
plt.plot(t, x_reconstructed, '--', label='Reconstructed Signal')
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.stem(alpha, use_line_collection=True, label='Original DCT Coefficients')
plt.title('Original DCT')
plt.subplot(1,2,2)
plt.stem(alpha_reconstructed, use_line_collection=True, label='Reconstructed DCT Coefficients')
plt.title('Reconstructed DCT (via OMP)')
plt.tight_layout()
plt.show()