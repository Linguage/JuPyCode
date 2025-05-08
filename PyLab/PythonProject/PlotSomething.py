import numpy as np
import matplotlib.pyplot as plt

# 生成 x 值，例如从 0 到 2*pi，可以使用 numpy 的 linspace 函数
x = np.linspace(0, 2 * np.pi, 1000)

# 计算对应的正弦值
y = np.sin(x)

# 使用 Matplotlib 绘制正弦曲线
plt.plot(x, y)

# 添加标题和标签
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')

# 显示图形
plt.show()