import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

# 创建一个图形和轴
fig, ax = plt.subplots()

# 生成 x 值
x = np.linspace(0, 2 * np.pi, 1000)

# 初始化一个空的正弦曲线
line, = ax.plot(x, np.sin(x))

# 设置图形的标题和标签
ax.set_title('Dynamic Sine Wave')
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')

# 定义更新函数，用于更新曲线的数据
def update(frame):
    # 在每一帧中更新曲线的数据
    line.set_ydata(np.sin(x + frame * 0.1))  # 在sin函数中引入时间变量来模拟波动
    return line,

# 创建动画对象
animation = FuncAnimation(fig, update, frames=range(100), interval=50)

# 保存动画为 GIF 图片
animation.save('sine_wave_animation.gif', writer='imagemagick', fps=30)

# 显示动画（可选）
plt.show()
