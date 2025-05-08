import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 创建一个图形和轴
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25)  # 调整布局以容纳滑动条

# 生成 x 值
x = np.linspace(0, 2 * np.pi, 1000)

# 初始参数
amplitude_init = 1.0
frequency_init = 1.0
phase_init = 0.0

# 初始化一个空的正弦曲线
line, = ax.plot(x, amplitude_init * np.sin(frequency_init * x + phase_init))

# 设置图形的标题和标签
ax.set_title('Interactive Sine Wave')
ax.set_xlabel('x')
ax.set_ylabel('sin(ax + b)')

# 添加滑动条
axcolor = 'lightgoldenrodyellow'
ax_amplitude = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_frequency = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_phase = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)

s_amplitude = Slider(ax_amplitude, 'Amplitude', 0.1, 2.0, valinit=amplitude_init)
s_frequency = Slider(ax_frequency, 'Frequency', 0.1, 2.0, valinit=frequency_init)
s_phase = Slider(ax_phase, 'Phase', 0.0, 2*np.pi, valinit=phase_init)

# 更新函数，用于更新曲线的数据
def update(val):
    amplitude = s_amplitude.val
    frequency = s_frequency.val
    phase = s_phase.val
    line.set_ydata(amplitude * np.sin(frequency * x + phase))
    fig.canvas.draw_idle()

# 绑定更新函数到滑动条的事件
s_amplitude.on_changed(update)
s_frequency.on_changed(update)
s_phase.on_changed(update)

# 显示图形
plt.show()
