import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def generate_polygon(sides, radius):
    # 计算多边形的顶点坐标
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    polygon = np.column_stack((x, y))
    polygon = np.vstack((polygon, polygon[0]))  # 添加起始点，使多边形闭合
    return polygon

def update(val):
    # 更新按钮状态时的回调函数
    sides = int(sides_text.get_text())
    polygon = generate_polygon(sides, radius)
    line.set_xdata(polygon[:, 0])
    line.set_ydata(polygon[:, 1])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    fig.canvas.draw_idle()

def increase_sides(event):
    current_sides = int(sides_text.get_text())
    new_sides = min(20, current_sides + 1)
    sides_text.set_text(str(new_sides))
    update(None)

def decrease_sides(event):
    current_sides = int(sides_text.get_text())
    new_sides = max(3, current_sides - 1)
    sides_text.set_text(str(new_sides))
    update(None)

# 初始参数
initial_sides = 6
radius = 1.0

# 生成初始多边形
polygon = generate_polygon(initial_sides, radius)

# 创建图形和轴
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.2, right=0.8)

# 绘制多边形
line, = ax.plot(polygon[:, 0], polygon[:, 1], 'b-')
ax.scatter(polygon[:, 0], polygon[:, 1], color='red')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal', adjustable='box')
ax.set_title('Regular Polygon')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.grid(True)

# 创建增大和减小按钮
ax_increase = plt.axes([0.85, 0.4, 0.1, 0.05])
ax_decrease = plt.axes([0.85, 0.3, 0.1, 0.05])
button_increase = Button(ax_increase, '+')
button_decrease = Button(ax_decrease, '-')
button_increase.on_clicked(increase_sides)
button_decrease.on_clicked(decrease_sides)

# 显示多边形边数的文本
sides_text = plt.text(0.85, 0.2, str(initial_sides), transform=ax.transAxes, fontsize=12, ha='center')

# 显示图形
plt.show()
