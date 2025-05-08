import os
import numpy as np
import matplotlib.pyplot as plt

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.txt')

data = np.loadtxt(data_path)
x = data[:,0]
u = data[:,1]

plt.plot(x, u, label='PDE Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Heat Equation Solution (Fortran->C++->Python)')
plt.legend()
plt.grid()
plt.show()