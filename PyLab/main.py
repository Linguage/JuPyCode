import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


dataset = {}
train_input, train_label = make_moons(
    n_samples=1000, shuffle=True, noise=0.1, random_state=None)
test_input, test_label = make_moons(
    n_samples=1000, shuffle=True, noise=0.1, random_state=None)

dataset['train_input'] = torch.from_numpy(train_input)
dataset['test_input'] = torch.from_numpy(test_input)
dataset['train_label'] = torch.from_numpy(train_label)
dataset['test_label'] = torch.from_numpy(test_label)

X = dataset['train_input']
y = dataset['train_label']
plt.scatter(X[:, 0], X[:, 1], c=y[:])
# Show the plot
plt.show()
plt.ion()

print("Hello,world!")
