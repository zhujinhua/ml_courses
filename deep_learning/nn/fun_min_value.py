import numpy as np


def fn(x):
    return 2 * x ** 2


def d_fun(x):
    return 4 * x


x0 = np.random.randint(low=-100, high=1001)
epochs = 10000
learn_rate = 1e-2
for _ in range(epochs):
    x0 -= learn_rate * d_fun(x0)  # each time minus tiny positive number

print(x0 < 1e-6)


def fn_x(x, y, z):
    return 2 * x


def fn_y(x, y, z):
    return 2 * y


def fn_z(x, y, z):
    return 2 * z


x1 = np.random.randint(low=-100, high=1001)
y1 = np.random.randint(low=-100, high=1001)
z1 = np.random.randint(low=-100, high=1001)
for _ in range(epochs):
    x1 -= learn_rate * fn_x(x1, y1, z1)
    y1 -= learn_rate * fn_y(x1, y1, z1)
    z1 -= learn_rate * fn_z(x1, y1, z1)

print(x1, y1, z1)