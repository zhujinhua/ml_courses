import torch

# torch help us to derivation, x0.data store the value, x0.grad store the gradient value
# print(torch.__version__)

x0 = torch.randint(low=-1000, high=1001, size=(1,), dtype=torch.float32, requires_grad=True)
y0 = torch.randint(low=-1000, high=1001, size=(1,), dtype=torch.float32, requires_grad=True)
z0 = torch.randint(low=-1000, high=1001, size=(1,), dtype=torch.float32, requires_grad=True)


def fn(x, y, z):
    return x ** 2 + y ** 2 + z ** 2


epochs = 10000
learning_rate = 1e-2
for _ in range(epochs):
    # Forward Propagation
    target = fn(x0, y0, z0)
    # Backward Propagation, taking the partial derivative
    target.backward()
    # gradient decrease
    x0.data -= learning_rate * x0.grad
    y0.data -= learning_rate * y0.grad
    z0.data -= learning_rate * z0.grad
    x0.grad.zero_()
    y0.grad.zero_()
    z0.grad.zero_()

print(x0, y0, z0)
