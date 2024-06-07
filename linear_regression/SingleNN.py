import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple two-layer neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the network, loss function, and optimizer
simple_net = SimpleNN()
mse_criterion = nn.MSELoss()
optimizer = optim.SGD(simple_net.parameters(), lr=0.01)

# Generate some random data
inputs = torch.randn(10)
target = torch.randn(1)

# Forward propagation
outputs = simple_net(inputs)
loss = mse_criterion(outputs, target)

# Print the results of forward propagation
print("Outputs:", outputs)
print("Loss:", loss.item())

# Backward propagation
optimizer.zero_grad()  # Clear the gradients
loss.backward()  # Backward propagation to compute gradients
optimizer.step()  # Update the parameters

# Print the gradients
for param in simple_net.parameters():
    print("Gradient:", param.grad)
