import torch.nn as nn
import torch

class LeNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        in_size: int = 28
    ):
        super(LeNet, self).__init__()
        self.in_dim = in_dim
        self.in_size = in_size
        self.conv1 = nn.Conv2d(in_dim, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        hidden_indim = int(in_size/4 - 3) ** 2 * 16
        self.fc1 = nn.Linear(hidden_indim, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.relu5(x)
        return x

if __name__ == "__main__":
    model = LeNet(3, 32)
    data = torch.randn(10, 3, 32, 32)
    output = model(data) 
