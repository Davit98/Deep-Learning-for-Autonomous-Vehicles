from torch import nn, max, flatten
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, n_input_channels=3, n_output=10):
        super().__init__()

        self.conv1 = nn.Conv2d(n_input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(8 * 8 * 64, 256)
        self.fc2 = nn.Linear(256, n_output)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = F.leaky_relu(self.conv3(x), 0.1)
        x = F.leaky_relu(self.conv4(x), 0.1)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = flatten(x, 1)

        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        self.eval()
        outputs = self.forward(x)
        _, predicted = max(F.softmax(outputs, dim=1).data, 1)
        return predicted