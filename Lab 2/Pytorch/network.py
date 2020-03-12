import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(n_input, 1000)
        self.fc2 = torch.nn.Linear(1000, 500)
        self.fc3 = torch.nn.Linear(500, n_output)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x

    def predict(self, x):
        outputs = self.forward(x)
        _, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
        return predicted
