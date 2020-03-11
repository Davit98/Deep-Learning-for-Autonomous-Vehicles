import torch
# import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_hidden_layers):
        super(Net, self).__init__()
        # Option 1
        #         self.fc1 = torch.nn.Linear(3072, 1000)
        #         self.fc2 = torch.nn.Linear(1000, 500)
        #         self.fc3 = torch.nn.Linear(500, n_output)

        # Option 2
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(n_feature, n_hidden)])
        self.fc_layers.extend([torch.nn.Linear(n_hidden, n_hidden) for i in range(0, n_hidden_layers)])
        self.fc_layers.append(torch.nn.Linear(n_hidden, n_output))

        # Initialize weights
        for layer in self.fc_layers:
            layer.weight.data.normal_(std=0.01)
            layer.bias.data.fill_(0)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # Option 1
        #         x = self.fc1(x)
        #         x = F.relu(x)

        #         x = self.fc2(x)
        #         x = F.relu(x)

        #         x = self.fc3(x)

        # Option 2
        for layer in self.fc_layers[:-1]:
            x = layer(x)
            x = torch.nn.functional.relu(x)

        x = self.fc_layers[-1](x)  # We do not need relu here
        return x

    def predict(self, x):
        outputs = self.forward(x)
        _, predicted = torch.max(torch.nn.functional.softmax(outputs, dim=1).data, 1)
        return predicted
