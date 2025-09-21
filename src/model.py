import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(input_shape, 512)
        self.norm1 = nn.BatchNorm1d(512)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(512, 256)
        self.norm2 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(256, 128)
        self.norm3 = nn.BatchNorm1d(128)
        self.act3 = nn.ReLU()
        self.lin4 = nn.Linear(128, output_shape)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.norm1(self.lin1(x)))
        x = self.act2(self.norm2(self.lin2(x)))
        x = self.act3(self.norm3(self.lin3(x)))
        logits = self.lin4(x)
        return logits
