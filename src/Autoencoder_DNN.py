import torch


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(input_shape, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),
            torch.nn.Linear(2, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, input_shape),
            torch.nn.LeakyReLU(),
        )

    def forward(self, x):

        x = self.seq(x)
        return x



"""
class Autoencoder(t.nn.Module):
    def __init__(self, input_shape=784):
        super().__init__()

        self.layer1 = t.nn.Linear(input_shape, 50)

        self.layer2 = t.nn.Linear(50, input_shape)
        self.loss = t.nn.BCELoss(reduction="sum")
        self.opti = t.optim.Adam(self.parameters(), lr=0.001)

    # Returns the output of the final layer, of the forward propogation
    def forward(self, x):
        self.train()

        x = t.relu(self.layer1(x))
        x = t.sigmoid(self.layer2(x))
        return x

    def evaluate(self, x, y):
        return self.loss(self.predict(x), y).item()
"""