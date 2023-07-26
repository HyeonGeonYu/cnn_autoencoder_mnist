import torch


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.seq1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,16,(3,3),padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Conv2d(16, 8, (3, 3), padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)))

        self.seq2 = torch.nn.Sequential(
            torch.nn.Linear(392,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2))

        self.seq3 = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 392))

        self.seq4= torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, (3, 3), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(8, 16, (3, 3), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(16, 1, (3, 3), padding='same'),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        bs = x.shape[0]
        x = self.seq1(x)
        x = x.reshape(bs,-1)
        x = self.seq2(x)
        x = self.seq3(x)
        x = x.reshape(bs, 8,7,7)
        x = self.seq4(x)
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