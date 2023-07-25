import torch
import matplotlib.pyplot as plt
import torchvision as tv


BATCH_SIZE = 1000
test_data = tv.datasets.MNIST("data/", train=False, download=True, transform=tv.transforms.ToTensor())
test_data = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

SAVE_PATH = "saved_models/autoencoder_2023_07_25_22_54_06"
net = torch.load(SAVE_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    for idx, (x_batch, target) in enumerate(test_data):
        x_batch = torch.flatten(x_batch, 1).to(device)
        target = target.to(device)
        for _ in range(5): # 2 dimension result
            output = net.seq[_](x_batch)
            x_batch = output
        plt.scatter(output[:, 0], output[:, 1], c=target, s=3)

    plt.colorbar()
    plt.show()

    n = 10
    plt.figure(figsize=(20, 4))
    for idx, (x_batch, target) in enumerate(test_data):
        x_batch = torch.flatten(x_batch, 1).to(device)
        output = net(x_batch)
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            ax.imshow(x_batch[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 + n)

            ax.imshow(net(output[i]).reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()







