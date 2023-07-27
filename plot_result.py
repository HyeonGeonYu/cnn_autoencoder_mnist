import torch
import matplotlib.pyplot as plt
import torchvision as tv


BATCH_SIZE = 1000
test_data = tv.datasets.MNIST("data/", train=False, download=True, transform=tv.transforms.ToTensor())
test_data = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

SAVE_PATH = "saved_models/autoencoder_2023_07_26_19_37_55"
test_scheme = "dnn"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(SAVE_PATH).to(device)

with torch.no_grad():


    ### test manifold
    for idx, (x_batch, target) in enumerate(test_data):
        x_batch = x_batch.to(device)
        target = target
        if test_scheme == "DNN" or "DAE":
            x_batch = torch.flatten(x_batch, 1) #For DNN
            for _ in range(5):  # 2 dimension result
                output = net.seq[_](x_batch)
                x_batch = output
        elif test_scheme == "CNN":
            bs = x_batch.shape[0]
            x_batch = net.seq1(x_batch)
            x_batch = x_batch.reshape(bs, -1)
            x_batch = net.seq2(x_batch)
            output = x_batch
        x_data = output.cpu().numpy()[:,0]
        y_data = output.cpu().numpy()[:,1]
        plt.scatter(x_data, y_data, c=target, s=3)

    plt.colorbar()
    plt.show()


    ### test re-generation
    n = 10
    plt.figure(figsize=(20*0.9, 4*0.9))
    for idx, (x_batch, target) in enumerate(test_data):
        """
        x_batch = x_batch.to(device)
        output = net(x_batch)
        """
        x_batch = x_batch.to(device)
        x_batch_noisy = x_batch + torch.randn(x_batch.shape).to(device)
        output = net(x_batch_noisy)

        for i in range(n):
            """
            ax = plt.subplot(2, n, i + 1)
            plt.gray()
            ax.imshow(x_batch[i].reshape(28, 28).cpu())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 +  n)
            plt.gray()
            ax.imshow(output[i].reshape(28, 28).cpu())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


            """
            ax = plt.subplot(3, n, i + 1)
            plt.gray()
            ax.imshow(x_batch[i].reshape(28,28).cpu())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            ax = plt.subplot(3, n, i + 1 + n)
            plt.gray()
            ax.imshow(x_batch_noisy[i].reshape(28,28).cpu())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            ax = plt.subplot(3, n, i + 1 + 2*n)
            plt.gray()
            ax.imshow(output[i].reshape(28, 28).cpu())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()
        break







