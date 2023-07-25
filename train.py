import parameters
from src.Autoencoder import Encoder
import torchvision as tv
import torch
import os
from datetime import datetime
import parameters
args = parameters.para_config()

NUM_EPOCHS = args.epochs
SAVE_PATH = "saved_models/autoencoder_1_epoch.pt"
BATCH_SIZE = 1000


if __name__ == "__main__":
    if (not os.path.isdir("data")):
        os.mkdir("data/")
    if (not os.path.isdir("saved_models")):
        os.mkdir("saved_models/")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_data = tv.datasets.MNIST("data/", train=True, download=True, transform=tv.transforms.ToTensor())
    training_data = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = tv.datasets.MNIST("data/", train=False, download=True, transform=tv.transforms.ToTensor())
    test_data = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    net = Encoder(784).to(device)
    deepsc_optimizer = torch.optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-9)
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    now = datetime.now()

    # train
    net.train()

    range(1, args.epochs + 1)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.
        for idx, (x_batch, target) in enumerate(training_data):
            x_batch = torch.flatten(x_batch, 1).to(device)
            target = target.to(device)
            output = net(x_batch)
            loss = criterion(output,target)

            deepsc_optimizer.zero_grad()
            loss.backward()
            deepsc_optimizer.step()
            # model_list[2].seq[0].weight[0]
            total_loss += loss.detach().item()
            mean_loss = total_loss / (idx + 1)


            match = target == torch.argmax(output, axis=1)
            acc = sum(match)/len(match)
            print("\r" + now.strftime(
                "%Y-%m-%d %H:%M:%S - ") +
                  'Epoch: {};  '
                  'Loss: {:.3f}; '
                  'Accuracy: {:.3f}; '
                  'batchs: {:d}/{:d}'.format
                  (epoch,
                   mean_loss,
                   acc,
                   idx + 1,
                   len(training_data)),
                  end="")
        print("\r", end="")
        print(
            "Epoch : " + str(epoch) +
            "; loss : %.2f" % (mean_loss))



"""    
    error_sum = 0
    num = 0
    for (x_batch, _) in test_data:
        num += x_batch.shape[0]
        x_batch = torch..flatten(x_batch, 1).to(device)
        for x in x_batch:
            error_sum += net.evaluate(net(x), x)

    net.save(SAVE_PATH)
    print("Avg Cross-Entropy testing error: {}".format(error_sum/num))
    
"""