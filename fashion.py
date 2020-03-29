import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import wandb

cud = False 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) # 28 + 1 - 5 = 24
        self.pool = nn.MaxPool2d(2, 2) # 24 /2 = 12
        self.conv2 = nn.Conv2d(32, 32, 5) # 12 + 1 - 5 = 8
        self.pool2 = nn.MaxPool2d(2, 2) # 8 / 2 = 4 
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.FashionMNIST('fashion_mnist_train', train=True, download=True, transform=transform)
    trainProportion = 0.8
    trainSize = int(len(dataset)*trainProportion)
    valSize = len(dataset) - trainSize
    trainset, valset = torch.utils.data.random_split(dataset, [trainSize, valSize])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=True)
    testset = datasets.FashionMNIST('fashion_mnist_test', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True)

    wandb.init(project="cs194-proj4")
    model = Net() 
    #wandb.watch(model)

    if cud:
        model = model.cuda()
    images, labels = next(iter(trainloader))
    if cud:
        images, labels = images.cuda(), labels.cuda()

    wandb.log({"inputs": [wandb.Image(i) for i in images]})

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0
        print("Starting epoch %d" % (epoch))
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if cud:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:    # every 1000 mini-batches...
                val_loss = 0
                for inputss, labelss in valloader:
                    if cud:
                        inputss, labelss = inputss.cuda(), labelss.cuda()
                    outputss = model(inputss)
                    val_loss += criterion(outputss, labelss).item()

                wandb.log({"step": epoch*len(trainloader) + i, "training loss": running_loss/1000, "validation loss": val_loss / len(valloader)})

                print("losses %f %f" % (val_loss/len(valloader), running_loss/1000))
                running_loss = 0.0
        torch.save(model.state_dict(), "model%d.model" % (epoch))
    print('Finished Training')

     

