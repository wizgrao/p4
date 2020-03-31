import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import wandb
import numpy as np

cud = True 

classes = ["T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
]

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


def accuracy(output, target):
    pred = output.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.view_as(pred)).sum()
    return correct.double() / len(pred) 

def per_class_accuracy(inputs, output, target):
    inputs = [i for i in inputs]
    total_class = np.zeros((10))
    correct_class = np.zeros((10))
    pred = output.data.max(1, keepdim=True)[1]
    target = target.view((-1)).tolist()
    pred = pred.view((-1)).tolist()
    corr = []
    incorr = []
    for t, p, i in zip(target, pred, inputs):
        total_class[t] += 1.0
        if t == p:
            correct_class += 1.0
            corr += [wandb.Image(i, caption="Predicted:%s Actual:%s"%(classes[p], classes[t]))]
        else:
            incorr += [wandb.Image(i, caption="Predicted:%s Actual:%s"%(classes[p], classes[t]))]
    return total_class, correct_class, corr, incorr

def evaluate_accuracies(model, loader):
    total_class = np.zeros((10))
    correct_class = np.zeros((10))
    corr = []
    incorr = []
    for inputss, labelss in loader:
        if cud:
            inputss, labelss = inputss.cuda(), labelss.cuda()
        outputss = model(inputss)
        new_total, new_correct, new_corr, new_incorr = per_class_accuracy(inputss, outputss, labelss)
        total_class += new_total
        correct_class += new_correct
        corr += new_corr
        incorr += new_incorr
    return corr, incorr, correct_class/total_class


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.FashionMNIST('fashion_mnist_train', train=True, download=True, transform=transform)
    trainProportion = 0.8
    trainSize = int(len(dataset)*trainProportion)
    valSize = len(dataset) - trainSize
    trainset, valset = torch.utils.data.random_split(dataset, [trainSize, valSize])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    testset = datasets.FashionMNIST('fashion_mnist_test', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    wandb.init(project="cs194-proj4")
    model = Net() 

    if cud:
        model = model.cuda()
    images, labels = next(iter(trainloader))
    if cud:
        images, labels = images.cuda(), labels.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    wandb.watch(model)
    wandb.log({"inputs": [wandb.Image(i) for i in images]})
    for epoch in range(3):  # loop over the dataset multiple times
        running_loss = 0
        running_accuracy = 0
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
            running_accuracy += accuracy(outputs, labels).item()
            running_loss += loss.item()
            every = 100
            if i % every == every - 1:
                val_loss = 0
                val_accuracy = 0
                for inputss, labelss in valloader:
                    if cud:
                        inputss, labelss = inputss.cuda(), labelss.cuda()
                    outputss = model(inputss)
                    val_loss += criterion(outputss, labelss).item()
                    val_accuracy += accuracy(outputss, labelss).item()

                wandb.log({"step": epoch*len(trainloader) + i, "train accuracy": running_accuracy/every, "val accuracy": val_accuracy / len(valloader), "training loss": running_loss/every, "validation loss": val_loss / len(valloader)})

                print("losses %f %f" % (val_loss/len(valloader), running_loss/every))
                running_loss = 0.0
                running_accuracy = 0.0
        torch.save(model.state_dict(), "model%d.model" % (epoch))
    valcorr, valincorr, valper = evaluate_accuracies(model, valloader)
    testcorr, testincorr, testper = evaluate_accuracies(model, testloader)
    wandb.log({"Validation Correct": valcorr, "Validation Incorrect": valincorr, "Validation Accuracies": valper})
    wandb.log({"Test Correct": testcorr, "Test Incorrect": testincorr, "Test Accuracies": testper})
    weight = model.conv1.weight.data
    wandb.log({"Filters": [wandb.Image(i) for i in weight]})

    print('Finished Training')

     

