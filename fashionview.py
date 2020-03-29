import torch, torchvision
from torch.utils.tensorboard import SummaryWriter
from fashion import Net
import wandb

wandb.init(project="cs194-proj4-visualization")

model = Net()
model.load_state_dict(torch.load("model4.model"))

weight = model.conv1.weight.data
wandb.log({"Filters": [wandb.Image(i) for i in weight]})
