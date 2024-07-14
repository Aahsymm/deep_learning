import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),
                                     download=True)
data_loader=DataLoader(dataset, batch_size=64,drop_last=True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1=nn.MaxPool2d(3)
    def forward(self,x):
        return self.maxpool1(x)

tudui=Tudui()

writer=SummaryWriter("logs_nn")
step=0

for data in data_loader:
    imgs,targets=data
    outputs=tudui(imgs)
    print(outputs.size())
    writer.add_images("maxpool1",outputs,step)
    step=step+1


writer.close()


