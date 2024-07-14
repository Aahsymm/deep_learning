import torch
import torchvision
from torch import nn
from torch.autograd._functions import tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#训练的数据
dataset=torchvision.datasets.CIFAR10('.\data',train=False,download=True,transform=torchvision.transforms.ToTensor())
data_loader=DataLoader(dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.conv1=nn.Conv2d(3,6,3)

    def forward(self,input):
        output=self.conv1(input)
        return output
tudui=Tudui()
print(tudui)

writer=SummaryWriter("logs")

step=0
for data in data_loader:
    imgs,targets=data
    outputs=tudui(imgs)

    print(imgs.size())
    print(outputs.size())
    outputs=torch.reshape(outputs,(-1,3,30,30))

    writer.add_images("input",imgs,step)
    writer.add_images("output",outputs,step)
    step=step+1


writer.close()