import time

import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data=torchvision.datasets.CIFAR10(".\data",train=False,download=True,transform=torchvision.transforms.ToTensor())
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1=Sequential(
            Conv2d(3,32,5,stride=1,padding=2),
            nn.MaxPool2d(2),
            Conv2d(32,32,5,stride=1,padding=2),
            nn.MaxPool2d(2),
            Conv2d(32,64,5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self,input):
        return self.model1(input)

tudui=Tudui()
loss=nn.CrossEntropyLoss()
# optim=torch.optim.SGD(tudui.parameters(),lr=0.01)
for epoch in range(200):
    running_loss=0.0
    start_time=time.time()
    if epoch<=150:
        optim=torch.optim.SGD(tudui.parameters(),lr=0.1)
    else:
        optim=torch.optim.SGD(tudui.parameters(),lr=0.01)
    for data in test_loader:
        imgs,targets=data
        outputs=tudui(imgs)
        result_loss=loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    end_time=time.time()
    duration=end_time-start_time
    print(running_loss)
    print(duration)




