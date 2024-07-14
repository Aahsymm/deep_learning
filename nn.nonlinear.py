import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#读取数据
test_data=torchvision.datasets.CIFAR10("./data", train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

#创建非线性变换网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1=nn.ReLU()
        self.relu2=nn.Sigmoid()
    def forward(self,input):
        output=self.relu2(self.relu1(input))
        return output


tudui=Tudui()
writer=SummaryWriter("logs")
step=0
for data in test_loader:
    imgs,targets=data
    outputs=tudui(imgs)
    # print(outputs.shape)
    writer.add_images("pre",imgs,step)
    writer.add_images("relu",outputs,step)
    step=step+1

writer.close()