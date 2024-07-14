import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


#数据处理
data_transforms = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])

#数据下载读取
train_data=torchvision.datasets.CIFAR10(root='.\data', train=True,transform=transforms.ToTensor(), download=True)
test_data=torchvision.datasets.CIFAR10(root='.\data', train=False,transform=transforms.ToTensor(), download=True)


writer=SummaryWriter("logs")
for i in range(10):
    img,label=train_data[i]
    writer.add_image('haha15_02_.{}'.format(i),img,i)


writer.close()