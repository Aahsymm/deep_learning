import torchvision
from torch.utils.data import dataloader, DataLoader
from torch.utils.tensorboard import SummaryWriter

#导入数据
train_set=torchvision.datasets.CIFAR10('.\data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_set=torchvision.datasets.CIFAR10('.\data', train=False, transform=torchvision.transforms.ToTensor(), download=True)


test_loader=DataLoader(test_set,batch_size=16,shuffle=False,num_workers=0,drop_last=False)
writer=SummaryWriter('logs_DL')


for epoch in range(2):
    step=0
    for dldata in test_loader:
        imgs,targets=dldata
        writer.add_images('Epoch_{}'.format(epoch),imgs,step)
        step=step+1

writer.close()