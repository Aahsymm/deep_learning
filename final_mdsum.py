import time

import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#1.读取数据并load
train_data=torchvision.datasets.CIFAR10(root='./data', train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root='./data', train=False,transform=torchvision.transforms.ToTensor(),download=True)
train_loader=DataLoader(train_data, batch_size=64)
test_loader=DataLoader(test_data, batch_size=64)
train_size=len(train_data)
test_size=len(test_data)
print("训练姐的长度为{}".format(train_size))
print("测试集的长度为{}".format(test_size))

#2.创建自己的十分类模型
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


#3.声明必要的模型和类
tudui=Tudui()
tudui=tudui.cuda()
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.cuda()
writer=SummaryWriter("log_finalmd")


#3.开始训练自己的模型
##必要的参数说明
total_train_step=0  #记录训练的次数
total_test_step=0  #记录测试的次数
epoch=10
start_time=time.time()
for i in range(epoch):
    learning_rate=1e-2
    optim = torch.optim.SGD(tudui.parameters(),lr=learning_rate)
    #训练步骤开始
    print("-------第{}轮训练开始-------".format(i+1))
    tudui.train()
    for data in train_loader:
        imgs,targets=data
        imgs=imgs.cuda()
        targets=targets.cuda()
        outputs=tudui(imgs)
        loss=loss_fn(outputs,targets)
        #优化器开始优化
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step+=1
        ####注意训练的次数（在一轮训练过程中要几次）和训练轮数（最外层--一共跑几轮）
        if total_train_step%100==0:
            end_time=time.time()
            print(end_time-start_time)
            print("训练次数:{}  Loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)


        #4.每训练完一轮，便在测试集中进行一次测试

        #测试步骤开始
        tudui.eval()
        total_test_loss=0
        total_accuracy=0
        with torch.no_grad():
            for data in test_loader:
                imgs,targets=data
                imgs=imgs.cuda()
                targets=targets.cuda()
                outputs=tudui(imgs)
                loss=loss_fn(outputs,targets)
                total_test_loss+=loss.item()
                accuracy=(outputs.argmax(1)==targets).sum()
                total_accuracy+=accuracy


    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的准确率:{}".format(total_accuracy/len(test_data)))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/len(test_data),total_test_step)
    total_test_step+=1
    torch.save(tudui,"tudui_{}.pth".format(i))
    print("模型已保存")


writer.close()


