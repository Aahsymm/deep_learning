import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F




#定义自己的数据集
image_extentions = [".jpg", ".png", ".JPG", ".PNG"]
class ImageDataset(Dataset):
    def __init__(self, images_folder, transform=None):
        images = []
        labels = []
        for dirname in os.listdir(images_folder):
            for filename in os.listdir(images_folder + "/" + dirname):
                if any(filename.endswith(extension) for extension in image_extentions):
                    images.append((dirname + '\\' + filename, int(dirname)))
        self.transform = transform
        self.images_folder = images_folder
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename, label = self.images[index]
        img = Image.open(os.path.join(self.images_folder, filename)).convert('RGB')
        img = self.transform(img)
        return img, label


# 声明类别和文件的路径
num_categroies = 2
images_folder_train = r"C:\Users\Wuu\PycharmProjects\02_15\catvsdog\train"
images_folder_test = r"C:\Users\Wuu\PycharmProjects\02_15\catvsdog\test"

# 构建pipeline 对图片数据进行预处理
pipeline = torchvision.transforms.Compose([

    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0], std=[0.229, 0.224, 0.225])
])


# 读取数据
train_data = ImageDataset(images_folder_train,transform=pipeline)
test_data = ImageDataset(images_folder_test, transform=pipeline)
# 打印数据,以免出错
# print(len(dog_train_data))
# print(len(cat_train_data))
# train_size = len(train_data)
# test_size = len(test_data)
# print(train_size)
train_size=train_data.__len__()
test_size=test_data.__len__()

# train_data = pipeline(train_data)
# test_data = pipeline(test_data)
# dataloader处理数据
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=0)


# 构建训练模型
class CatVsDog(nn.Module):
    def __init__(self):
        super().__init__()
        super(CatVsDog, self).__init__()
        # 卷积层1：输入是224*224*3 计算(224-5)/1+1=220 即通过Conv1输出的结果是220
        self.conv1 = nn.Conv2d(3, 6, 5)  # input:3 output6 kernel:5
        # 池化层：输入是220*220*6 窗口2*2  计算(220-0)/2=110 那么通过max_pooling层输出的是110*110*6
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2， 输入是220*220*6，计算（110 - 5）/ 1 + 1 = 106，那么通过conv2输出的结果是106*106*16
        self.conv2 = nn.Conv2d(6, 16, 5)  # input:6, output:16, kernel:5
        # 全连接层1
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)  # input:16*53*53, output:1024
        # 全连接层2
        self.fc2 = nn.Linear(1024, 512)  # input:1024, output:512
        # 全连接层3
        self.fc3 = nn.Linear(512, 2)  # input:512, output:2
        # dropout 层
        self.dropout = nn.Dropout(p=0.2)  # 因为数据较少，丢掉20%的神经元，防止过拟合

    def forward(self, x):
        # 卷积1
        """
        224x224x3 --> 110x110x6 -->106x106*6
        """
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积2
        """
        106x106x6 --> 53x53x16 
        """
        x = self.pool(F.relu(self.conv2(x)))
        # 改变shape
        x = x.view(-1, 16 * 53 * 53)
        # 全连接层1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 全连接层2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # 全连接层3
        x = F.relu(self.fc3(x))
        # print("x size", x.shape)  # x size torch.Size([16, 2])
        return x


catvsdog = CatVsDog()

# 构造损失函数和优化函数
loss_fn = nn.CrossEntropyLoss()
LEARNING_RATE = 1e-3
optimizer = optim.SGD(catvsdog.parameters(), lr=LEARNING_RATE, momentum=0.9)

#开始训练
EPOCH=20
for i in range(EPOCH):
    total_train_step=0
    total_train_loss=0
    print("--------Epoch:{}/{}----------".format(i + 1, EPOCH))
    for data in train_dataloader:
        imgs,targets = data
        outputs = catvsdog(pipeline(imgs))
        train_loss=loss_fn(outputs,targets)
        total_train_loss =total_train_loss + train_loss
        total_train_step = total_train_step+1
        if total_train_step%100==0:
            print("第{}次训练：Loss:{}".format(total_train_step,total_train_loss))
        #进行梯度优化
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


    #进行测试，with torch.no_grad()
    with torch.no_grad():
        total_test_step=0
        total_test_loss=0
        accurate_num=0
        for data in test_dataloader:
            imgs,targets = data
            outputs = catvsdog(pipeline(imgs))
            test_loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss + test_loss
            accurate_num=(outputs.argmax(1)==targets).sum()
            accuracy=accurate_num/test_size
            total_test_step =total_test_step+1
            if total_test_step%100==0:
                print("第{}次测试  Test Loss:{} 准确率为{}".format(total_test_step,total_test_loss,accuracy))
#......





