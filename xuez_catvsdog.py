import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ReLU, Dropout, AdaptiveAvgPool2d
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# 创建模型
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 64, 3, padding=1),
            # ReLU(inplace=True),
            Conv2d(64, 64, 3, padding=1),
            # ReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(64, 128, 3, padding=1),
            # ReLU(inplace=True),
            Conv2d(128, 128, 3, padding=1),
            # ReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(128, 256, 3, padding=1),
            # ReLU(inplace=True),
            Conv2d(256, 256, 3, padding=1),
            # ReLU(inplace=True),
            Conv2d(256, 256, 3, padding=1),
            # ReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(256, 512, 3, padding=1),
            # ReLU(inplace=True),
            Conv2d(512, 512, 3, padding=1),
            # ReLU(inplace=True),
            Conv2d(512, 512, 3, padding=1),
            # ReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(512, 512, 3, padding=1),
            # ReLU(inplace=True),
            Conv2d(512, 512, 3, padding=1),
            # ReLU(inplace=True),
            Conv2d(512, 512, 3, padding=1),
            # ReLU(inplace=True),
            MaxPool2d(2),
        )
        self.model2 = Sequential(
            AdaptiveAvgPool2d(7),
            Flatten(),
            Linear(25088, 4096),
            Linear(4096, 2)
        )



    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x


# 创建读取图片的dataset
class MyData(Dataset):
    def __init__(self, dir_root, dir_label):
        self.dir_root = dir_root
        self.dir_label = dir_label
        self.path = os.path.join(self.dir_root, self.dir_label)
        self.image_path = os.listdir(self.path)
        self.trans = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img_name = self.image_path[idx]
        img_item_path = os.path.join(self.dir_root, self.dir_label, img_name)
        img = Image.open(img_item_path)
        img = self.trans(img)
        if self.dir_label == "Cat":
            label = 0
        else:
            label = 1
        return img, label

    def __len__(self):
        return len(self.image_path)


# 导入数据集
# dir_root_train = r"C:\Users\Timmy\Desktop\Pytorch project\learn pytorch\练手数据集\PetImages\train"
# dir_root_test = r"C:\Users\Timmy\Desktop\Pytorch project\learn pytorch\练手数据集\PetImages\val"
# dir_label_cat = "Cat"
# dir_label_dog = "Dog"
# dataset_train_cat = MyData(dir_root_train, dir_label_cat)
# dataset_train_dog = MyData(dir_root_train, dir_label_dog)
# dataset_test_cat = MyData(dir_root_test, dir_label_cat)
# dataset_test_dog = MyData(dir_root_test, dir_label_dog)
# dataset_train = dataset_train_cat + dataset_train_dog
# dataset_test = dataset_test_cat + dataset_test_dog

# 测试图片大小
# for i in range(100):
#     sample_img, sample_target = dataset_test[i]
#     print(i, sample_img.shape, sample_target)


dataset_train = torchvision.datasets.CIFAR10(root="./练手数据集/CIFAR10", train=True,
                                             transform=torchvision.transforms.ToTensor(), download=True)
dataset_test = torchvision.datasets.CIFAR10(root="./练手数据集/CIFAR10", train=False,
                                            transform=torchvision.transforms.ToTensor(), download=True)
# 查看长度
size_dataset_train = len(dataset_train)
size_dataset_test = len(dataset_test)
print("训练数据集的长度为：{}".format(size_dataset_train))
print("测试数据集的长度为：{}".format(size_dataset_test))

# 用DataLoader来加载数据集

dataloader_train = DataLoader(dataset_train, batch_size=16)
dataloader_test = DataLoader(dataset_test, batch_size=16)

# 添加tensorboard
writer = SummaryWriter("logs_train")

# 定义训练的设备
device = torch.device("cuda:0")

# 创建网络模型
_nn = NN()
_nn = _nn.to(device)

# 损失函数
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(_nn.parameters(), lr=learning_rate)

epoch = 60
cnt_train = 0

for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i+1))
    # 测试步骤
    _nn.eval()
    loss_test = 0
    cnt = 0
    accuracy = 0
    with torch.no_grad():
        for data in dataloader_test:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            _outputs = _nn.forward(imgs)
            loss = loss_function(_outputs, targets)
            loss_test += loss.item()
            accuracy += (_outputs.argmax(1) == targets).sum()
            if cnt == 0:
                print(_outputs)
            if cnt == 20:
                print(_outputs)
            cnt += 1
            # pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in _outputs]).to(device)
            # accuracy += pred.eq(targets.long()).sum().item()

    writer.add_scalar("loss_test", loss_test, i)
    accuracy_rate = accuracy / size_dataset_test
    print("测试集上的正确率:%{}".format(accuracy_rate * 100))
    # 训练步骤
    _nn.train()
    for data in dataloader_train:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        _outputs = _nn.forward(imgs)
        loss = loss_function(_outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt_train += 1
        if cnt_train % 100 == 0:
            writer.add_scalar("loss_train", loss.item(), cnt_train)

writer.close()
