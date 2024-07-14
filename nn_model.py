from torch import nn
from torch.nn import Conv2d, Sequential, Linear


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

