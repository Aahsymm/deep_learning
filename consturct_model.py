import torch
from torch import nn, tensor


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()


    def forward(self,input):
        output=input+1
        return output

tudui=Tudui()
test=torch.tensor(1,dtype=torch.float32)
test=tudui(test)
print(test)