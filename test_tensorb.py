import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("logs")

img_path="dataset/train/ants_image/0013035.jpg"
img_PIL=Image.open(img_path)
img_array=np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
writer.add_image("test_img", img_array,1,dataformats='HWC')
# for i in range(100):
#     writer.add_scalar('y=x',i,i)
# writer.close()
#
# tensor_trans=transforms.ToTensor()
# tensor_img=tensor_trans(img_PIL)
#
# writer.add_image("tensor_img",tensor_img)

writer.close()