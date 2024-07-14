import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2
#读取图片并将其转化成RGB模式，不同截图软件保留的维度不同
img_path="images/test_01.png"
img=Image.open(img_path)
img=img.convert('RGB')
print(type(img))
#totensor 的使用
tensor_trans=transforms.ToTensor()
img_tensor=tensor_trans(img)
print(img_tensor)
# img_array=cv2.imread(img_path)

#normlize的使用
tensor_norm=transforms.Normalize([2, 1, 0.5], [0.5, 0.5, 0.5])
img_norm=tensor_norm(img_tensor)

#resize的使用
tensor_resize=transforms.Resize((512,512))
img_resize=tensor_resize(img_tensor)
print(img_resize.shape)
#
#compose
tensor_resize2=transforms.Resize(512)
tensor_compose=transforms.Compose([tensor_resize2,tensor_norm])
img_compose=tensor_compose(img_tensor)
#compose
# PIL_resize=transforms.Resize(512)
# tensor_compose=transforms.Compose([PIL_resize, tensor_trans])




writer=SummaryWriter("logs")
writer.add_image("transform",img_tensor,1)
# writer.add_image("transform",img_array,2
writer.add_image("transform",img_norm,2)

writer.add_image("transform",img_resize,3)
writer.add_image("transform",img_compose,4)
writer.close()