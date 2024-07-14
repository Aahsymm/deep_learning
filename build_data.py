import os

import torch
from PIL import Image
from torch.utils.data import DataLoader,Dataset

num_categroies=2
images_folder_train=r"C:\Users\Wuu\PycharmProjects\02_15\catvsdog\train"
images_folder_val=r"C:\Users\Wuu\PycharmProjects\02_15\catvsdog\test"

image_extentions=[".jpg",".png",".JPG",".PNG"]
class ImageDataset(Dataset):
    def __init__(self,images_folder,transform=None):
        images=[]
        labels=[]
        for dirname in os.listdir(images_folder):
            for filename in os.listdir(images_folder+"/"+dirname):
                if any(filename.endswith(extension) for extension in image_extentions):
                    images.append((dirname+'\\'+filename,int(dirname)))
        self.transform = transform
        self.images_folder=images_folder
        self.images=images

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        filename,label=self.images[index]
        img=Image.open(os.path.join(self.images_folder,filename)).convert('RGB')
        img=self.transform(img)
        return img,label
