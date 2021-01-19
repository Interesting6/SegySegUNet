import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn
from PIL import Image
import random
import segyio
import albumentations as A
from matplotlib import pyplot as plt
import cv2





class Norm(nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, x, i):
        mean = self.mean
        std = self.std
        # print("mean, std", mean, std)
        x.sub_(mean).div_(std)
        return x 



class F3DS(Dataset):
    def __init__(self, data_dir, train=True, ptsize=128):
        # 
        super(F3DS, self).__init__()
        self.train = train
        self.ptsize = ptsize
        self.data_dir = data_dir
        
        self.get_data()  # 
        self.mapDataTo1()   # 先归一化到[0, 1]之间，并用均值标准化，
        
        

        self.train_seg = [0, 1, 3]
        self.test_seg = [2]
        if train==True: # 训练数据和测试数据
            self.data_cube = self.data_cube[self.train_seg]
            self.label_cube = self.label_cube[self.train_seg]
        else: # test 
            self.data_cube = self.data_cube[self.test_seg]
            self.label_cube = self.label_cube[self.test_seg]
        
        self.hw = self.data_cube.shape[1:] # 图片大小  [463, 951,]
        
        c = 20
        if self.train:
            self.hwarange = [torch.arange(0, x-ptsize+1, 10) for x in self.hw]
        else:
            self.hwarange = [torch.arange(0, x-ptsize+1, ptsize) for x in self.hw]
        self.chwn = [self.data_cube.shape[0] ] + [len(x) for x in self.hwarange] # 切面数、h方向滑动个数、w方向滑动个数
        
        self.augment = A.Compose([
            A.PadIfNeeded(min_height=self.ptsize+c, min_width=self.ptsize+c, ),
            A.OneOf([
                A.RandomRotate90(),
                A.Rotate((45, 45)),
                A.Rotate((135,135)),
            ], p=1),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
        ])
        self.tsf = transforms.Compose([
            transforms.ToTensor(),
        ])

        

    def get_data(self):
        self.data_path = os.path.join(self.data_dir, "F3Seis_IL190_490_Amplitude.segy") # [iline, xline, time]
        self.label_path = os.path.join(self.data_dir, "F3Seis_IL190_490_Label.segy")
        self.data_cube = np.transpose(segyio.cube(self.data_path), (0,2,1))  # 只是hw互换 [c,h,w]
        self.label_cube = np.transpose(segyio.cube(self.label_path), (0,2,1))
        

        self.labels = np.unique(self.label_cube)
        self.num_class = self.labels.max()


    def mapDataTo1(self):
        min_hw = self.data_cube.min(axis=(1,2))
        min_hw = np.expand_dims(min_hw, axis=(1, 2))
        self.data_cube = self.data_cube - min_hw
        max_hw = self.data_cube.max(axis=(1,2))
        max_hw = np.expand_dims(max_hw, axis=(1,2))
        self.data_cube = 1 - self.data_cube / max_hw        # [0-1]的float


        self.data_cube = np.transpose(self.data_cube, (1,2,0))
        self.label_cube = np.transpose(self.label_cube, (1,2,0))
        rsz = A.Resize(512, 1024)
        rszed = rsz(image=self.data_cube, mask=self.label_cube)
        self.data_cube = np.transpose(rszed["image"], (2, 0,1))
        self.label_cube = np.transpose(rszed["mask"], (2, 0,1))

        self.data2 = self.data_cube[2]
        self.label2 = self.label_cube[2]

        # 计算均值方差，并标准化
        data_mean = self.data_cube.mean(axis=(1,2), keepdims=True)
        data_std = self.data_cube.std(axis=(1,2), keepdims=True)
        # print(data_mean, data_std)
        self.data_cube = (self.data_cube - data_mean) / data_std

        

    def get_item(self, index):
        grid_size = self.chwn[1] * self.chwn[2]
        i = index // grid_size
        temp = index % grid_size
        w = self.chwn[2]
        y = temp // w
        x = temp % w

        hy = self.hwarange[0][y]
        wx = self.hwarange[1][x]
        # print(f"the box position is {i, hy, wx}")
        image = self.data_cube[i, hy:hy+self.ptsize, wx:wx+self.ptsize] # HW
        label = self.label_cube[i, hy:hy+self.ptsize, wx:wx+self.ptsize]
        return image, label, hy, wx
        
        
    def __getitem__(self, index):
        # 根据index读取图片
        image, label, hy, wx = self.get_item(index)
        if self.train:
            auged = self.augment(image=image, mask=label)
            image, label = auged["image"], auged["mask"]
            image = self.tsf(image)
            label = self.tsf(label)
            return image, label
        else:
            image = self.tsf(image)
            label = self.tsf(label)
            return image, label, hy, wx
        

    def __len__(self):
        # 返回训练集大小
        return np.prod(self.chwn)

    
if __name__ == "__main__":
    print("******************dataset2*********************8")
    data_dir = "/home/cym/Datasets/StData-12/F3_block/"
    dataset = F3DS(data_dir, ptsize=108, train=True)
    print("数据个数：", len(dataset))

    img, label = dataset[65]

    print(img.shape)
    print(img.min(), img.max(), "--min, max  |  mean, std:", img.mean(), img.std())
    print(label.shape)
    print(label.min(), label.max(), "--min, max  |  mean, std:", label.mean(), label.std())
    # print(torch.unique(label))
    labelarr = np.array(label)
    print(np.unique(labelarr))


    dataset = F3DS(data_dir, ptsize=128, train=False)
    print("数据个数：", len(dataset))

    img, label,_,_ = dataset[25]

    print(img.shape)
    print(img.min(), img.max(), "--min, max  |  mean, std:", img.mean(), img.std())
    print(label.shape)
    print(label.min(), label.max(), "--min, max  |  mean, std:", label.mean(), label.std())
    # print(torch.unique(label))
    labelarr = np.array(label)
    print(np.unique(labelarr))


    # img = (img * 255).astype("uint8")
    # tpi = transforms.ToPILImage()
    # imgp = tpi(img)
    # tts = transforms.ToTensor()
    # imgt = tts(imgp)
    # print(imgt.shape)
    # print(imgt.min(), imgt.max(), "--min, max  |  mean, std:", imgt.mean(), imgt.std())

    

    

