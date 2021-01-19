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



class F3DS(Dataset):
    def __init__(self, data_dir, train=True, ptsize=128, p=20):
        # 
        super(F3DS, self).__init__()
        self.train = train
        self.data_dir = data_dir
        self.ptsize = ptsize
        self.p = p # 一个维度上padding总数
        
        self.get_data()  # 先导入数据，并归一化到[0, 1]之间 [h,w,c]
        self.resize_and_norm()   # 调整尺度，并用均值标准化，
        self.hw = self.data_cube.shape[:2] # 图片大小  [h,w]

        self.train_seg = [0, 1, 3]
        self.test_seg = [2]
        if train==True: # 训练数据和测试数据
            self.data_cube_norm = self.data_cube_norm[..., self.train_seg]
            self.label_cube = self.label_cube[..., self.train_seg]
            self.hwarange = [np.arange(0, x-ptsize+1, 10) for x in self.hw]
        else: # test 
            self.data_cube_norm = self.data_cube_norm[..., self.test_seg]
            self.label_cube = self.label_cube[..., self.test_seg]
            self.hwarange = [np.arange(0, x-ptsize+1, ptsize) for x in self.hw]
        
        self.chw_gridnum = [self.data_cube_norm.shape[-1] ] + [len(x) for x in self.hwarange] # c切面数、h方向滑动个数、w方向滑动个数
        self.grid_size = self.chw_gridnum[1] * self.chw_gridnum[2]

        self.train_augment = A.Compose([
            A.PadIfNeeded(min_height=self.ptsize+p, min_width=self.ptsize+p, ),
            A.OneOf([
                A.RandomRotate90(),
                A.Rotate((45, 45)),
                A.Rotate((135,135)),
            ], p=1),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
        ])
        self.test_augment = A.PadIfNeeded(min_height=self.ptsize+p, min_width=self.ptsize+p)
        self.tsf = transforms.Compose([
            transforms.ToTensor(),
        ])


    def get_data(self):
        data_path = os.path.join(self.data_dir, "F3Seis_IL190_490_Amplitude.segy") # [iline, xline, time]
        label_path = os.path.join(self.data_dir, "F3Seis_IL190_490_Label.segy")
        data_cube = segyio.cube(data_path)  # [c,h,w,]
        data_cube = np.transpose(data_cube, (2, 1, 0))  #[h, w, c]
        # 将data映射到01之间
        min_hw = data_cube.min(axis=(0,1))
        data_cube = data_cube - min_hw
        max_hw = data_cube.max(axis=(0,1))
        self.data_cube = 1 - data_cube / max_hw     # [0-1]的float
        self.data_cube_img = self.data_cube

        label_cube = segyio.cube(label_path)
        self.label_cube = np.transpose(label_cube, (2, 1, 0))
        self.labels = np.unique(self.label_cube)
        self.num_class = self.labels.max()
        self.label_cube_img = self.label_cube


    def resize_and_norm(self):
        # 将data和label都resize到一个尺度
        rsz = A.Resize(512, 1024)
        rszed = rsz(image=self.data_cube, mask=self.label_cube)
        self.data_cube = rszed["image"]
        self.label_cube = rszed["mask"]

        # 计算均值方差，并标准化
        data_mean = self.data_cube.mean(axis=(0,1), keepdims=True)
        data_std = self.data_cube.std(axis=(0,1), keepdims=True)
        self.data_cube_norm = (self.data_cube - data_mean) / data_std


    def get_item(self, index):
        i = index // self.grid_size
        temp = index % self.grid_size
        w = self.chw_gridnum[2]
        y = temp // w
        x = temp % w

        hy = self.hwarange[0][y]
        wx = self.hwarange[1][x]
        # print(f"the box position is {i, hy, wx}")
        image = self.data_cube_norm[hy:hy+self.ptsize, wx:wx+self.ptsize, i] # HW
        label = self.label_cube[hy:hy+self.ptsize, wx:wx+self.ptsize, i]
        return image, label, hy, wx
        
        
    def __getitem__(self, index):
        # 根据index读取图片
        image, label, hy, wx = self.get_item(index)  # [128, 128, 1]
        if self.train:
            auged = self.train_augment(image=image, mask=label) 
            image, label = auged["image"], auged["mask"] # [148, 148, 1]
            image = self.tsf(image) # [1, 128, 128]
            label = self.tsf(label)
            return image, label
        else:
            image = self.test_augment(image=image)["image"]
            image = self.tsf(image)
            label = self.tsf(label)
            return image, label, hy, wx
        

    def __len__(self):
        # 返回训练集大小
        return np.prod(self.chw_gridnum)

    
if __name__ == "__main__":
    print("******************dataset2*********************8")
    data_dir = "/home/cym/Datasets/StData-12/F3_block/"
    dataset = F3DS(data_dir, ptsize=108, train=True)
    print("数据个数：", len(dataset), "grid数", dataset.chw_gridnum)

    img, label = dataset[65]

    print(img.shape)
    print(img.min(), img.max(), "--min, max  |  mean, std:", img.mean(), img.std())
    print(label.shape)
    # print(torch.unique(label))
    labelarr = np.array(label)
    print(np.unique(labelarr))


    dataset = F3DS(data_dir, ptsize=128, train=False)
    print("数据个数：", len(dataset))

    img, label,_,_ = dataset[25]

    print(img.shape)
    print(img.min(), img.max(), "--min, max  |  mean, std:", img.mean(), img.std())
    print(label.shape)
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

    

    

