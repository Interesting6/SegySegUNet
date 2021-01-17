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

class randomRotate(nn.Module):
    def __init__(self, degrees=[0, 90, 180, 270]):
        super(randomRotate, self).__init__()
        self.degrees = degrees

    def __call__(self, x):
        degree = random.choice(self.degrees)
        return transforms.functional.rotate(x, degree,)

class labelInverse(nn.Module):
    def __init__(self, num_class):
        super(labelInverse, self).__init__()
        self.num_class = num_class

    def __call__(self, x):  # ToTensor时已经除了255，只需还原到0-13即可
        x = x * self.num_class
        x = torch.round(x)
        return x


class Norm(nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, x, i):
        mean = self.mean[i]
        std = self.std[i]
        print("mean, std", mean, std)
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
        self.mapDataTo1()   # 先归一化到[0, 1]之间，并求均值方差，
        self.mapTo255()  # 并转为[0, 255]的uint 
        self.hw = self.data_cube.shape[1:] # 图片大小  [463, 951]

        self.train_seg = [0, 1, 2, 3]
        self.test_seg = [2]
        if train==True: # 训练数据和测试数据
            self.data_cube = self.data_cube[self.train_seg]
            self.label_cube = self.label_cube[self.train_seg]
        else: # test 
            self.data_cube = self.data_cube[self.test_seg]
            self.label_cube = self.label_cube[self.test_seg]

        c = 0
        if self.train:
            self.hwarange = [torch.arange(0, x-ptsize+c, 10) for x in self.hw]
        else:
            self.hwarange = [torch.arange(0, x-ptsize+c, ptsize) for x in self.hw]
        self.chwn = [self.data_cube.shape[0] ] + [len(x) for x in self.hwarange] # 切面数、h方向滑动个数、w方向滑动个数
        print(self.chwn)
        
        self.augment = transforms.Compose([
            transforms.ToPILImage(),  # 后续操作需要是PIL
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            randomRotate(),
            transforms.ToTensor(),    # [0, 1]tensor
            # nn.ReflectionPad2d((c,c,c,c))
        ])

        

    def get_data(self):
        self.data_path = os.path.join(self.data_dir, "F3Seis_IL190_490_Amplitude.segy") # [iline, xline, time]
        self.label_path = os.path.join(self.data_dir, "F3Seis_IL190_490_Label.segy")
        self.data_cube = np.transpose(segyio.cube(self.data_path), (0,2,1))  # 只是hw互换 [c,h,w]
        self.label_cube = np.transpose(segyio.cube(self.label_path), (0,2,1))

        self.labels = np.unique(self.label_cube)
        self.num_class = self.labels.max()
        self.label_inv = labelInverse(self.num_class) # 得到从PIL图片到标签[0,1,2,3,4]的映射


    def mapDataTo1(self):
        min_hw = self.data_cube.min(axis=(1,2))
        min_hw = np.expand_dims(min_hw, axis=(1, 2))
        self.data_cube = self.data_cube - min_hw
        max_hw = self.data_cube.max(axis=(1,2))
        max_hw = np.expand_dims(max_hw, axis=(1,2))
        self.data_cube = 1 - self.data_cube / max_hw        # [0-1]的float

        # 计算均值方差
        data_mean = self.data_cube.mean(axis=(1,2))
        data_std = self.data_cube.std(axis=(1,2))
        print(data_mean, data_std)
        self.norm = Norm(data_mean, data_std) # 因为要转PIL，所以还不能现在用。



    def mapTo255(self):  
        self.data_cube = (self.data_cube*255).astype("uint8") # 转为uint8
        # 标签也要映射成图像，最后记得映射回来
        # 因为标签的类别固定13个，有的切片不含某个标签，所以不能像上面一样处理
        self.label_cube = self.label_cube / self.num_class
        self.label_cube = (self.label_cube *255).astype("uint8")
        

    def get_train_item(self, index):
        grid_size = self.chwn[1] * self.chwn[2]
        i = index // grid_size
        # i = self.train_seg[i] if self.train else self.test_seg[i]
        temp = index % grid_size
        w = self.chwn[2]
        y = temp // w
        x = temp % w

        hy = self.hwarange[0][y]
        wx = self.hwarange[1][x]
        print(f"the box position is {i, hy, wx}")
        image = self.data_cube[i, hy:hy+self.ptsize, wx:wx+self.ptsize] # HW
        label = self.label_cube[i, hy:hy+self.ptsize, wx:wx+self.ptsize]
        
        
        image = self.augment(image)
        label = self.augment(label)
        segi = self.train_seg[i] if self.train else self.test_seg[i]
        image = self.norm(image, segi)
        label = self.label_inv(label)
        return image, label, hy, wx


    def get_test_item(index):
        image, label = None, None
        return image, label
        
    def __getitem__(self, index):
        # 根据index读取图片
        if self.train:
            image, label, _,_ = self.get_train_item(index)
            return image, label
        else:
            image, label, hy, wx = self.get_train_item(index)
            return image, label, hy, wx
        

    def __len__(self):
        # 返回训练集大小
        return np.prod(self.chwn)

    
if __name__ == "__main__":
    print("---------------dataset1----------------")
    data_dir = "/home/cym/Datasets/StData-12/F3_block/"
    dataset = F3DS(data_dir, ptsize=64, train=True)
    print("数据个数：", len(dataset))

    img, label = dataset[65]
    # print(img.mode)
    # img.save("./outs/img1.png")
    # label.save("./outs/label1.png")

    print(img.shape)
    print(img.min(), img.max(), "--min, max  |  mean, std:", img.mean(), img.std())
    # print(torch.unique(label))
    labelarr = np.array(label)
    print(np.unique(labelarr))

